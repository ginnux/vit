import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle

def main():
    att = np.load("figure/att_example_1.npy")
    mask_list = []

    # 加载并预处理CIFAR-100数据集
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # ViT期望的输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=8
    )

    images, _ = next(iter(testloader))
    image = images.squeeze(dim=0)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = ((image_np + 1) / 2 * 255).astype(np.uint8)
    im = image_np

    att = torch.Tensor(att)

    for i in range(att.size(0)):
        att_mat = att[i]

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(
                aug_att_mat[n], joint_attentions[n - 1]
            )

        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), im.shape[:2])[..., np.newaxis]
        mask_list.append(mask)

        with open("figure/temp.pkl", "wb") as f:
            pickle.dump(mask_list, f)
    return mask_list



def show(mask_list = None):
    plt.rcParams["font.size"] = 13
    plt.rcParams["font.family"] = "Times New Roman"

    if mask_list is None:
        with open("temp.pkl", "rb") as f:
            mask_list = pickle.load(f)
    for i in range(len(mask_list)):
        mask = mask_list[i]
        plt.subplot(4, 3, i+1)
        plt.axis("off")
        plt.imshow(mask, cmap="gray")
        plt.title(f"Head NO.{i+1}")

    plt.savefig("figure/att.pdf")   

if __name__ ==  "__main__":
    main()
    # 在安装了Times new Roman字体的情况下，可以使用下面的设置
    # show()
