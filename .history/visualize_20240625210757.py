import torch
import torchvision
from model import ViTForImageClassification
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from itertools import islice
import json
import torch.nn as nn


def visualize(num, testloader=None, model=None, load=None):

    # 可视化矩阵
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 导入相关模型
    if model is None:
        try:
            model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                output_attentions=True,
                attn_implementation="eager",
            )
            model.classifier = nn.Linear(model.classifier.in_features, 100)
            model.load_state_dict(torch.load("pth/base.pth"))
            model.to(device)
        except:
            raise ValueError("Please provide a model to evaluate.")
    else:
        try:
            model = ViTForImageClassification.from_pretrained(
                model, output_attentions=True, attn_implementation="eager"
            )
            model.classifier = nn.Linear(model.classifier.in_features, 100)
            model.load_state_dict(torch.load(load))
            model.to(device)
        except:
            raise ValueError("Please provide a model to evaluate.")

    if testloader is None:
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

    # 推理与可视化
    labels_name = testset.classes
    with torch.no_grad():
        i = 0
        im_list = []
        mask_list = []
        result_list = []
        label_list = []

        # 仅可视化num个样本
        for data in iter(testloader):
            i += 1
            if i > num:
                break

            # 模型推理
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits.data, 1)

            predicted_label_name = labels_name[predicted]
            label_name = labels_name[labels]

            # 预测结果
            print(
                f"{'Correct' if label_name == predicted_label_name else 'Error'}. Predict [{label_name}] as [{predicted_label_name}]."
            )
            label_list.append(label_name)

            # 可视化注意力矩阵
            att_mat = outputs.attentions
            att_mat = torch.stack(att_mat).squeeze(1).cpu()
            image = images.squeeze(dim=0)

            image_np = image.permute(1, 2, 0).cpu().numpy()

            image_np = ((image_np + 1) / 2 * 255).astype(np.uint8)

            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            im = image_np
            # cv2.imwrite("tempim.png",image_bgr)

            # 将所有注意力矩阵保存为npy文件，以便后续分析
            all_att_mat_sample = att_mat.cpu().detach().numpy()
            np.save(f"figure/att_example_{i}.npy", all_att_mat_sample)

            # 可视化注意力矩阵将所有头的注意力取平均
            att_mat = torch.mean(att_mat, dim=1)

            # 为了解释残差连接，将一个单位矩阵加到注意力矩阵上，然后重新归一化权重。
            residual_att = torch.eye(att_mat.size(1))
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # 递归乘注意力矩阵，直到最后一个注意力矩阵
            joint_attentions = torch.zeros(aug_att_mat.size())
            joint_attentions[0] = aug_att_mat[0]

            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(
                    aug_att_mat[n], joint_attentions[n - 1]
                )

            # 求解从token到输入空间的注意力
            v = joint_attentions[-1]
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))
            mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask = cv2.resize(mask / mask.max(), im.shape[:2])[..., np.newaxis]
            result = (mask * im).astype("uint8")

            # 返回可视化结果
            im_list.append(im)
            mask_list.append(mask)
            result_list.append(result)

    return im_list, mask_list, result_list, label_list


# 展示可视化结果
def show(im_list, mask_list, result_list, label_list, pickle=False):
    if pickle:
        import pickle

        with open("figure/visualize.pkl", "wb") as f:
            pickle.dump((im_list, mask_list, result_list, label_list), f)
    else:
        plt.rcParams["font.size"] = 10
        num = len(im_list)
        plt.figure(figsize=(12, 4 * num))

        for i in range(num):
            im = im_list[i]
            mask = mask_list[i]
            result = result_list[i]
            label_name = label_list[i]

            plt.subplot(num, 3, 3 * i + 1)
            plt.title(f"NO.{i+1}:{label_name}")
            plt.imshow(im)

            plt.subplot(num, 3, 3 * i + 2)
            plt.title("Attention")
            plt.imshow(mask, cmap="gray")

            plt.subplot(num, 3, 3 * i + 3)
            plt.title("Attention Map")
            plt.imshow(result)

        plt.savefig(f"figure/visualize.pdf")


if __name__ == "__main__":
    im_list, mask_list, result_list, label_list = visualize(5)
    show(im_list, mask_list, result_list, label_list, pickle=True)
