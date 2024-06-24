import torch
import torchvision
from model import ViTForImageClassification
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import numpy as np
from itertools import islice
import json

def inference(num, testloader = None, model = None):
    outputs_list = []
    data_list = []
    # 可视化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        try:
            model = ViTForImageClassification.from_pretrained("vit-base-patch16-224-in21k", num_labels=100, output_attentions = True, attn_implementation="eager")
            model.load_state_dict(torch.load("last_ckpt.pth"))
            model.to(device)
        except:
            raise ValueError("Please provide a model to evaluate.")
    if testloader is None:
        # 加载并预处理CIFAR-100数据集
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT期望的输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                 shuffle=False, num_workers=8)

    labels_name = testset.classes
    with torch.no_grad():
        for i in range(num):
            data = next(iter(testloader))
            data_list.append(data)
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs_list.append(outputs)

    return outputs_list, data_list



def visualize_single(output, data, idx = 0):
    with torch.no_grad():
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        _, predicted = torch.max(outputs.logits.data, 1)

        predicted_label_name = labels_name[predicted]
        label_name = labels_name[labels]
        
        print(f"{'Correct' if label_name == predicted_label_name else 'Error'}. Predict [{label_name}] as [{predicted_label_name}].")

        att_mat = outputs.attentions
        att_mat = torch.stack(att_mat).squeeze(1).cpu()
        image = images.squeeze(dim=0)

        image_np = image.permute(1, 2, 0).cpu().numpy()

        image_np = ((image_np+1)/2 * 255).astype(np.uint8)


        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        im = image_np
        # cv2.imwrite("tempim.png",image_bgr)


        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
            
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), im.shape[:2])[..., np.newaxis]
        result = (mask * im).astype("uint8")

    return im, mask, result

def visualize(outputs_list, data_list):
    im_list = []
    mask_list = []
    result_list = []
    for i in range(len(outputs_list)):
        outputs = outputs_list[i]
        data = data_list[i]
        im, mask, result = visualize_single(outputs, data)
        im_list.append(im)
        mask_list.append(mask)
        result_list.append(result)

    return im_list, mask_list, result_list



def show(im_list, mask_list, result_list):
    num = len(im_list)
    plt.figure(figsize = (12,4*num))

    for i in range(num):
        im = im_list[i]
        mask = mask_list[i]
        result = result_list[i]

        plt.subplot(num,3,3*i+1)
        plt.title('Original')
        plt.imshow(im)

        plt.subplot(num,3,3*i+2)
        plt.title("Attention")
        plt.imshow(mask, cmap='gray')

        plt.subplot(num,3,3*i+3)
        plt.title('Attention Map')
        plt.imshow(result)

    plt.savefig(f"figure/visualize.png")




if __name__ == "__main__":

    num = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained("google/vit-large-patch16-224", output_attentions = True, attn_implementation="eager")
    model.classifier = nn.Linear(model.classifier.in_features, 100)
    model.load_state_dict(torch.load("pth/large.pth"))
    model.to(device)

    outputs, data = inference(num = num, model = model)

    im_list, mask_list, result_list = visualize(outputs, data)

    show(im_list, mask_list, result_list)


    