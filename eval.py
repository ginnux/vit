import torch
import torchvision
from model import ViTForImageClassification
import torchvision.transforms as transforms
from tqdm import tqdm


def eval(testloader=None, model=None):
    # 评估模型
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        try:
            model = ViTForImageClassification.from_pretrained("vit-base-patch16-224-in21k", num_labels=100)
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
        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                 shuffle=False, num_workers=8)

    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy of the network on CIFAR100 test images: {accuracy:.2%}")

if __name__ == "__main__":
    eval()