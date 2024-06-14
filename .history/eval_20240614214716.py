import torch
from model import ViTForImageClassification


def eval(testloader, model=None):
    # 评估模型
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        try:
            model = ViTForImageClassification.from_pretrained("vit-base-patch16-224")
            model.load_state_dict(torch.load("last_ckpt.pth"))
        except:
            raise ValueError("Please provide a model to evaluate.")

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy of the network on CIFAR100 test images: {accuracy:.2f} %")
