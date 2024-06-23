import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
from eval import eval
from config import get_config
import torch.optim.lr_scheduler as lr_scheduler

config = get_config()


# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载并预处理CIFAR-100数据集
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # ViT期望的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=8
)

testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=8
)

# 3. 定义ViT模型

# from transformers import ViTForImageClassification, ViTFeatureExtractor
from model import ViTForImageClassification

# 加载预训练的 ViT 模型和特征提取器

# 1k版本
if config["pretrained_model"] == "vit-base-patch16-224":
    model = ViTForImageClassification.from_pretrained("vit-base-patch16-224")
    model.classifier = nn.Linear(model.classifier.in_features, 100)
# 21k版本
else:
    model = ViTForImageClassification.from_pretrained(
        config["pretrained_model"], num_labels=100
    )

if config["load_last_checkpoint"]:
    model.load_state_dict(torch.load("last_ckpt.pth"))


# weights = ViT_B_16_Weights.DEFAULT
# model = vit_b_16(weights=weights)
# model.heads[0] = nn.Linear(model.heads[0].in_features, 100)  # 修改分类头为100类

# 如果有可用的GPU，则将模型转到GPU
model.to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# 5. 训练模型
for epoch in range(config["epochs"]):  # 遍历数据集多次
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader), 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        

        running_loss += loss.item()
        if i % 200 == 199:  # 每200个批次打印一次
            with open("log.txt","a") as f:
                f.write(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}\n")
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
            running_loss = 0.0
            # 保存checkpoints
            torch.save(model.state_dict(), "./last_ckpt.pth")
    # 更新学习率
    scheduler.step()

    eval(testloader, model)

print("Finished Training")
