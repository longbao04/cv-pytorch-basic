"""PyTorch 入门：用简单 CNN 识别 MNIST 手写数字。"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 保存和加载都使用脚本旁的路径，避免工作目录不同导致找不到模型。
MODEL_PATH = Path(__file__).resolve().parent / "mnist_cnn.pth"


def build_transform():
    """训练和预测共用相同的张量转换与归一化操作。"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


class SimpleCNN(nn.Module):
    """输入为单通道 28×28 图片，输出为数字 0～9 的分类分数。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 卷积提取局部特征；padding=1 保持图片宽高不变。
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # 池化将宽高减半：28×28 → 14×14。
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14×14 → 7×7。
            nn.Flatten(),  # 将每张图片的特征展开为一维向量。
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        # 返回原始分类分数（logits），交叉熵损失内部会处理 softmax。
        return self.layers(images)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for batch_index, (images, labels) in enumerate(loader, start=1):
        # 图片、标签和模型必须放在同一个设备上。
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # 清空上一次计算的梯度。
        scores = model(images)  # 前向传播：预测各类别的分数。
        loss = criterion(scores, labels)
        loss.backward()  # 反向传播：计算参数的梯度。
        optimizer.step()  # 根据梯度更新模型参数。

        if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
            print(
                f"Epoch 1/1 | Batch {batch_index}/{len(loader)} "
                f"| Loss: {loss.item():.4f}",
                flush=True,
            )


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    # 测试时不计算梯度，减少内存占用和计算开销。
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    torch.manual_seed(42)
    # Apple Silicon Mac 优先使用 MPS；不可用时自动使用 CPU。
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备：{device}", flush=True)

    # 将灰度图片转为 [1, 28, 28] 张量，并将像素值从 [0, 1] 归一化到 [-1, 1]。
    transform = build_transform()
    # 数据保存在脚本旁的 data 目录；已有数据时不会重复下载。
    data_dir = Path(__file__).resolve().parent / "data"
    train_dataset = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    # num_workers=0 避免 Mac 新手遇到多进程数据加载问题。
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()  # 多分类任务常用的交叉熵损失。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 一个 epoch 表示完整遍历一次训练集。
    train_one_epoch(model, train_loader, criterion, optimizer, device)
    accuracy = evaluate(model, test_loader, device)
    print(f"测试集 Accuracy：{accuracy:.2%}")

    # 只保存模型参数；预测时先创建相同结构，再加载这些参数。
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型参数已保存到：{MODEL_PATH}")


if __name__ == "__main__":
    main()
