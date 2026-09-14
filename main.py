"""PyTorch 入门：用简单 CNN 识别 MNIST 手写数字。"""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 保存和加载都使用脚本旁的路径，避免工作目录不同导致找不到模型。
HISTORY_PATH = Path(__file__).resolve().parent / "training_history.csv"


def get_model_path(model_name):
    """不同结构使用不同参数文件，避免训练时互相覆盖。"""
    return Path(__file__).resolve().parent / f"mnist_cnn_{model_name}.pth"


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


class DeeperCNN(nn.Module):
    """每次池化前使用两层卷积，尝试提取更丰富的数字特征。"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # padding=1 保持宽高；第二层卷积继续加工第一层提取的特征。
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28×28 → 14×14。
            # 通道数增加到 32，随后再次卷积、池化。
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14×14 → 7×7。
            nn.Flatten(),  # 每张图片展开成 32×7×7 个特征。
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 输出数字 0～9 的分类分数。
        )

    def forward(self, images):
        return self.layers(images)


def build_model(model_name):
    """训练和预测共用模型选择逻辑，确保加载参数时结构一致。"""
    if model_name == "simple":
        return SimpleCNN()
    if model_name == "deeper":
        return DeeperCNN()
    raise ValueError(f"不支持的模型：{model_name}，请选择 simple 或 deeper。")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    total_loss, total_samples = 0.0, 0
    for batch_index, (images, labels) in enumerate(loader, start=1):
        # 图片、标签和模型必须放在同一个设备上。
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # 清空上一次计算的梯度。
        scores = model(images)  # 前向传播：预测各类别的分数。
        loss = criterion(scores, labels)
        loss.backward()  # 反向传播：计算参数的梯度。
        optimizer.step()  # 根据梯度更新模型参数。

        # loss 是当前 batch 的平均值，乘以图片数后累加。
        # 最后除以总图片数，让不足一个 batch 的最后一批也得到正确权重。
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

        if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(loader):
            print(
                f"Epoch {epoch}/{epochs} | Batch {batch_index}/{len(loader)} "
                f"| Loss: {loss.item():.4f}",
                flush=True,
            )
    return total_loss / total_samples


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
    # 命令行可指定训练轮数；不传 --epochs 时默认训练 1 轮。
    parser = argparse.ArgumentParser(description="训练 MNIST 手写数字分类模型")
    # choices 限定可选结构；不传 --model 时沿用原来的简单 CNN。
    parser.add_argument("--model", choices=["simple", "deeper"], default="simple",
                        help="模型结构（默认：simple）")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（默认：1）")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs 必须是大于或等于 1 的整数")

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

    model = build_model(args.model).to(device)
    model_path = get_model_path(args.model)
    print(f"使用模型：{args.model}", flush=True)
    criterion = nn.CrossEntropyLoss()  # 多分类任务常用的交叉熵损失。
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 一个 epoch 表示完整遍历一次训练集。
    # 每轮训练后测试，并写入 CSV；再次训练会覆盖上一次的记录。
    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file, fieldnames=["epoch", "avg_train_loss", "test_accuracy"]
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            avg_train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, args.epochs
            )
            accuracy = evaluate(model, test_loader, device)
            # 准确率保存为 0～1 的小数，终端显示时转换为百分比。
            writer.writerow({
                "epoch": epoch,
                "avg_train_loss": avg_train_loss,
                "test_accuracy": accuracy,
            })
            history_file.flush()  # 每轮立即保存，便于查看已完成的训练记录。
            print(
                f"Epoch {epoch}/{args.epochs} | Average train loss: {avg_train_loss:.4f} "
                f"| 测试集 Accuracy：{accuracy:.2%}",
                flush=True,
            )
    print(f"训练记录已保存到：{HISTORY_PATH}")

    # 只保存模型参数；预测时先创建相同结构，再加载这些参数。
    torch.save(model.state_dict(), model_path)
    print(f"模型参数已保存到：{model_path}")


if __name__ == "__main__":
    main()
