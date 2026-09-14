"""统计 MNIST 测试集的混淆矩阵，并显示最常见的预测错误。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from main import build_model, build_transform, get_model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 与训练脚本使用相同的模型名称，默认选择 simple。
    parser.add_argument("--model", choices=["simple", "deeper"], default="simple",
                        help="模型结构（默认：simple）")
    args = parser.parse_args()
    model_path = get_model_path(args.model)

    # 导入 main.py 不会触发训练；这里仅加载已有的模型参数。
    if not model_path.is_file():
        parser.error(
            f"找不到模型参数：{model_path}。"
            f"请先运行 python main.py --model {args.model} 训练并保存模型。"
        )

    # Apple Silicon Mac 优先使用 MPS，不可用时自动使用 CPU。
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备：{device}")
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式，不更新模型参数。

    # 复用训练时的预处理；已有数据会复用，缺少时只下载 MNIST。
    data_dir = Path(__file__).resolve().parent / "data"
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=build_transform()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    # 行对应真实标签，列对应预测标签；整数矩阵记录每种组合的图片数量。
    matrix = torch.zeros((10, 10), dtype=torch.int64)
    with torch.no_grad():  # 推理时关闭梯度计算，节省内存。
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for true_label, predicted_label in zip(labels.tolist(), predictions.tolist()):
                matrix[true_label, predicted_label] += 1

    # 对角线是真实标签与预测标签相同的正确分类，不参与错误类别对排名。
    confusion_pairs = [
        (matrix[true_label, predicted_label].item(), true_label, predicted_label)
        for true_label in range(10)
        for predicted_label in range(10)
        if true_label != predicted_label and matrix[true_label, predicted_label].item() > 0
    ]
    confusion_pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    print(f"已统计测试集全部 {len(test_dataset)} 张图片。")
    print("最容易混淆的前 10 个错误类别对（不足 10 对时显示全部）：")
    if not confusion_pairs:
        print("没有预测错误的样本。")
    for count, true_label, predicted_label in confusion_pairs[:10]:
        print(f"True {true_label} -> Pred {predicted_label}: {count} samples")

    figure, axis = plt.subplots(figsize=(10, 8))
    heatmap = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(heatmap, ax=axis, label="Number of samples")
    axis.set_title("MNIST Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))

    # 每个格子显示数量；深色背景使用白字，浅色背景使用黑字。
    threshold = matrix.max().item() / 2
    for true_label in range(10):
        for predicted_label in range(10):
            count = matrix[true_label, predicted_label].item()
            axis.text(
                predicted_label, true_label, str(count),
                ha="center", va="center",
                color="white" if count > threshold else "black",
            )
    figure.tight_layout()
    plt.show()  # 关闭图形窗口后程序结束。


if __name__ == "__main__":
    main()
