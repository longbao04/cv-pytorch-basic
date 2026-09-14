"""显示 MNIST 测试集中前 16 个预测错误的样本。"""

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
    # 此开关只选择增强训练得到的参数；预测图片不做随机增强。
    parser.add_argument("--augment", action="store_true", help="加载数据增强训练的模型")
    # 学习率决定训练时每次参数更新的步长，也用于区分实验文件。
    parser.add_argument("--lr", type=float, default=0.001,
                        help="训练学习率（默认：0.001）")
    args = parser.parse_args()
    model_path = get_model_path(args.model, args.augment, args.lr)

    # Apple Silicon Mac 优先使用 MPS；不可用时自动使用 CPU。
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备：{device}")

    # main.py 有入口保护，导入它不会训练；这里只加载已保存的模型参数。
    if not model_path.is_file():
        parser.error(
            f"找不到模型参数：{model_path}。"
            f"请先运行 python main.py --model {args.model}"
            f"{' --augment' if args.augment else ''} --lr {args.lr} 训练并保存模型。"
        )
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式，使用模型进行预测。

    # 测试集只转换和归一化，不做随机增强；缺少数据时只下载 MNIST 数据集。
    data_dir = Path(__file__).resolve().parent / "data"
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=build_transform()
    )
    # 不打乱顺序，使“前 16 个错误样本”对应测试集的原始顺序。
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    error_samples = []
    error_count = 0
    # 关闭梯度计算以节省内存；完整遍历测试集，但只保存前 16 个错误样本。
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            # 找出预测标签与真实标签不同的图片在当前批次中的位置。
            wrong_indices = (predictions != labels).nonzero(as_tuple=True)[0]
            error_count += wrong_indices.numel()
            for index in wrong_indices.tolist():
                if len(error_samples) >= 16:
                    break
                error_samples.append(
                    (images[index].clone(), labels[index].item(), predictions[index].item())
                )

    print(f"测试集共 {len(test_dataset)} 张图片，预测错误 {error_count} 张。")
    if not error_samples:
        print("没有预测错误的样本，无需显示图片。")
        return

    print(f"显示前 {len(error_samples)} 个错误样本。")
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        axis.axis("off")  # 隐藏坐标轴；不足 16 个样本时，剩余位置留空。
        if index >= len(error_samples):
            continue
        image, true_label, predicted_label = error_samples[index]
        # 预处理将像素归一化到 [-1, 1]，显示时还原到 [0, 1]。
        display_image = image[0] * 0.5 + 0.5
        axis.imshow(display_image.numpy(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"True: {true_label}, Pred: {predicted_label}")

    figure.tight_layout()
    plt.show()  # 显示 4×4 网格，关闭窗口后程序结束。


if __name__ == "__main__":
    main()
