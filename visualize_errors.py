"""显示 MNIST 测试集中前 16 个预测错误的样本。"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from main import MODEL_PATH, SimpleCNN, build_transform


def main():
    # Apple Silicon Mac 优先使用 MPS；不可用时自动使用 CPU。
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备：{device}")

    # main.py 有入口保护，导入它不会训练；这里只加载已保存的模型参数。
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到模型参数，请确认文件存在：{MODEL_PATH}")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式，使用模型进行预测。

    # 使用与训练相同的预处理，缺少数据时只下载 MNIST 数据集。
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
