"""显示 MNIST 测试集前 16 张图片及模型预测结果。"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from main import MODEL_PATH, SimpleCNN, build_transform


def main():
    # Apple Silicon Mac 优先使用 MPS；不可用时使用 CPU。
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 导入 main.py 不会触发训练；这里只创建模型并加载已有参数。
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到模型参数，请确认文件存在：{MODEL_PATH}")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式。

    # 复用训练时的预处理；已有 MNIST 数据时不会重复下载。
    data_dir = Path(__file__).resolve().parent / "data"
    test_dataset = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=build_transform()
    )

    # 按测试集原始顺序取前 16 张图片，堆叠成 [16, 1, 28, 28]。
    samples = [test_dataset[index] for index in range(16)]
    images = torch.stack([image for image, _ in samples])
    labels = [label for _, label in samples]

    # 预测时关闭梯度计算，每张图片取分数最高的类别作为预测标签。
    with torch.no_grad():
        predictions = model(images.to(device)).argmax(dim=1).cpu().tolist()

    # 将像素从归一化后的 [-1, 1] 还原到 [0, 1]，方便显示。
    display_images = images * 0.5 + 0.5
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        true_label = labels[index]
        predicted_label = predictions[index]
        title = f"True: {true_label}, Pred: {predicted_label}"
        if predicted_label != true_label:
            title += " WRONG"  # 预测错误时，在标题中明确标注。
        axis.imshow(display_images[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")  # 隐藏坐标轴，让图片更清晰。

    figure.tight_layout()
    plt.show()  # 显示 4×4 图片网格，关闭窗口后程序结束。


if __name__ == "__main__":
    main()
