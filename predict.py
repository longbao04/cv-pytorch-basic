"""加载训练好的 CNN，预测一张手写数字图片。"""

import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

# main.py 的入口保护确保导入时不会运行训练或下载数据。
from main import MODEL_PATH, SimpleCNN, build_transform


def main():
    parser = argparse.ArgumentParser(description="预测单张图片中的手写数字（0～9）")
    parser.add_argument("image_path", type=Path, help="待预测图片的路径，例如 digit.png")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        parser.error("找不到 mnist_cnn.pth，请先运行 python main.py 训练并保存模型。")
    if not args.image_path.is_file():
        parser.error(f"找不到图片：{args.image_path}")

    # 使用 CPU 加载，即使权重是在 MPS 上训练得到的，也可以预测。
    model = SimpleCNN()
    parameters = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(parameters)
    model.eval()  # 切换到预测模式。

    try:
        with Image.open(args.image_path) as image:
            # 转为单通道灰度图，再缩放到 MNIST 使用的 28×28 大小。
            image = image.convert("L").resize((28, 28))
            # 复用训练 transform；增加批次维度：[1, 28, 28] → [1, 1, 28, 28]。
            images = build_transform()(image).unsqueeze(0)
    except (OSError, UnidentifiedImageError) as error:
        parser.error(f"无法读取图片：{error}")

    # 预测无需计算梯度，分数最高的类别就是预测数字。
    with torch.no_grad():
        predicted_digit = model(images).argmax(dim=1).item()
    print(f"Predicted digit: {predicted_digit}")


if __name__ == "__main__":
    main()
