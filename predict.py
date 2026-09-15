"""加载训练好的 CNN，预测一张手写数字图片。"""

import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

# main.py 的入口保护确保导入时不会运行训练或下载数据。
from main import build_model, build_transform, get_model_path


def main():
    parser = argparse.ArgumentParser(description="预测单张图片中的手写数字（0～9）")
    parser.add_argument("image_path", type=Path, help="待预测图片的路径，例如 digit.png")
    # 根据模型名称同时选择结构和参数文件。
    parser.add_argument("--model", choices=["simple", "deeper"], default="simple",
                        help="模型结构（默认：simple）")
    # 此开关只选择增强训练得到的参数；预测图片不做随机增强。
    parser.add_argument("--augment", action="store_true", help="加载数据增强训练的模型")
    # 学习率决定训练时每次参数更新的步长，也用于区分实验文件。
    parser.add_argument("--lr", type=float, default=0.001,
                        help="训练学习率（默认：0.001）")
    # batch size 表示每批图片的数量，也用于选择对应实验文件。
    parser.add_argument("--batch-size", type=int, default=128,
                        help="训练批次大小（默认：128，必须大于 0）")
    # 优化器决定如何根据梯度更新参数，也用于区分实验文件。
    parser.add_argument("--optimizer", choices=["adam", "sgd", "sgd_momentum"],
                        default="adam", help="训练优化器（默认：adam；sgd_momentum 的 momentum 为 0.9）")
    # seed 用于区分不同随机初始化的实验；加载文件时应与训练保持一致。
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认：42）")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须是大于或等于 1 的整数")
    model_path = get_model_path(args.model, args.augment, args.lr, args.batch_size, args.optimizer, args.seed)

    if not model_path.is_file():
        parser.error(
            f"找不到模型参数：{model_path}。"
            f"请先运行 python main.py --model {args.model}"
            f"{' --augment' if args.augment else ''} --lr {args.lr} --batch-size {args.batch_size} --optimizer {args.optimizer} --seed {args.seed} 训练并保存模型。"
        )
    if not args.image_path.is_file():
        parser.error(f"找不到图片：{args.image_path}")

    # 使用 CPU 加载，即使权重是在 MPS 上训练得到的，也可以预测。
    model = build_model(args.model)
    parameters = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(parameters)
    model.eval()  # 切换到预测模式。

    try:
        with Image.open(args.image_path) as image:
            # 转为单通道灰度图，再缩放到 MNIST 使用的 28×28 大小。
            image = image.convert("L").resize((28, 28))
            # 只复用张量转换和归一化，不做随机增强；增加批次维度：[1, 28, 28] → [1, 1, 28, 28]。
            images = build_transform()(image).unsqueeze(0)
    except (OSError, UnidentifiedImageError) as error:
        parser.error(f"无法读取图片：{error}")

    # 预测无需计算梯度，分数最高的类别就是预测数字。
    with torch.no_grad():
        predicted_digit = model(images).argmax(dim=1).item()
    print(f"Predicted digit: {predicted_digit}")


if __name__ == "__main__":
    main()
