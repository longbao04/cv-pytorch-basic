"""读取每轮训练记录，用两个子图展示损失和测试准确率。"""

import argparse
import csv

import matplotlib.pyplot as plt


# 复用训练脚本的命名规则；导入 main.py 不会启动训练或下载数据。
from main import get_history_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["simple", "deeper"], default="simple",
                        help="模型结构（默认：simple）")
    parser.add_argument("--augment", action="store_true", help="读取数据增强实验的训练记录")
    # 学习率决定训练时每次参数更新的步长，也用于区分实验文件。
    parser.add_argument("--lr", type=float, default=0.001,
                        help="训练学习率（默认：0.001）")
    args = parser.parse_args()
    # 模型结构、增强开关和学习率一起决定读取哪个 CSV。
    history_path = get_history_path(args.model, args.augment, args.lr)
    if not history_path.is_file():
        print(
            f"找不到训练记录：{history_path}\n"
            f"请先运行 python main.py --model {args.model} --epochs 3"
            f"{' --augment' if args.augment else ''} --lr {args.lr} 生成 {history_path.name}。"
        )
        return

    # CSV 中保存的是文本，绘图前需要转换成整数或浮点数。
    try:
        with history_path.open(newline="", encoding="utf-8") as history_file:
            rows = list(csv.DictReader(history_file))
        epochs = [int(row["epoch"]) for row in rows]
        losses = [float(row["avg_train_loss"]) for row in rows]
        accuracies = [float(row["test_accuracy"]) for row in rows]
    except (OSError, KeyError, ValueError, csv.Error) as error:
        print(f"无法读取训练记录，请检查 CSV 字段和数据：{error}")
        return

    if not rows:
        print(f"{history_path.name} 中没有训练记录，请先完成至少一个 epoch 的训练。")
        return

    # 一张图中放两个子图，方便同时观察损失和准确率。
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_title("Average Training Loss")
    axes[0].set_ylabel("Average Train Loss")

    axes[1].plot(epochs, accuracies, marker="o")
    axes[1].set_title("Test Accuracy")
    axes[1].set_ylabel("Accuracy (0-1)")
    axes[1].set_ylim(0, 1)

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.grid(True, alpha=0.3)

    figure.tight_layout()
    plt.show()  # 关闭图形窗口后，脚本结束。


if __name__ == "__main__":
    main()
