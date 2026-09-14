"""读取每轮训练记录，用两个子图展示损失和测试准确率。"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


# 与 main.py 使用同一个路径，从其他目录运行也能找到记录。
HISTORY_PATH = Path(__file__).resolve().parent / "training_history.csv"


def main():
    if not HISTORY_PATH.is_file():
        print(
            f"找不到训练记录：{HISTORY_PATH}\n"
            "请先运行 python main.py --epochs 3 生成 training_history.csv。"
        )
        return

    # CSV 中保存的是文本，绘图前需要转换成整数或浮点数。
    try:
        with HISTORY_PATH.open(newline="", encoding="utf-8") as history_file:
            rows = list(csv.DictReader(history_file))
        epochs = [int(row["epoch"]) for row in rows]
        losses = [float(row["avg_train_loss"]) for row in rows]
        accuracies = [float(row["test_accuracy"]) for row in rows]
    except (OSError, KeyError, ValueError, csv.Error) as error:
        print(f"无法读取训练记录，请检查 CSV 字段和数据：{error}")
        return

    if not rows:
        print("training_history.csv 中没有训练记录，请先完成至少一个 epoch 的训练。")
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
