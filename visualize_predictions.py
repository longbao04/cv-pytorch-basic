"""显示 MNIST 测试集前 16 张图片及模型预测结果。"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
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
    # batch size 表示每批图片的数量，也用于选择对应实验文件。
    parser.add_argument("--batch-size", type=int, default=128,
                        help="训练批次大小（默认：128，必须大于 0）")
    # 优化器决定如何根据梯度更新参数，也用于区分实验文件。
    parser.add_argument("--optimizer", choices=["adam", "sgd", "sgd_momentum"],
                        default="adam", help="训练优化器（默认：adam；sgd_momentum 的 momentum 为 0.9）")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须是大于或等于 1 的整数")
    model_path = get_model_path(args.model, args.augment, args.lr, args.batch_size, args.optimizer)

    # Apple Silicon Mac 优先使用 MPS；不可用时使用 CPU。
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 导入 main.py 不会触发训练；这里只创建模型并加载已有参数。
    if not model_path.is_file():
        parser.error(
            f"找不到模型参数：{model_path}。"
            f"请先运行 python main.py --model {args.model}"
            f"{' --augment' if args.augment else ''} --lr {args.lr} --batch-size {args.batch_size} --optimizer {args.optimizer} 训练并保存模型。"
        )
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式。

    # 测试集只转换和归一化，不做随机增强；已有 MNIST 数据时不会重复下载。
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
