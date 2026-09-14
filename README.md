# PyTorch 入门：MNIST 图像分类

用一个简单的卷积神经网络（CNN）识别手写数字 0～9。只训练 **1 个 epoch**，适合 Mac 初学者快速体验完整的训练与测试流程。

## 1. 创建并激活虚拟环境

先在终端进入本项目目录，确保已安装 Python 3，然后执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

如果已有 `.venv`，直接执行激活命令即可。之后的命令都在已激活的环境中运行。

## 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

此步骤会安装 `torch`、`torchvision` 及它们所需的依赖。

## 3. 运行

```bash
python main.py
```

首次运行需要联网，`torchvision.datasets.MNIST` 会自动将 MNIST 数据下载到项目的 `data/` 目录；后续运行会复用已有数据。代码不会下载预训练模型或其他文件。

程序自动检查 `torch.backends.mps.is_available()`：可用时使用 `mps`，否则使用 `cpu`。运行速度取决于电脑性能。

程序先打印使用的设备，再在第一个 batch、每 100 个 batch 和最后一个 batch 打印 loss，训练结束后输出测试集 accuracy（正确预测数 / 测试图片总数）。具体数值会随设备和运行情况变化。

## 代码流程

1. 加载训练集和测试集，将 28×28 灰度图片转换并归一化为张量。
2. 使用两层卷积提取图像特征，通过池化缩小特征图，再用全连接层输出 10 个类别的分数。
3. 使用交叉熵损失和 Adam 优化器，完整训练一次训练集。
4. 关闭梯度计算，在独立测试集上统计准确率。

`main.py` 包含中文注释，可以从 `main()` 开始阅读。训练结束后不会保存模型文件。

退出虚拟环境：

```bash
deactivate
```
