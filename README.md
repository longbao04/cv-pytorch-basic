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

此步骤会安装 `torch`、`torchvision`、`matplotlib` 及它们所需的依赖。

## 3. 训练并保存模型

```bash
python main.py
```

首次运行需要联网，`torchvision.datasets.MNIST` 会自动将 MNIST 数据下载到项目的 `data/` 目录；后续运行会复用已有数据。代码不会下载预训练模型或其他文件。

程序自动检查 `torch.backends.mps.is_available()`：可用时使用 `mps`，否则使用 `cpu`。运行速度取决于电脑性能。

程序先打印使用的设备，再在第一个 batch、每 100 个 batch 和最后一个 batch 打印 loss，训练结束后输出测试集 accuracy（正确预测数 / 测试图片总数）。具体数值会随设备和运行情况变化。

训练完成后，模型参数会保存到 `main.py` 所在目录的 `mnist_cnn.pth`；再次训练会覆盖该文件。该文件已加入 `.gitignore`，不会上传到 Git 仓库。

## 4. 单张图片预测

先完成训练，再将图片路径作为命令行参数传入：

```bash
python predict.py path/to/image.png
```

路径包含空格时请加引号，例如 `python predict.py "my images/digit.png"`。

脚本从项目目录加载 `mnist_cnn.pth`，复用 `main.py` 中的 CNN 结构和 transform，使用 CPU 预测。图片会转为灰度图、缩放至 28×28，再转换为张量并按训练时的方式归一化。预测不会下载数据或运行训练。

图片应尽量使用与 MNIST 相似的黑色背景、浅色数字，且只包含一个居中的手写数字；图片风格不同可能影响准确率。

输出示例（具体数字取决于输入图片）：

```text
Predicted digit: 7
```

## 5. 可视化模型预测结果

确保项目目录中已有 `mnist_cnn.pth`，然后运行：

```bash
python visualize_predictions.py
```

脚本复用 `main.py` 中的 `SimpleCNN` 和预处理，加载已有模型参数，自动选择 `mps` 或 `cpu`，不会重新训练模型。MNIST 数据保存在项目的 `data/` 目录；缺少数据时只下载 MNIST 数据集。

程序使用 matplotlib 显示测试集前 16 张图片，排列为 4×4 网格。每张图片的标题显示 `True: 真实标签, Pred: 预测标签`；预测错误时额外标注 `WRONG`。关闭图片窗口即可结束程序。

## 代码流程

1. 加载训练集和测试集，将 28×28 灰度图片转换并归一化为张量。
2. 使用两层卷积提取图像特征，通过池化缩小特征图，再用全连接层输出 10 个类别的分数。
3. 使用交叉熵损失和 Adam 优化器，完整训练一次训练集。
4. 关闭梯度计算，在独立测试集上统计准确率。
5. 保存模型参数，再通过 `predict.py` 加载参数预测单张图片。

`main.py`、`predict.py` 和 `visualize_predictions.py` 均包含中文注释，可以从各自的 `main()` 开始阅读。

退出虚拟环境：

```bash
deactivate
```
