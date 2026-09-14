# PyTorch 入门：MNIST 图像分类

用卷积神经网络（CNN）识别手写数字 0～9，支持 `simple` 和 `deeper` 两种结构，默认使用 `simple`。默认训练 **1 个 epoch**，也可以通过 `--epochs` 指定训练轮数，适合 Mac 初学者体验完整的训练与测试流程。

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

训练 1 个 epoch（默认值）：

```bash
python main.py
```

训练 simple 模型 3 个 epochs：

```bash
python main.py --model simple --epochs 3
```

训练 deeper 模型 3 个 epochs：

```bash
python main.py --model deeper --epochs 3
```

`SimpleCNN` 使用两层卷积；`DeeperCNN` 在每次池化前使用两层卷积，共四层卷积。两种模型使用相同的全连接层、预处理、优化器和学习率，便于比较模型结构的影响。`--model` 仅支持 `simple` / `deeper`，省略时使用 `simple`。

`--epochs` 必须是大于或等于 1 的整数。可以在 [experiment_notes.md](experiment_notes.md) 中记录训练轮数实验和模型结构对比实验的结果。

首次运行需要联网，`torchvision.datasets.MNIST` 会自动将 MNIST 数据下载到项目的 `data/` 目录；后续运行会复用已有数据。代码不会下载预训练模型或其他文件。

程序自动检查 `torch.backends.mps.is_available()`：可用时使用 `mps`，否则使用 `cpu`。运行速度取决于电脑性能。

程序先打印使用的设备，再在第一个 batch、每 100 个 batch 和最后一个 batch 打印 loss。每个 epoch 结束后，输出按图片数加权的平均训练损失和测试集 accuracy（正确预测数 / 测试图片总数）。具体数值会随设备和运行情况变化。

每个 epoch 的结果会立即写入项目目录的 `training_history.csv`，用于查看训练过程和绘制曲线。字段为 `epoch`（轮数）、`avg_train_loss`（平均训练损失）、`test_accuracy`（0～1 的准确率小数，例如 0.98 表示 98%）。再次训练会覆盖旧记录，该文件已加入 `.gitignore`，不会上传到 Git 仓库。

训练完成后，模型参数会保存到脚本所在目录：`simple` 保存为 `mnist_cnn_simple.pth`，`deeper` 保存为 `mnist_cnn_deeper.pth`。再次训练同一结构会覆盖对应文件，两种结构的参数文件互不覆盖。这些文件以及旧的 `mnist_cnn.pth` 均已加入 `.gitignore`。新脚本按模型名称加载新文件名，旧文件不会自动加载。

`training_history.csv` 仍只保存最近一次训练的记录。比较模型时，请在每次训练后将最后一轮的 `avg_train_loss` 和 `test_accuracy` 填入实验表，再运行下一次训练。

## 4. 绘制训练曲线

训练生成 `training_history.csv` 后，在已激活的虚拟环境中运行：

```bash
python plot_training_curve.py
```

脚本使用 matplotlib 在一张图的两个子图中展示平均训练损失和测试准确率随 epoch 的变化。关闭图形窗口即可结束程序。如果记录文件不存在或没有训练记录，脚本会给出清楚提示。绘图只读取 CSV，不训练模型或下载数据。

## 5. 单张图片预测

先完成训练，再将图片路径作为命令行参数传入：

```bash
python predict.py path/to/image.png
```

路径包含空格时请加引号，例如 `python predict.py "my images/digit.png"`。

默认加载项目目录中的 `mnist_cnn_simple.pth`；使用 deeper 预测：

```bash
python predict.py sample_digit.png --model deeper
```

脚本根据 `--model` 加载对应的 CNN 结构和参数文件，复用 `main.py` 中的 transform，使用 CPU 预测。模型文件缺失时会提示文件路径和对应的训练命令。图片会转为灰度图、缩放至 28×28，再转换为张量并按训练时的方式归一化。预测不会下载数据或运行训练。

图片应尽量使用与 MNIST 相似的黑色背景、浅色数字，且只包含一个居中的手写数字；图片风格不同可能影响准确率。

输出示例（具体数字取决于输入图片）：

```text
Predicted digit: 7
```

## 6. 可视化模型预测结果

确保项目目录中已有所选模型的参数文件（默认是 `mnist_cnn_simple.pth`），然后运行：

```bash
python visualize_predictions.py
# 查看 deeper 模型的预测结果：
python visualize_predictions.py --model deeper
```

脚本根据 `--model` 选择 `main.py` 中的模型结构，复用预处理并加载对应参数文件，自动选择 `mps` 或 `cpu`，不会重新训练模型。MNIST 数据保存在项目的 `data/` 目录；缺少数据时只下载 MNIST 数据集。

程序使用 matplotlib 显示测试集前 16 张图片，排列为 4×4 网格。每张图片的标题显示 `True: 真实标签, Pred: 预测标签`；预测错误时额外标注 `WRONG`。关闭图片窗口即可结束程序。

## 7. 可视化预测错误的样本

确保项目目录中已有所选模型的参数文件（默认是 `mnist_cnn_simple.pth`），在已激活的虚拟环境中运行：

```bash
python visualize_errors.py
# 查看 deeper 模型的错误样本：
python visualize_errors.py --model deeper
```

脚本根据 `--model` 加载 `main.py` 中的对应模型结构和参数文件，复用训练时的预处理，自动选择 `mps` 或 `cpu`，不会重新训练模型。缺少数据时只下载 MNIST 数据集，并保存在项目的 `data/` 目录。

程序完整遍历 MNIST 测试集，打印预测错误的总数，按测试集原始顺序显示前 16 个错误样本。matplotlib 窗口使用 4×4 网格，每张图片标题为 `True: 真实标签, Pred: 预测标签`。不足 16 个错误样本时，剩余位置留空；没有错误样本时打印提示并结束。关闭图片窗口即可结束程序。

## 8. 混淆矩阵分析

确保项目目录中已有所选模型的参数文件（默认是 `mnist_cnn_simple.pth`），在已激活的虚拟环境中运行：

```bash
python confusion_matrix.py
# 分析 deeper 模型：
python confusion_matrix.py --model deeper
```

脚本根据 `--model` 加载对应模型结构和参数文件，自动选择 `mps` 或 `cpu`，完整遍历 MNIST 测试集，不会重新训练模型。缺少数据时只下载 MNIST 数据集到项目的 `data/` 目录；模型参数文件不存在时会给出清楚提示。

混淆矩阵的行是真实标签（True label），列是模型预测标签（Predicted label）。10×10 热力图的每个格子显示对应的样本数量：对角线表示正确分类，其他位置表示预测错误。终端按数量从高到低打印最容易混淆的前 10 个错误类别对，例如 `True 5 -> Pred 3: 12 samples`，不包含正确分类；不足 10 对时显示全部。关闭图形窗口即可结束程序。

## 代码流程

1. 加载训练集和测试集，将 28×28 灰度图片转换并归一化为张量。
2. 根据 `--model` 选择两层卷积的 SimpleCNN 或四层卷积的 DeeperCNN，通过池化缩小特征图，再用全连接层输出 10 个类别的分数。
3. 使用交叉熵损失和 Adam 优化器，按 `--epochs` 指定的轮数遍历训练集（默认 1 轮）。
4. 每轮训练后关闭梯度计算，在独立测试集上统计准确率，将平均训练损失和准确率写入 CSV；可通过 `plot_training_curve.py` 绘制曲线。
5. 保存模型参数，再通过 `predict.py` 加载参数预测单张图片。

`main.py`、`plot_training_curve.py`、`predict.py`、`visualize_predictions.py`、`visualize_errors.py` 和 `confusion_matrix.py` 均包含中文注释，可以从各自的 `main()` 开始阅读。

退出虚拟环境：

```bash
deactivate
```
