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

普通训练：训练 simple 模型 3 个 epochs（默认不使用数据增强）：

```bash
python main.py --model simple --epochs 3
```

数据增强训练：

```bash
python main.py --model simple --epochs 3 --augment
```

`--augment` 默认关闭。开启后，仅训练集加入 `RandomRotation(10)`（最多旋转 10 度）和 `RandomAffine(degrees=0, translate=(0.1, 0.1))`（最多平移宽高的 10%），然后执行 `ToTensor` 和 `Normalize`。测试集和预测图片始终只执行 `ToTensor` 和 `Normalize`，方便比较同一模型有无增强的效果。`deeper` 也可以添加 `--augment`。

学习率 `--lr` 默认是 `0.001`，Adam 优化器使用此值。保持模型、增强开关和训练轮数一致，可以比较不同学习率的效果：

```bash
python main.py --model simple --epochs 3 --lr 0.0005
python main.py --model simple --epochs 3 --lr 0.005
```

`--batch-size` 默认是 `128`，表示每批处理的图片数量，必须是大于或等于 1 的整数。训练集和测试集的 DataLoader 都使用此值。保持其他配置一致，可以比较不同 batch size 的效果：

```bash
python main.py --model simple --epochs 3 --lr 0.001 --batch-size 64
python main.py --model simple --epochs 3 --lr 0.001 --batch-size 256
```

训练开始时会打印 `model`、`epochs`、`augment`、`lr` 和 `batch_size`。

训练 deeper 模型 3 个 epochs：

```bash
python main.py --model deeper --epochs 3
```

`SimpleCNN` 使用两层卷积；`DeeperCNN` 在每次池化前使用两层卷积，共四层卷积。两种模型使用相同的全连接层、预处理和优化器，默认学习率均为 `0.001`，便于比较模型结构的影响。`--model` 仅支持 `simple` / `deeper`，省略时使用 `simple`。

`--epochs` 必须是大于或等于 1 的整数。可以在 [experiment_notes.md](experiment_notes.md) 中记录训练轮数实验和模型结构对比实验的结果。

首次运行需要联网，`torchvision.datasets.MNIST` 会自动将 MNIST 数据下载到项目的 `data/` 目录；后续运行会复用已有数据。代码不会下载预训练模型或其他文件。

程序自动检查 `torch.backends.mps.is_available()`：可用时使用 `mps`，否则使用 `cpu`。运行速度取决于电脑性能。

程序打印当前训练配置和使用的设备，再在第一个 batch、每 100 个 batch 和最后一个 batch 打印 loss。每个 epoch 结束后，输出按图片数加权的平均训练损失和测试集 accuracy（正确预测数 / 测试图片总数）。具体数值会随设备和运行情况变化。

每个 epoch 的结果会立即写入对应的 CSV，字段为 `epoch`（轮数）、`avg_train_loss`（平均训练损失）、`test_accuracy`（0～1 的准确率小数，例如 0.98 表示 98%）。训练完成后保存模型参数，文件均位于脚本所在目录：

以下为默认学习率 `0.001`、batch size `128`、optimizer `adam`、seed `42` 的文件名：

| --model | --augment | 模型文件 | 训练记录 |
| --- | --- | --- | --- |
| simple | 不传 | mnist_cnn_simple_lr0p001_bs128_adam_seed42.pth | training_history_simple_lr0p001_bs128_adam_seed42.csv |
| deeper | 不传 | mnist_cnn_deeper_lr0p001_bs128_adam_seed42.pth | training_history_deeper_lr0p001_bs128_adam_seed42.csv |
| simple | 传入 | mnist_cnn_simple_aug_lr0p001_bs128_adam_seed42.pth | training_history_simple_aug_lr0p001_bs128_adam_seed42.csv |
| deeper | 传入 | mnist_cnn_deeper_aug_lr0p001_bs128_adam_seed42.pth | training_history_deeper_aug_lr0p001_bs128_adam_seed42.csv |

文件名同时包含模型、增强开关、学习率、batch size、optimizer 和 seed，小数点写成 `p`，开启增强时加入 `_aug`，关闭时省略。例如 `--lr 0.0005` 保存为 `mnist_cnn_simple_lr0p0005_bs128_adam_seed42.pth` 和 `training_history_simple_lr0p0005_bs128_adam_seed42.csv`；`--lr 0.005` 保存为 `mnist_cnn_simple_lr0p005_bs128_adam_seed42.pth` 和 `training_history_simple_lr0p005_bs128_adam_seed42.csv`。

再次训练相同模型、增强设置、学习率、batch size、optimizer 和 seed，会覆盖对应的模型和 CSV；不同设置的文件互不覆盖。模型文件和所有 `training_history*.csv` 均已加入 `.gitignore`。旧的不含学习率、batch size、optimizer 或 seed 的模型和训练记录文件不会自动加载，请根据新命名规则选择文件。每次实验后，可将最后一轮结果填入 [experiment_notes.md](experiment_notes.md)。

batch size 为 `64` 时保存 `mnist_cnn_simple_lr0p001_bs64_adam_seed42.pth` 和 `training_history_simple_lr0p001_bs64_adam_seed42.csv`；为 `256` 时保存 `mnist_cnn_simple_lr0p001_bs256_adam_seed42.pth` 和 `training_history_simple_lr0p001_bs256_adam_seed42.csv`。

## Optimizer 优化器实验

`--optimizer` 支持 `adam`、`sgd` 和 `sgd_momentum`，默认值为 `adam`。Adam 使用自适应更新步长；SGD 使用普通梯度下降；SGD + momentum 使用 `momentum=0.9` 累积更新方向。比较时保持其他参数一致，根据实际结果判断训练效果。

```bash
# Adam：
python main.py --model simple --epochs 3 --lr 0.001 --batch-size 64 --optimizer adam
# SGD：
python main.py --model simple --epochs 3 --lr 0.001 --batch-size 64 --optimizer sgd
# SGD + momentum：
python main.py --model simple --epochs 3 --lr 0.001 --batch-size 64 --optimizer sgd_momentum
# 绘制 SGD + momentum 对应的训练曲线：
python plot_training_curve.py --model simple --lr 0.001 --batch-size 64 --optimizer sgd_momentum
```

上述实验分别生成 `training_history_simple_lr0p001_bs64_adam_seed42.csv`、`training_history_simple_lr0p001_bs64_sgd_seed42.csv` 和 `training_history_simple_lr0p001_bs64_sgd_momentum_seed42.csv`；模型文件分别为 `mnist_cnn_simple_lr0p001_bs64_adam_seed42.pth`、`mnist_cnn_simple_lr0p001_bs64_sgd_seed42.pth` 和 `mnist_cnn_simple_lr0p001_bs64_sgd_momentum_seed42.pth`。开启 `--augment` 时，模型名称后加入 `_aug`。

预测、预测可视化、错误样本和混淆矩阵脚本均支持 `--optimizer`（默认 `adam`），需与训练时一致。例如：

```bash
python predict.py sample_digit.png --model simple --lr 0.001 --batch-size 64 --optimizer sgd_momentum
python visualize_predictions.py --model simple --lr 0.001 --batch-size 64 --optimizer sgd_momentum
python visualize_errors.py --model simple --lr 0.001 --batch-size 64 --optimizer sgd_momentum
python confusion_matrix.py --model simple --lr 0.001 --batch-size 64 --optimizer sgd_momentum
```

相同模型、增强开关、学习率、batch size、optimizer 和 seed 再次训练会覆盖对应文件；不同 optimizer 的文件互不覆盖。缺少文件时会提示具体路径和对应训练命令。实验结果填写到 [experiment_notes.md](experiment_notes.md) 的 Optimizer 优化器实验一节。

## 实验自动化

在已激活的虚拟环境中运行：

```bash
python run_experiments.py
```

脚本通过 Python `subprocess` 依次调用 `main.py`，运行以下四组实验，均为 `epochs=3`、`augment=False`、`batch_size=64`：

| model | optimizer | lr |
| --- | --- | --- |
| simple | adam | 0.001 |
| simple | sgd | 0.01 |
| simple | sgd_momentum | 0.01 |
| deeper | adam | 0.001 |

每组完成后，终端打印当前实验结果。`experiment_results.csv` 是多组实验的汇总表，保存在项目目录，字段为 `model, augment, epochs, lr, batch_size, optimizer, avg_train_loss, test_accuracy`。配置从 `main.py` 输出提取；指标取最后一轮的 Average train loss 和测试准确率，准确率保存为 0～1 小数（例如 `0.9855` 表示 `98.55%`），受终端输出精度限制。

某组失败时会打印失败原因并继续下一组；汇总表保留该组配置，指标留空。每完成一组就保存结果，再次运行会覆盖汇总表。`experiment_results.csv` 可以提交到 GitHub；模型参数 `mnist_cnn_*.pth` 和每轮训练记录 `training_history*.csv` 继续忽略。运行后可在 [experiment_notes.md](experiment_notes.md) 的“实验自动化”一节填写观察和结论。

## 4. 绘制训练曲线

训练生成对应的 CSV 后，在已激活的虚拟环境中运行：

```bash
python plot_training_curve.py --model simple
# 查看 batch size 64 的训练曲线：
python plot_training_curve.py --model simple --lr 0.001 --batch-size 64
# 查看学习率 0.0005 的训练曲线：
python plot_training_curve.py --model simple --lr 0.0005
# 查看增强模型的曲线，读取 training_history_simple_aug_lr0p001_bs128_adam_seed42.csv：
python plot_training_curve.py --model simple --augment
```

脚本根据 `--model`、`--augment`、`--lr`、`--batch-size`、`--optimizer` 和 `--seed` 读取对应记录，默认读取 `training_history_simple_lr0p001_bs128_adam_seed42.csv`，使用 matplotlib 在一张图的两个子图中展示平均训练损失和测试准确率随 epoch 的变化。关闭图形窗口即可结束程序。如果记录文件不存在或没有训练记录，脚本会给出清楚提示。绘图只读取 CSV，不训练模型或下载数据。

## 5. 单张图片预测

先完成训练，再将图片路径作为命令行参数传入：

```bash
python predict.py path/to/image.png
```

路径包含空格时请加引号，例如 `python predict.py "my images/digit.png"`。

默认加载项目目录中的 `mnist_cnn_simple_lr0p001_bs128_adam_seed42.pth`；使用 deeper 预测：

```bash
python predict.py sample_digit.png --model deeper
# 加载 mnist_cnn_simple_aug_lr0p001_bs128_adam_seed42.pth：
python predict.py sample_digit.png --model simple --augment
```

脚本根据 `--model` 选择 CNN 结构，根据 `--augment`、`--lr`、`--batch-size`、`--optimizer` 和 `--seed` 选择对应实验的模型文件，复用 `main.py` 中不含随机增强的 transform，使用 CPU 预测。模型文件缺失时会提示文件路径和对应的训练命令。图片会转为灰度图、缩放至 28×28，再转换为张量并按训练时的方式归一化。预测不会下载数据或运行训练。

图片应尽量使用与 MNIST 相似的黑色背景、浅色数字，且只包含一个居中的手写数字；图片风格不同可能影响准确率。

输出示例（具体数字取决于输入图片）：

```text
Predicted digit: 7
```

预测和以下三个分析脚本都支持 `--lr`（默认 `0.001`）和 `--batch-size`（默认 `128`），应与训练时一致。`--batch-size` 在这些脚本中用于选择模型文件；单张预测仍只处理一张图片，分析脚本沿用原有的推理批次。例如：

```bash
python predict.py sample_digit.png --model simple --lr 0.0005
python visualize_predictions.py --model simple --lr 0.0005
python visualize_errors.py --model simple --lr 0.0005
python confusion_matrix.py --model simple --lr 0.0005
```

例如加载 batch size 64 训练得到的模型：

```bash
python predict.py sample_digit.png --model simple --lr 0.001 --batch-size 64
python visualize_predictions.py --model simple --lr 0.001 --batch-size 64
python visualize_errors.py --model simple --lr 0.001 --batch-size 64
python confusion_matrix.py --model simple --lr 0.001 --batch-size 64
```

增强实验还需添加 `--augment`。缺少对应模型时会提示具体路径及包含学习率、batch size、optimizer 和 seed 的训练命令。

## 6. 可视化模型预测结果

确保项目目录中已有所选模型的参数文件（默认是 `mnist_cnn_simple_lr0p001_bs128_adam_seed42.pth`），然后运行：

```bash
python visualize_predictions.py
# 查看 deeper 模型的预测结果：
python visualize_predictions.py --model deeper
# 查看增强 simple 模型的预测结果：
python visualize_predictions.py --model simple --augment
```

脚本根据 `--model` 选择 `main.py` 中的模型结构，根据 `--augment`、`--lr`、`--batch-size`、`--optimizer` 和 `--seed` 选择对应实验的模型文件，复用预处理并加载对应参数文件，自动选择 `mps` 或 `cpu`，不会重新训练模型。MNIST 数据保存在项目的 `data/` 目录；缺少数据时只下载 MNIST 数据集。

程序使用 matplotlib 显示测试集前 16 张图片，排列为 4×4 网格。每张图片的标题显示 `True: 真实标签, Pred: 预测标签`；预测错误时额外标注 `WRONG`。关闭图片窗口即可结束程序。

## 7. 可视化预测错误的样本

确保项目目录中已有所选模型的参数文件（默认是 `mnist_cnn_simple_lr0p001_bs128_adam_seed42.pth`），在已激活的虚拟环境中运行：

```bash
python visualize_errors.py
# 查看 deeper 模型的错误样本：
python visualize_errors.py --model deeper
# 查看增强 simple 模型的错误样本：
python visualize_errors.py --model simple --augment
```

脚本根据 `--model`、`--augment`、`--lr`、`--batch-size`、`--optimizer` 和 `--seed` 加载 `main.py` 中的对应模型结构和参数文件，使用不含随机增强的测试集预处理，自动选择 `mps` 或 `cpu`，不会重新训练模型。缺少数据时只下载 MNIST 数据集，并保存在项目的 `data/` 目录。

程序完整遍历 MNIST 测试集，打印预测错误的总数，按测试集原始顺序显示前 16 个错误样本。matplotlib 窗口使用 4×4 网格，每张图片标题为 `True: 真实标签, Pred: 预测标签`。不足 16 个错误样本时，剩余位置留空；没有错误样本时打印提示并结束。关闭图片窗口即可结束程序。

## 8. 混淆矩阵分析

确保项目目录中已有所选模型的参数文件（默认是 `mnist_cnn_simple_lr0p001_bs128_adam_seed42.pth`），在已激活的虚拟环境中运行：

```bash
python confusion_matrix.py
# 分析 deeper 模型：
python confusion_matrix.py --model deeper
# 查看增强模型的混淆矩阵：
python confusion_matrix.py --model simple --augment
```

脚本根据 `--model`、`--augment`、`--lr`、`--batch-size`、`--optimizer` 和 `--seed` 加载对应模型结构和参数文件，自动选择 `mps` 或 `cpu`，完整遍历 MNIST 测试集，不会重新训练模型。缺少数据时只下载 MNIST 数据集到项目的 `data/` 目录；模型参数文件不存在时会给出清楚提示。

混淆矩阵的行是真实标签（True label），列是模型预测标签（Predicted label）。10×10 热力图的每个格子显示对应的样本数量：对角线表示正确分类，其他位置表示预测错误。终端按数量从高到低打印最容易混淆的前 10 个错误类别对，例如 `True 5 -> Pred 3: 12 samples`，不包含正确分类；不足 10 对时显示全部。关闭图形窗口即可结束程序。

## 代码流程

1. 加载训练集和测试集，将 28×28 灰度图片转换并归一化为张量。
2. 根据 `--model` 选择两层卷积的 SimpleCNN 或四层卷积的 DeeperCNN，通过池化缩小特征图，再用全连接层输出 10 个类别的分数。
3. 使用交叉熵损失和 `--optimizer` 选择的优化器（默认 Adam），按 `--epochs` 指定的轮数遍历训练集（默认 1 轮）。
4. 每轮训练后关闭梯度计算，在独立测试集上统计准确率，将平均训练损失和准确率写入 CSV；可通过 `plot_training_curve.py` 绘制曲线。
5. 保存模型参数，再通过 `predict.py` 加载参数预测单张图片。

`main.py`、`plot_training_curve.py`、`predict.py`、`visualize_predictions.py`、`visualize_errors.py` 和 `confusion_matrix.py` 均包含中文注释，可以从各自的 `main()` 开始阅读。

退出虚拟环境：

```bash
deactivate
```

## 重复实验与平均值/标准差

```bash
python run_repeated_experiments.py
```

脚本对 SimpleCNN + Adam（lr=0.001）、SimpleCNN + SGD（lr=0.01）、SimpleCNN + SGD + momentum（lr=0.01）、DeeperCNN + Adam（lr=0.001）分别运行 seed=0、1、2，共 12 次训练。每次 epochs=3、augment=False、batch_size=64。

`repeated_experiment_results.csv` 保存所有单次实验结果，包含配置、seed 和最后一轮的平均训练损失、测试准确率。失败实验的指标留空，脚本打印原因并继续下一次实验。`repeated_experiment_summary.csv` 按 model、augment、epochs、lr、batch_size、optimizer 汇总成功实验，保存 mean_accuracy、std_accuracy、mean_loss、std_loss。标准差使用样本标准差（分母为 n-1）；少于两次成功实验时标准差留空，没有成功实验时平均值也留空。准确率采用 0～1 小数，乘以 100 可转为百分比。再次运行会覆盖这两张结果表，它们可以提交到 GitHub。

所有训练、预测和绘图脚本均支持 `--seed`，默认值为 42。加载时各参数应与训练一致，例如：

```bash
python main.py --model deeper --epochs 3 --lr 0.001 --batch-size 64 --optimizer adam --seed 0
python plot_training_curve.py --model deeper --lr 0.001 --batch-size 64 --optimizer adam --seed 0
```

对应文件为 `mnist_cnn_deeper_lr0p001_bs64_adam_seed0.pth` 和 `training_history_deeper_lr0p001_bs64_adam_seed0.csv`。不同 seed 的文件分别保存；相同配置和 seed 再次训练会覆盖文件。随机种子用于控制随机性，跨设备运行仍可能存在差异。
