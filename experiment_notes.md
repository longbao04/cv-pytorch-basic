# MNIST CNN 实验记录

## 实验目的

比较训练 1 epoch 和 3 epochs 的训练效果，观察最终损失、测试集准确率和错误样本特点的变化。

## 实验结果

| epochs | final loss | test accuracy | 观察到的错误样本特点 | 结论 |
| --- | --- | --- | --- | --- |
| 1 | | 97.36% | | |
| 3 | 0.0676 | 98.43% | | 增加训练轮数后，测试准确率提升。 |

两次实验结果均来自已有记录，未记录的信息留空。3 epochs 的 final loss 为最后一个 batch 的 loss，不是每个 epoch 的平均 loss。

## 实验结论

- 增加训练轮数后，测试准确率从 1 epoch 的 97.36% 提升到 3 epochs 的 98.43%，提高了 1.07 个百分点。
- 这一结果说明模型在训练 1 epoch 后还没有完全收敛，继续训练仍能改善测试表现。
- 最后一个 batch 的 loss（0.0676）只能作为参考，更严谨的实验应记录每个 epoch 的平均 loss，以观察整体训练损失的变化。
- 这次实验只改变 epochs，其他参数保持不变，所以可以初步认为准确率提升主要来自训练轮数增加。

## 填写说明

- 1 epoch 运行命令：`python main.py`。
- 3 epochs 运行命令：`python main.py --epochs 3`。
- final loss 填写最后一轮最后一个 batch 打印的 Loss（不是整轮平均损失）。
- test accuracy 填写训练结束后输出的测试集 Accuracy。
- 可运行 `python visualize_errors.py` 查看当前保存模型的错误样本，记录字形模糊、笔画缺失或数字形状相似等实际观察。
- 根据两次实验结果填写结论，比较增加训练轮数是否改善效果。

## 模型结构对比实验

### 实验目的

在训练轮数、数据预处理、batch size、优化器和学习率相同的条件下，比较两层卷积的 SimpleCNN 与四层卷积的 DeeperCNN，观察增加卷积层是否改善平均训练损失、测试准确率及错误样本表现。更深的模型不一定更准确，结论应根据实际运行结果填写。

### 实验结果

| model | epochs | avg final loss | test accuracy | 观察 | 结论 |
| --- | --- | --- | --- | --- | --- |
| SimpleCNN | 3 | | 98.35% | | |
| DeeperCNN | 3 | | 98.45% | | |

以上准确率来自已有实验结果，未提供的平均训练损失和错误样本观察留空。实验运行命令如下：

```bash
python main.py --model simple --epochs 3
python main.py --model deeper --epochs 3
```

各模型的记录分别保存在 `training_history_simple_lr0p001.csv` 和 `training_history_deeper_lr0p001.csv`；再次训练相同模型、增强设置和学习率会覆盖对应文件，请及时记录结果。`avg final loss` 填最后一轮的 Average train loss（CSV 中最后一行的 `avg_train_loss`），不是最后一个 batch 的 loss；`test accuracy` 填最后一轮测试准确率，CSV 中的小数可转换为百分比。

使用 `python visualize_errors.py --model simple` 和 `python visualize_errors.py --model deeper` 查看各自错误样本，也可用 `python confusion_matrix.py --model deeper` 分析 deeper 的混淆情况，再填写实际观察与结论。

### 实验结论

- 在相同训练轮数 3 epochs 下，DeeperCNN 的测试准确率略高于 SimpleCNN。
- 准确率提升幅度为 98.45% - 98.35% = 0.10 个百分点。
- 这个提升很小，不能说明深模型一定明显更好。
- MNIST 数据集较简单，SimpleCNN 已经能取得较高准确率。
- 更严谨的实验应该多次运行，记录平均 accuracy 和标准差。

## 数据增强实验

### 实验目的

比较同一模型在有无数据增强时的效果。在模型结构、训练轮数、batch size、优化器和学习率相同的条件下，仅改变训练集是否使用轻量随机旋转和平移，观察测试准确率和错误样本的变化。测试集始终不使用数据增强。

### 实验结果

| model | augment | epochs | avg final loss | test accuracy | 观察 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| SimpleCNN | False | 3 | | 98.35% | | |
| SimpleCNN | True | 3 | 0.1352 | 98.18% | | 当前增强设置没有带来提升。 |

以上结果来自已有实验记录。增强实验的 Average train loss 为 0.1352，测试准确率为 98.18%；未提供的 baseline 平均训练损失和错误样本观察留空。

```bash
# 普通训练（baseline）：
python main.py --model simple --epochs 3
# 数据增强训练：
python main.py --model simple --epochs 3 --augment
# 查看增强实验曲线和混淆矩阵：
python plot_training_curve.py --model simple --augment
python confusion_matrix.py --model simple --augment
```

普通和增强实验分别保存到 `training_history_simple_lr0p001.csv` 与 `training_history_simple_aug_lr0p001.csv`。test accuracy 填最后一轮的测试集 Accuracy；观察可结合曲线、混淆矩阵或 `python visualize_errors.py --model simple --augment` 的错误样本填写。数据增强是否改善效果，应根据实际结果得出结论。

### 实验结论

- 在 SimpleCNN 和 3 epochs 条件下，加入数据增强后，测试准确率从 98.35% 下降到 98.18%。
- 下降幅度为 98.35% - 98.18% = 0.17 个百分点，说明当前增强设置没有带来提升。
- 可能原因是 MNIST 数据集本身较简单且干净，无增强 baseline 的准确率已经较高。
- 数据增强增加了训练难度，3 epochs 可能不足以让模型充分适应增强数据。
- 不能因此简单认为数据增强无效；更严谨的实验可以增加 epochs 或调整增强强度，继续比较有无数据增强的效果。

## 学习率实验

### 实验目的

比较不同 learning rate 对训练效果的影响。保持模型为 SimpleCNN、augment=False、epochs=3，只改变学习率，观察训练损失、测试准确率和收敛情况。

### 实验结果

| model | augment | epochs | lr | avg final loss | test accuracy | 观察 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SimpleCNN | False | 3 | 0.001 | | 98.35% | | baseline |
| SimpleCNN | False | 3 | 0.0005 | 0.0700 | 98.22% | | 准确率低于 baseline。 |
| SimpleCNN | False | 3 | 0.005 | 0.0368 | 98.37% | | 准确率略高于 baseline，但差异非常小。 |

以上结果来自已有实验记录；未提供的 baseline 平均训练损失和训练曲线观察留空。`avg final loss` 填写最后一轮的 Average train loss，不是最后一个 batch 的 loss。

```bash
python main.py --model simple --epochs 3 --lr 0.0005
python main.py --model simple --epochs 3 --lr 0.005
python plot_training_curve.py --model simple --lr 0.0005
python plot_training_curve.py --model simple --lr 0.005
```

test accuracy 填写最后一轮测试集 Accuracy。结合各自训练曲线填写观察和结论；不同学习率分别保存文件，相同模型、增强设置和学习率再次训练会覆盖对应记录。

### 实验结论

- lr=0.0005 的准确率为 98.22%，低于 baseline（lr=0.001）的 98.35%。
- lr=0.005 的准确率为 98.37%，略高于 baseline 的 98.35%。
- lr=0.005 只提升了 98.37% - 98.35% = 0.02 个百分点，差异非常小，不能说明它明显更优。
- lr=0.0005 可能因为学习率较小，3 epochs 内参数更新更保守，所以效果略低。
- 学习率会影响训练速度和稳定性。
- 更严谨的实验应该多次重复运行，记录平均 accuracy 和标准差。


## Batch Size 实验

### 实验目的

比较不同 batch size 对训练效果的影响。保持模型为 SimpleCNN、augment=False、epochs=3、lr=0.001，只改变 batch size，观察测试准确率和训练曲线的变化。

### 实验结果

| model | augment | epochs | lr | batch_size | avg final loss | test accuracy | 观察 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SimpleCNN | False | 3 | 0.001 | 128 | | 98.35% | | baseline |
| SimpleCNN | False | 3 | 0.001 | 64 | 0.0424 | 98.55% | | 当前设置下准确率最高，比 baseline 提升 0.20 个百分点。 |
| SimpleCNN | False | 3 | 0.001 | 256 | 0.0664 | 98.36% | | 与 baseline 几乎相同。 |

以上结果来自已有实验记录，本次未重新训练。`avg final loss` 记录最后一轮的 Average train loss，不是最后一个 batch 的 loss；未提供的 baseline 平均训练损失和训练曲线观察留空。

```bash
python main.py --model simple --epochs 3 --lr 0.001 --batch-size 64
python main.py --model simple --epochs 3 --lr 0.001 --batch-size 256
python plot_training_curve.py --model simple --lr 0.001 --batch-size 64
python plot_training_curve.py --model simple --lr 0.001 --batch-size 256
```

test accuracy 填写最后一轮测试集 Accuracy。训练记录和模型文件均以 `_bs64`、`_bs128` 或 `_bs256` 区分；相同模型、增强开关、学习率和 batch size 再次训练会覆盖对应文件，请及时记录结果。结合实际训练曲线和测试准确率填写观察与结论。

### 实验结论

- 在当前实验设置下，batch_size=64 的测试准确率最高，为 98.55%。
- 相比 baseline batch_size=128 的 98.35%，提升了 98.55% - 98.35% = 0.20 个百分点。
- batch_size=256 的准确率为 98.36%，与 baseline 几乎相同，仅高出 0.01 个百分点。
- batch_size 较小时，每个 epoch 中参数更新次数更多，可能带来更充分的训练。
- batch_size 较大时，更新更稳定，但每个 epoch 更新次数更少。
- 单次实验结果会受到随机初始化和数据顺序影响，更严谨的实验需要多次运行，计算准确率的平均值和标准差。
