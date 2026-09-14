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

每次训练后先记录结果，再运行下一次训练，因为 `training_history.csv` 会被覆盖。`avg final loss` 填最后一轮的 Average train loss（CSV 中最后一行的 `avg_train_loss`），不是最后一个 batch 的 loss；`test accuracy` 填最后一轮测试准确率，CSV 中的小数可转换为百分比。

使用 `python visualize_errors.py --model simple` 和 `python visualize_errors.py --model deeper` 查看各自错误样本，也可用 `python confusion_matrix.py --model deeper` 分析 deeper 的混淆情况，再填写实际观察与结论。

### 实验结论

- 在相同训练轮数 3 epochs 下，DeeperCNN 的测试准确率略高于 SimpleCNN。
- 准确率提升幅度为 98.45% - 98.35% = 0.10 个百分点。
- 这个提升很小，不能说明深模型一定明显更好。
- MNIST 数据集较简单，SimpleCNN 已经能取得较高准确率。
- 更严谨的实验应该多次运行，记录平均 accuracy 和标准差。
