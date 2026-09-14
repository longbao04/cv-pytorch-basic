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
