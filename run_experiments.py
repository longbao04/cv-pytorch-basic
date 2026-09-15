"""自动运行四组 MNIST 实验，并把最后一轮结果保存到汇总 CSV。"""

import csv
from pathlib import Path
import re
import subprocess
import sys


# 使用脚本所在目录，即使从其他目录启动也能找到 main.py。
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_DIR / "experiment_results.csv"
FIELDS = [
    "model", "augment", "epochs", "lr", "batch_size", "optimizer",
    "avg_train_loss", "test_accuracy",
]

# 四组实验均训练 3 轮，batch size 为 64，不传 --augment 即关闭增强。
EXPERIMENTS = [
    {"model": "simple", "lr": 0.001, "optimizer": "adam"},
    {"model": "simple", "lr": 0.01, "optimizer": "sgd"},
    {"model": "simple", "lr": 0.01, "optimizer": "sgd_momentum"},
    {"model": "deeper", "lr": 0.001, "optimizer": "adam"},
]


def parse_result(output):
    """从 main.py 的配置行和最后一轮指标行提取结果。"""
    config_match = re.search(r"^训练配置：(.*)$", output, re.MULTILINE)
    if config_match is None:
        raise ValueError("输出中没有找到训练配置")

    # 配置行以 | 分隔，每项为 key=value；先拆成字典，再转换数值类型。
    config = dict(
        item.strip().split("=", 1)
        for item in config_match.group(1).split("|")
    )
    if config["augment"] not in ("True", "False"):
        raise ValueError("无法识别 augment 的值")
    result = {
        "model": config["model"],
        "augment": config["augment"] == "True",
        "epochs": int(config["epochs"]),
        "lr": float(config["lr"]),
        "batch_size": int(config["batch_size"]),
        "optimizer": config["optimizer"],
    }

    # 每轮都有指标，选择最后一次匹配，而不是最后一个 batch 的 Loss。
    metrics = re.findall(
        r"Epoch (\d+)/(\d+) \| Average train loss: ([\d.eE+-]+) "
        r"\| 测试集 Accuracy：([\d.]+)%",
        output,
    )
    if not metrics:
        raise ValueError("输出中没有找到平均训练损失和测试准确率")
    epoch, epochs, loss, accuracy = metrics[-1]
    if int(epoch) != result["epochs"] or int(epochs) != result["epochs"]:
        raise ValueError("输出中没有找到最后一轮的完整指标")
    result["avg_train_loss"] = float(loss)
    # 终端的 98.55% 转成 CSV 的 0.9855，与 training_history 的单位一致。
    result["test_accuracy"] = float(accuracy) / 100
    return result


def main():
    # 每次运行重新生成汇总表；每完成一组就写入并刷新，保留已完成结果。
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=FIELDS)
        writer.writeheader()
        results_file.flush()

        for index, experiment in enumerate(EXPERIMENTS, start=1):
            result = {
                **experiment, "augment": False, "epochs": 3, "batch_size": 64,
                "avg_train_loss": "", "test_accuracy": "",
            }
            # sys.executable 复用当前 Python，避免子进程使用其他虚拟环境。
            # 参数使用列表传递，无需 shell=True，也无需手动处理命令转义。
            command = [
                sys.executable, str(PROJECT_DIR / "main.py"),
                "--model", result["model"], "--epochs", str(result["epochs"]),
                "--lr", str(result["lr"]), "--batch-size", str(result["batch_size"]),
                "--optimizer", result["optimizer"],
            ]
            print(f"\n开始实验 {index}/{len(EXPERIMENTS)}：{' '.join(command)}", flush=True)
            try:
                # 合并标准输出和错误输出，失败时也能看到 main.py 的错误详情。
                completed = subprocess.run(
                    command, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"main.py 退出码为 {completed.returncode}\n{completed.stdout}"
                    )
                result = parse_result(completed.stdout)
            except (OSError, RuntimeError, ValueError, KeyError) as error:
                # 单组失败不终止循环，失败行的指标留空，不填入虚构的结果。
                print(f"实验 {index} 失败：{error}\n继续尝试下一组实验。", flush=True)

            writer.writerow(result)
            results_file.flush()
            print(f"当前实验结果：{result}", flush=True)

    print(f"\n实验汇总已保存到：{RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
