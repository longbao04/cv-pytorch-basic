"""自动重复运行 MNIST 实验，保存单次结果和按配置汇总的均值、标准差。"""

import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_DIR / "repeated_experiment_results.csv"
SUMMARY_PATH = PROJECT_DIR / "repeated_experiment_summary.csv"
GROUP_FIELDS = ["model", "augment", "epochs", "lr", "batch_size", "optimizer"]
RESULT_FIELDS = GROUP_FIELDS + ["seed", "avg_train_loss", "test_accuracy"]
SUMMARY_FIELDS = GROUP_FIELDS + [
    "mean_accuracy", "std_accuracy", "mean_loss", "std_loss",
]
SEEDS = [0, 1, 2]
EXPERIMENTS = [
    {"model": "simple", "lr": 0.001, "optimizer": "adam"},
    {"model": "simple", "lr": 0.01, "optimizer": "sgd"},
    {"model": "simple", "lr": 0.01, "optimizer": "sgd_momentum"},
    {"model": "deeper", "lr": 0.001, "optimizer": "adam"},
]


def parse_result(output, expected):
    """读取 main.py 最后输出的 JSON，指标保持完整精度。"""
    prefix = "EXPERIMENT_RESULT: "
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    if not lines:
        raise ValueError("输出中没有找到完整实验结果")
    result = json.loads(lines[-1][len(prefix):])
    # 核对配置，防止将其他配置的结果写到当前实验中。
    for field in GROUP_FIELDS + ["seed"]:
        if result[field] != expected[field]:
            raise ValueError(f"输出配置不一致：{field}")
    for field in ["avg_train_loss", "test_accuracy"]:
        result[field] = float(result[field])
        if not math.isfinite(result[field]):
            raise ValueError(f"指标不是有限数值：{field}")
    if result["avg_train_loss"] < 0 or not 0 <= result["test_accuracy"] <= 1:
        raise ValueError("损失或准确率超出合理范围")
    return {field: result[field] for field in RESULT_FIELDS}


def write_summary(results):
    """按相同配置分组；失败行保留在单次表中，不参与均值和标准差。"""
    groups = {}
    for result in results:
        key = tuple(result[field] for field in GROUP_FIELDS)
        groups.setdefault(key, [])
        if result["test_accuracy"] != "" and result["avg_train_loss"] != "":
            groups[key].append(result)

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for key, successful in groups.items():
            row = dict(zip(GROUP_FIELDS, key))
            for metric, mean_field, std_field in [
                ("test_accuracy", "mean_accuracy", "std_accuracy"),
                ("avg_train_loss", "mean_loss", "std_loss"),
            ]:
                values = [result[metric] for result in successful]
                row[mean_field] = statistics.mean(values) if values else ""
                # 样本标准差使用 n-1；只有一个结果无法估计波动，留空。
                row[std_field] = statistics.stdev(values) if len(values) >= 2 else ""
            writer.writerow(row)


def main():
    results = []
    total = len(EXPERIMENTS) * len(SEEDS)
    # 每次启动重新生成表格，每完成一次实验立即保存单次结果。
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        results_file.flush()
        for experiment in EXPERIMENTS:
            for seed in SEEDS:
                result = {
                    **experiment, "augment": False, "epochs": 3, "batch_size": 64,
                    "seed": seed, "avg_train_loss": "", "test_accuracy": "",
                }
                index = len(results) + 1
                # 使用当前虚拟环境的 Python；不传 --augment 即为 False。
                # 参数以列表传递，不需要 shell=True。
                command = [
                    sys.executable, "-u", str(PROJECT_DIR / "main.py"),
                    "--model", result["model"], "--epochs", str(result["epochs"]),
                    "--lr", str(result["lr"]), "--batch-size", str(result["batch_size"]),
                    "--optimizer", result["optimizer"], "--seed", str(seed),
                ]
                print(f"\n开始实验 {index}/{total}：{result}", flush=True)
                try:
                    # 实时显示训练进度，同时收集输出用于提取最后一轮结果。
                    with subprocess.Popen(
                        command, cwd=PROJECT_DIR, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                        errors="replace",
                    ) as process:
                        output_lines = []
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            output_lines.append(line)
                        returncode = process.wait()
                    if returncode != 0:
                        raise RuntimeError(f"main.py 退出码为 {returncode}，错误详情见上方输出")
                    result = parse_result("".join(output_lines), result)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                    # 单次失败不终止后续实验；指标留空，不编造结果。
                    print(f"实验 {index} 失败：{error}。继续下一次实验。", flush=True)

                results.append(result)
                writer.writerow(result)
                results_file.flush()
                write_summary(results)
                print(f"实验 {index}/{total} 结果：{result}", flush=True)

    print(f"\n单次实验结果已保存到：{RESULTS_PATH}", flush=True)
    print(f"配置均值/标准差已保存到：{SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
