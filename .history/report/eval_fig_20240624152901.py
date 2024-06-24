import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import re
import numpy as np


# 修改全局配置
plt.rcParams["font.family"] = "Times New Roman"

with open("../log.txt", "r") as f:
    lines = f.read()

accs = re.findall(r"\d\d.\d\d\%", lines)
acc_num = [float(acc.replace("%", "")) / 100 for acc in accs]
acc_array = np.array(acc_num)

acc_idx = np.arange(5)
large_acc1_idx = 2 * acc_idx
large_acc5_idx = 2 * acc_idx + 1
base_acc1_idx = 10 + 2 * acc_idx
base_acc5_idx = 10 + 2 * acc_idx + 1
small_acc1_idx = 20 + 2 * acc_idx
small_acc5_idx = 20 + 2 * acc_idx + 1


plt.figure(figsize=(6, 4))
plt.title("ACC@1 Visualization")
plt.xlabel("epoch")
plt.ylabel("ACC@1")
plt.xticks([1, 2, 3, 4, 5])
plt.plot([1, 2, 3, 4, 5], acc_array[small_acc1_idx])
plt.plot([1, 2, 3, 4, 5], acc_array[base_acc1_idx])
plt.plot([1, 2, 3, 4, 5], acc_array[large_acc1_idx])
plt.grid(True)
plt.legend(["small", "base", "large"], loc="lower right")
# 设置 Y 轴为百分比
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.tight_layout()
plt.savefig("figure/acc1.pdf")

plt.figure(figsize=(6, 4))
plt.title("ACC@5 Visualization")
plt.xlabel("epoch")
plt.ylabel("ACC@5")
plt.xticks([1, 2, 3, 4, 5])
plt.plot([1, 2, 3, 4, 5], acc_array[small_acc5_idx])
plt.plot([1, 2, 3, 4, 5], acc_array[base_acc5_idx])
plt.plot([1, 2, 3, 4, 5], acc_array[large_acc5_idx])
plt.grid(True)
plt.legend(["small", "base", "large"])
# 设置 Y 轴为百分比
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.tight_layout()
plt.savefig("figure/acc5.pdf")
