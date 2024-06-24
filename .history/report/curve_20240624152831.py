import json
import matplotlib.pyplot as plt
import numpy as np
import os

small_path = "loss_log/loss_2024-06-23-22-48-30.json"
base_path = "loss_log/loss_2024-06-23-22-39-45.json"
large_path = "loss_log/loss_2024-06-23-22-21-51.json"

with open(small_path,"r") as f:
    small_loss_list = json.load(f)
with open(base_path,"r") as f:
    base_loss_list = json.load(f)
with open(large_path,"r") as f:
    large_loss_list = json.load(f)

small_loss = np.array(small_loss_list)
base_loss = np.array(base_loss_list)
large_loss = np.array(large_loss_list)

if not os.path.exists("figure"):
    os.mkdir("figure")

# 单图
plt.figure(figsize = (6,4))
plt.title("Training Loss Visualization of Vit-Small")
plt.xlabel("iter")
plt.ylabel("Training Loss")
plt.plot(small_loss, color="C0")
plt.grid(True)
plt.legend(["small"])
plt.savefig("figure/train_loss_small.pdf")

plt.figure(figsize = (6,4))
plt.title("Training Loss Visualization of Vit-Base")
plt.xlabel("iter")
plt.ylabel("Training Loss")
plt.plot(base_loss, color="C1")
plt.grid(True)
plt.legend(["base"])
plt.savefig("figure/train_loss_base.pdf")

plt.figure(figsize = (6,4))
plt.title("Training Loss Visualization of Vit-Large")
plt.xlabel("iter")
plt.ylabel("Training Loss")
plt.plot(large_loss, color="C2")
plt.grid(True)
plt.legend(["large"])
plt.savefig("figure/train_loss_large.pdf")

# 对比
plt.figure(figsize = (6,4))
plt.title("Training Loss Visualization")
plt.xlabel("iter")
plt.ylabel("Training Loss")
plt.plot(small_loss, linewidth=0.5)
plt.plot(base_loss, linewidth=0.5)
plt.plot(large_loss, linewidth=0.5)
plt.grid(True)
plt.legend(["small","base","large"])
plt.savefig("figure/train_loss.pdf")