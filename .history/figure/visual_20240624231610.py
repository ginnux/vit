import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("figure/base.pkl", "rb") as f:
    data = pickle.load(f)

# Plot
im_list, mask_list, result_list, label_list = data

plt.rcParams["font.size"] = 13
plt.rcParams["font.family"] = "Times New Roman"
num = len(im_list)

plt.figure(figsize=(10, 3 * num))

for i in range(num):
    im = im_list[i]
    mask = mask_list[i]
    result = result_list[i]
    label_name = label_list[i]

    plt.subplot(num, 3, 3 * i + 1)
    plt.axis("off")
    plt.title(f"NO.{i+1} Picture: {label_name}")
    plt.imshow(im)

    plt.subplot(num, 3, 3 * i + 2)
    plt.axis("off")
    plt.title(f"NO.{i+1} Attention Mask")
    plt.imshow(mask, cmap="gray")

    plt.subplot(num, 3, 3 * i + 3)
    plt.axis("off")
    plt.title(f"NO.{i+1} Attention Map")
    plt.imshow(result)

plt.tight_layout()
plt.savefig("figure/visualize.pdf")
