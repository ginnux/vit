import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("figure/visualize.pkl", "rb") as f:
    data = pickle.load(f)

# Plot
im_list, mask_list, result_list = data

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
num = len(im_list)

plt.figure(figsize=(12, 5 * num))

for i in range(num):
    im = im_list[i]
    mask = mask_list[i]
    result = result_list[i]

    plt.subplot(num, 3, 3 * i + 1)
    plt.axis("off")
    plt.title(f"NO.{i+1} Original")
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
plt.show()
