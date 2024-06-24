import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open("figure/visualize.pkl", "rb") as f:
    data = pickle.load(f)

# Plot
im_list, mask_list, result_list = data

plt.rcParams["font.size"] = 10
num = len(im_list)
plt.figure(figsize=(12, 4 * num))

for i in range(num):
    im = im_list[i]
    mask = mask_list[i]
    result = result_list[i]

    plt.subplot(num, 3, 3 * i + 1)
    plt.title("Original")
    plt.imshow(im)

    plt.subplot(num, 3, 3 * i + 2)
    plt.title("Attention")
    plt.imshow(mask, cmap="gray")

    plt.subplot(num, 3, 3 * i + 3)
    plt.title("Attention Map")
    plt.imshow(result)

plt.savefig(f"figure/visualize.pdf")
