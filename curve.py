import json
import matplotlib.pyplot as plt
import numpy as np

json_path = "loss_2024-06-23-16-48-17.json"

with open(json_path,"r") as f:
    loss_list = json.load(f)

loss = np.array(loss_list)

plt.figure(figsize = (6,4))
plt.title("Training Loss Visualization")
plt.xlabel("iter")
plt.ylabel("Training Loss")
plt.plot(loss)
plt.savefig(json_path.replace(".json",".png"))