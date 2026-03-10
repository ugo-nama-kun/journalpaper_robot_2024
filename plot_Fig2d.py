"""
Filename: plot_Fig2d.py
Author: Naoto Yoshida
Date: 2026-03-14
Description: This script plots Figure 2d of the manuscript.
"""
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm

FOOD_INDEX = 3
array_index = FOOD_INDEX - 1
file_name_ = "default"
prefix = f"./data/"
file_name = prefix + file_name_ + "/data_all.json"

with open(file_name, "r") as outfile:
    data = json.load(outfile)

print(data.keys())

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)  # Width, Height

mask = np.concatenate([data["is_food_captured"][1:], np.zeros(1, dtype=np.bool_)])

food_timesteps = np.nonzero(mask)[0]
print(food_timesteps)
target_timestep = food_timesteps[array_index]
print("target timestep: ", target_timestep)

x_food = np.array(data["food_position"])[mask, 0]
y_food = np.array(data["food_position"])[mask, 1]

for i in range(len(x_food)):
    if i == array_index:
        ax.scatter([x_food[i]], [y_food[i]], alpha=1, s=150, edgecolor="none", label=str(i + 1), color="g")

x = np.array(data["position"])[:, 0]
y = np.array(data["position"])[:, 1]
time_ = np.array(data["time"])
index = np.arange(len(np.array(data["position"])[:, 0]))

cmap = plt.get_cmap('viridis')
norm = plt.Normalize(time_.min(), time_.max())

last_i = None
for i in tqdm(range(len(x) - 1)):
    if target_timestep - 3000 < i <= target_timestep:
        ax.plot(x[i:i + 2], y[i:i + 2], color=cmap(norm(time_[index[i]])), alpha=0.5)
        last_i = i

plt.scatter([x[last_i]], [y[last_i]], c="k", s=int(0.22 * 40000 / 0.2), alpha=0.2, edgecolors="none")
plt.xlim([-0.45, 0.45])
plt.ylim([-0.45, 0.45])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal', 'box')
plt.show()
