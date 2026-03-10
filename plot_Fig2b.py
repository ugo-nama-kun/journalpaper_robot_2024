"""
Filename: plot_Fig2b.py
Author: Naoto Yoshida
Date: 2026-03-14
Description: This script plots Figure 2b of the manuscript.
"""
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

file_name_ = "default"
prefix = f"./data/"
file_name = prefix + file_name_ + "/data_all.json"

with open(file_name, "r") as outfile:
    data = json.load(outfile)

print(data.keys())

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)  # Width, Height

x = np.array(data["position"])[::10, 0]
y = np.array(data["position"])[::10, 1]
time_ = np.array(data["time"])
index = np.arange(len(np.array(data["position"])[:, 0]))[::10]

cmap = plt.get_cmap('viridis')
norm = plt.Normalize(time_.min(), time_.max())

for i in tqdm(range(len(x) - 1)):
    ax.plot(x[i:i + 2], y[i:i + 2], color=cmap(norm(time_[index[i]])), alpha=0.5)

mask = np.concatenate([data["is_food_captured"][5:], np.zeros(5, dtype=np.bool_)])
x_food = np.array(data["food_position"])[mask, 0]
y_food = np.array(data["food_position"])[mask, 1]

for i in range(len(x_food)):
    ax.scatter([x_food[i]], [y_food[i]], alpha=1, s=150, edgecolor="none", label=str(i + 1))

plt.xlim([-0.45, 0.45])
plt.ylim([-0.45, 0.45])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Create an inset_axes instance with desired dimensions x, y, width, height
axins = inset_axes(ax,
                   width="5%",  # width = 5% of parent_bbox width
                   height="50%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(0.05, 0.1, 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )

# Add colorbar to inset_axes
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, cax=axins, orientation="vertical", label='Time')
ax.set_aspect('equal', 'box')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
