
"""
Filename: plot_FigS2.py
Author: Naoto Yoshida
Date: 2026-03-14
Description: This script plots Figure S2 of the manuscript.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import pandas as pd
import seaborn as sns
import numpy as np

data_path_dict = {
    "thermal_energy": [
        "default",
        f"./data/",
    ],
    "energy_only": [
        "data_2023-10-01-17-56-08",
        f"./data/energy_only/",
    ]
}

fig = plt.figure(figsize=(16, 8), dpi=100)
gs = gridspec.GridSpec(3, 4)  # 3 rows, 4 columns

data_joints = []
first_food_capture_step = []
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])

for data_id, data_name in enumerate(["thermal_energy", "energy_only"]):
    file_name_ = data_path_dict[data_name][0]
    prefix = data_path_dict[data_name][1]
    file_name = prefix + file_name_ + "/data_all.json"

    with open(file_name, "r") as outfile:
        data = json.load(outfile)

    data_joints.append(np.array(data["joint"]))
    print(data.keys())
    diff_charge = np.array(data["charge_normal"][1:]) - np.array(data["charge_normal"][:-1])
    first_food_capture_step.append(np.argmax(diff_charge > 0.2))
    if data_id == 0:
        ax1.plot(data["time"], data["charge_normal"], "r")
        ax1.plot(data["time"], 0.8 * np.ones_like(data["charge_normal"]), "--k", alpha=0.5)
        ax2.plot(data["time"], np.array(data["temp"]).mean(axis=1), "r")
        ax2.plot(data["time"], 40 + np.zeros_like(data["temp"])[:, 0], "--k", alpha=0.5)
    else:
        ax1.plot(data["time"], data["charge_normal"])
        ax2.plot(data["time"], np.array(data["temp"]).mean(axis=1))

ax1.set_xlim(0, 2000)
ax1.set_ylim(0.4, 1.1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylabel("Normalized Energy")
ax1.set_xlabel("Time [sec]")

ax2.set_xlim(0, 2000)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel("Time [sec]")
ax2.set_ylabel("Average Temperature")

data0 = np.array(data_joints[0])[:first_food_capture_step[0]]
data1 = np.array(data_joints[1])[:first_food_capture_step[1]]

joint_labels = ['hip1', 'ankle1', 'hip2', 'ankle2', 'hip3', 'ankle3', 'hip4', 'ankle4']
type_labels = ['Full', 'Energy-only']

data_list = []
for data_id, d_ in enumerate([data0, data1]):
    joints = d_.shape[1]
    time_ticks = d_.shape[0]
    for i in range(joints):
        for j in range(time_ticks):
            if data_id == 0:
                data_list.append([d_[j, i], 'Full', f'Joint {i + 1}'])
            else:
                data_list.append([d_[j, i], 'Energy-only', f'Joint {i + 1}'])

df = pd.DataFrame(data_list, columns=['Value', 'Type', 'Joint'])

color_red = sns.color_palette("hls", 8)[0]
color_blue = sns.color_palette("hls", 8)[5]
for i in range(8):
    ax = fig.add_subplot(gs[1 + i // 4, i % 4])
    sns.kdeplot(data=df[(df['Joint'] == f'Joint {i + 1}') & (df['Type'] == 'Full')],
                x='Value', ax=ax, color=color_red, label='Full', fill=True, alpha=0.5)
    sns.kdeplot(data=df[(df['Joint'] == f'Joint {i + 1}') & (df['Type'] == 'Energy-only')],
                x='Value', ax=ax, color=color_blue, label='Energy-only', fill=True, alpha=0.5)
    ax.set_title(joint_labels[i])
    if i == 0:
        ax.set_xlabel('Joint Angle')
        ax.set_ylabel('Probability Density')
    else:
        ax.set_xlabel('')
        ax.set_ylabel('')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
