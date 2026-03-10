"""
Filename: plot_Fig3b_hot_cold.py
Author: Naoto Yoshida
Date: 2026-03-14
Description: This script plots Figure 3b of the manuscript. Switch mode for "cold" or "hot" results.
"""
import matplotlib.pyplot as plt
import json
import numpy as np

from const import FLAT_POSTURE_ACTION

flat_action = 0.5 * (FLAT_POSTURE_ACTION + 1)
mode = "cold"  # hot or cold

# (hot) single leg impulse at 60s. splay 20s
hot_file_list = [
    "data_2023-10-12-19-50-30_hot_impulse",
    "data_2023-10-12-20-03-51_hot_impulse",
    "data_2023-10-12-20-18-08_hot_impulse",
    "data_2023-10-12-20-26-40_hot_impulse",
    "data_2023-10-12-20-35-46_hot_impulse",
    "data_2023-11-02-14-02-42_hot_impulse",
    "data_2023-11-02-14-12-09_hot_impulse",
    "data_2023-11-02-14-23-46_hot_impulse",
    "data_2023-11-02-14-35-52_hot_impulse",
    "data_2023-11-02-14-49-03_hot_impulse",
]
hot_prefix = f"./data/hot_impulse/"

# (cold) single leg impulse at 60s. splay 20s
cold_file_list = [
    "data_2023-10-12-18-49-39_cold_impulse",
    "data_2023-11-01-14-14-45_cold_impulse",
    "data_2023-11-01-14-22-03_cold_impulse",
    "data_2023-11-01-14-29-13_cold_impulse",
    "data_2023-11-01-14-38-12_cold_impulse",
    "data_2023-11-01-14-46-14_cold_impulse",
    "data_2023-11-01-14-57-34_cold_impulse",
    "data_2023-11-01-15-07-19_cold_impulse",
    "data_2023-11-01-15-16-21_cold_impulse",
    "data_2023-11-01-15-39-50_cold_impulse",
]
cold_prefix = f"./data/cold_impulse/"

if mode == "hot":
    file_list = hot_file_list
    prefix = hot_prefix
elif mode == "cold":
    file_list = cold_file_list
    prefix = cold_prefix
else:
    raise ValueError


def exponential_moving_variance(arr, alpha=0.01):
    emv = np.zeros(len(arr))
    ema = arr[0]
    emv[0] = 0

    for i in range(1, len(arr)):
        ema = (1 - alpha) * ema + alpha * arr[i]
        emv[i] = (1 - alpha) * emv[i - 1] + alpha * (arr[i] - ema) ** 2

    return emv


plot_from = 80  # [sec]
plot_end = 320  # [sec]
impulse_from = 120
impulse_end = 140
data_len = plot_end - plot_from
print(data_len)

data_index = np.zeros((len(file_list), 2))
data_time = []
data_temp = []
data_activity = []

for i in range(len(file_list)):
    file_name_ = file_list[i]
    file_name = prefix + file_name_ + "/data_all.json"

    with open(file_name, "r") as outfile:
        data_ = json.load(outfile)

    # mask
    mask = np.array(data_["time"]) < plot_from
    index_start = np.argmax(data_["time"] * mask)
    mask = np.array(data_["time"]) < plot_end
    index_end = np.argmax(data_["time"] * mask)

    data_index[i] = [index_start, index_end]

    data_time.append(np.array(data_["time"])[index_start:index_end])
    data_temp.append(np.array(data_["temp"])[index_start:index_end].mean(axis=1))

    activity = np.sqrt(((np.array(data_["motor_action"]) - flat_action) ** 2).mean(axis=1))  # RMS
    activity_emv = exponential_moving_variance(activity)  # exponentially moving variance
    data_activity.append(activity_emv[index_start:index_end])

print(data_.keys())

# plotting
plt.figure(figsize=(5, 5), dpi=100)
plt.clf()

# mask
mask = data_time[0] < 120
index0 = np.argmax(data_time[0] * mask)
mask = data_time[0] < 140
index1 = np.argmax(data_time[0] * mask)

plt.subplot(211)
plt.plot(data_time[0], data_temp[0] * 0.0 + 40., "--k", alpha=0.5)  # target
for i in range(len(file_list)):
    plt.plot(data_time[i], data_temp[i], color="r", alpha=0.6)  # average temp

plt.axvspan(impulse_from, impulse_end, color='gray', alpha=0.4, edgecolor='none')
plt.xlim([plot_from, plot_end])
plt.xticks([])
plt.ylabel("Average Temperature")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(212)
for i in range(len(file_list)):
    plt.plot(data_time[i], data_activity[i], alpha=0.4, color="k")  # motor activity

plt.axvspan(impulse_from, impulse_end, color='gray', alpha=0.4, edgecolor='none')
plt.xlim([plot_from, plot_end])
plt.yticks([0.0, 0.01, 0.02, 0.03, 0.04])
plt.ylabel("Motor Activity (EMV)")
plt.xlabel("Time [sec]")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
