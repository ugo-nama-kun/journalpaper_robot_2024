import json
import glob
from copy import deepcopy

import numpy as np

from const import FLAT_POSTURE_ACTION
import matplotlib.pyplot as plt
import seaborn as sns

path_dir = "data_2023-11-15_all"

data_dir = "data/heatmap/" + path_dir
normal_energy_arr = np.load(data_dir + "/normal_energy_arr.npy")
normal_temp_arr = np.load(data_dir + "/normal_temp_arr.npy")

N_grid = len(normal_energy_arr)

heatmap_food = np.zeros((N_grid, N_grid))
heatmap_activity = np.zeros((N_grid, N_grid))
heatmap_prob_cooling = np.zeros((N_grid, N_grid))

std_food = np.zeros((N_grid, N_grid))
std_activity = np.zeros((N_grid, N_grid))
std_prob_cooling = np.zeros((N_grid, N_grid))

flat_action = 0.5 * (FLAT_POSTURE_ACTION + 1)


def exponential_moving_variance(arr, alpha=0.01):
    emv = np.zeros(len(arr))
    ema = arr[0]
    emv[0] = 0

    for i in range(1, len(arr)):
        ema = (1 - alpha) * ema + alpha * arr[i]
        emv[i] = (1 - alpha) * emv[i - 1] + alpha * (arr[i] - ema) ** 2

    return emv


for i in range(N_grid):  # energy
    for j in range(N_grid):  # temp

        data_list = glob.glob(data_dir + f"/sample_{i}_{j}_*.json")

        tmp_food = []
        tmp_activity = []
        tmp_prob_cooling = []
        for file_name in data_list:
            with open(file_name, "r") as outfile:
                data = json.load(outfile)

            heatmap_food[i, j] += np.array(data['is_food_captured'][-1])
            tmp_food.append(np.array(data['is_food_captured'][-1]))

            mean_activity = np.sqrt(((np.array(data["motor_action"]) - flat_action) ** 2)).mean(axis=1)  # RMS
            heatmap_activity[i, j] += exponential_moving_variance(mean_activity).mean()  # exponentially moving variance
            tmp_activity.append(exponential_moving_variance(mean_activity).mean())

            heatmap_prob_cooling[i, j] += np.array(data['prob_cooling']).mean()
            tmp_prob_cooling.append(np.array(data['prob_cooling']).mean())

        heatmap_food[i, j] /= len(data_list)
        heatmap_activity[i, j] /= len(data_list)
        heatmap_prob_cooling[i, j] /= len(data_list)
        std_food[i, j] = np.std(tmp_food)
        std_activity[i, j] = np.std(tmp_activity)
        std_prob_cooling[i, j] = np.std(tmp_prob_cooling)

data_food = heatmap_food.transpose()
std_food = std_food.transpose()

data_activity = heatmap_activity.transpose()
std_activity = std_activity.transpose()

data_prob_cooling = heatmap_prob_cooling.transpose()
std_prob_cooling = std_prob_cooling.transpose()

x_axis_data = normal_energy_arr
y_axis_data = normal_temp_arr

x_axis_labels = ['{:.2f}'.format(x) for x in x_axis_data]
x_axis_labels_original = deepcopy(x_axis_labels)
y_axis_labels = ['{:.1f}'.format(40 * (y + 1) / 2 + 20) for y in y_axis_data]
y_axis_labels_original = deepcopy(y_axis_labels)

tmp = []
for i, label in enumerate(y_axis_labels):
    if i == 0 or i == len(y_axis_labels) - 1:
        tmp.append(label)
    else:
        tmp.append(None)
y_axis_labels = deepcopy(tmp)

tmp = []
for i, label in enumerate(x_axis_labels):
    if i == 0 or i == len(x_axis_labels) - 1:
        tmp.append(label)
    else:
        tmp.append(None)
x_axis_labels = deepcopy(tmp)

# get trajectory
file_name_ = "default"  # good plot
prefix = f"./data/"
file_name = prefix + file_name_ + "/data_all.json"
with open(file_name, "r") as outfile:
    data = json.load(outfile)
hist_energy = np.array(data["charge_normal"])
hist_temp = np.array(data["temp"]).mean(axis=1)
hist_temp_norm = 2 * (hist_temp - 20) / (60 - 20) - 1  # [-1, 1]

print(hist_energy.shape, hist_temp.shape)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
color_red = sns.color_palette("hls", 8)[0]
color_blue = sns.color_palette("hls", 8)[5]
n_energy_point = 8
print(f"energy list: {x_axis_labels_original}")
print(f"Energy of Left Panel: {x_axis_labels_original[n_energy_point]}")
ci = 1.96 * std_activity[:, n_energy_point] / np.sqrt(len(std_activity[:, n_energy_point]))
ax[0].plot(range(len(y_axis_data)),
           data_activity[:, n_energy_point],
           c=color_red)
ax[0].fill_between(range(len(y_axis_data)),
                   (data_activity[:, n_energy_point] - ci),
                   (data_activity[:, n_energy_point] + ci),
                   color=color_red,
                   alpha=.1)

ax0_1 = ax[0].twinx()
ci = 1.96 * std_prob_cooling[:, n_energy_point] / np.sqrt(len(std_prob_cooling[:, n_energy_point]))
ax0_1.plot(range(len(y_axis_data)),
           data_prob_cooling[:, n_energy_point],
           c=color_blue,
           )
ax0_1.fill_between(range(len(y_axis_data)),
                   (data_prob_cooling[:, n_energy_point] - ci),
                   (data_prob_cooling[:, n_energy_point] + ci),
                   color=color_blue,
                   alpha=.1)
ax0_1.plot([9 * (40 - 20) / (60 - 20), ] * 2, [0, 1.1], "k--", alpha=0.5)
ax[0].set_xticks(range(len(y_axis_labels)))
ax[0].set_xticklabels(y_axis_labels)
ax[0].set_xlabel('Average Temperature', fontsize=20)
ax0_1.set_ylabel('Cooling Pose Probability', fontsize=20)
ax[0].set_ylabel('Average Activity (EMV)', fontsize=20)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax0_1.spines['top'].set_visible(False)

n_temp_point = 4
print(f"temp list: {y_axis_labels_original}")
print(f"Temp. of Left Panel: {y_axis_labels_original[n_temp_point]}")

color_red = sns.color_palette("hls", 8)[0]
color_blue = sns.color_palette("hls", 8)[5]
ci = 1.96 * std_food[n_temp_point, :] / np.sqrt(len(std_food[n_temp_point, :]))
ax[1].plot(range(len(x_axis_data)),
           data_food[n_temp_point, :],
           c=color_red,
           )
ax[1].fill_between(range(len(x_axis_data)),
                   (data_food[n_temp_point, :] - ci),
                   (data_food[n_temp_point, :] + ci),
                   color=color_red,
                   alpha=.1)
ax[1].plot([0.8 * 9, ] * 2, [0, 1.1], "k--", alpha=0.5)
ax[1].plot([0.6 * 9, ] * 2, [0, 1.1], "r--", alpha=0.5)
ax[1].set_xticks(range(len(x_axis_labels)))
ax[1].set_xticklabels(x_axis_labels)
ax[1].set_xlabel('Normalized Energy', fontsize=20)
ax[1].set_ylabel('Food Capture', fontsize=20)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
plt.tight_layout()
plt.pause(0.001)

fig, ax = plt.subplots(1, 3, figsize=(12, 5))

# Heatmap for food data with updated titles, axis labels, x-ticks at the bottom, shorter color bar, and upside-down y-axis
cax1 = ax[0].matshow(data_food, cmap='viridis')
fig.colorbar(cax1, ax=ax[0], fraction=0.046, pad=0.04)
ax[0].set_title('Food Capture', fontsize=15)
ax[0].set_xlabel('Normalized Energy', fontsize=15)
ax[0].set_ylabel('Average Temperature', fontsize=15)
ax[0].xaxis.set_ticks_position('bottom')
ax[0].set_xticks(range(len(x_axis_data)))
ax[0].set_xticklabels(x_axis_labels)
ax[0].set_yticks(range(len(y_axis_data)))
ax[0].set_yticklabels(y_axis_labels)
ax[0].invert_yaxis()

# Heatmap for activity data with updated titles, axis labels, x-ticks at the bottom, shorter color bar, and upside-down y-axis
cax2 = ax[1].matshow(data_activity, cmap='viridis')
fig.colorbar(cax2, ax=ax[1], fraction=0.046, pad=0.04)
ax[1].set_title('Average Activity (EMV)', fontsize=15)
ax[1].xaxis.set_ticks_position('bottom')
ax[1].set_xticks(range(len(x_axis_data)))
ax[1].set_xticklabels(x_axis_labels)
ax[1].set_yticks(range(len(y_axis_data)))
ax[1].set_yticklabels([])
ax[1].invert_yaxis()

cax3 = ax[2].matshow(data_prob_cooling, cmap='viridis', vmin=0, vmax=1)
fig.colorbar(cax3, ax=ax[2], fraction=0.046, pad=0.04)
ax[2].set_title('Cooling Pose Probability', fontsize=15)
ax[2].xaxis.set_ticks_position('bottom')
ax[2].set_xticks(range(len(x_axis_data)))
ax[2].set_xticklabels(x_axis_labels)
ax[2].set_yticks(range(len(y_axis_data)))
ax[2].set_yticklabels([])
ax[2].invert_yaxis()

plt.tight_layout()
plt.show()
