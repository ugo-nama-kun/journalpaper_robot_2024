
"""
Filename: plot_FigS3.py
Author: Naoto Yoshida
Date: 2026-03-14
Description: This script plots Figure S3 of the manuscript.
"""

import matplotlib.pyplot as plt
import json
import numpy as np

from const import FLAT_POSTURE_ACTION

file_name = "./data/no_battery_exchange/data_all.json"
with open(file_name, "r") as outfile:
    data = json.load(outfile)

fig = plt.figure(figsize=(8, 8), dpi=100)
plt.subplot(4, 1, 1)
plt.plot(data["time"], data["charge_normal"])
plt.plot(data["time"], 0.8 * np.ones_like(data["charge_normal"]), "--k", alpha=0.5)
plt.xlabel("Time [s]")
plt.ylabel("Normalized Energy")

x_ = np.array(data["time"])[data["is_food_captured"]]
y_ = 1.05 * np.array(data["is_food_captured"])[data["is_food_captured"]]
plt.plot(x_, y_, "*k")
plt.ylim(0.4, 1.1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(4, 1, 2)
plt.plot(data["time"], np.array(data["temp"]).mean(axis=1))
plt.plot(data["time"], 40 + np.zeros_like(data["temp"])[:, 0], "--k", alpha=0.5)
plt.xlabel("Time [s]")
plt.ylabel("Average Temperature")
plt.ylim([34, 47])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(4, 1, 3)

def exponential_moving_variance(arr, alpha=0.1):
    emv = np.zeros(len(arr))
    ema = arr[0]  # 初期値
    emv[0] = 0  # 初期値

    for i in range(1, len(arr)):
        ema = (1 - alpha) * ema + alpha * arr[i]
        emv[i] = (1 - alpha) * emv[i - 1] + alpha * (arr[i] - ema) ** 2

    return emv

flat_action = 0.5 * (FLAT_POSTURE_ACTION + 1)
activity = np.sqrt(((np.array(data["motor_action"]) - flat_action) ** 2).mean(axis=1))  # RMS
activity_emv = exponential_moving_variance(activity, alpha=0.01)

plt.plot(data["time"], activity_emv)
plt.xlabel("Time [s]")
plt.ylabel("Motor Activity")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.subplot(4, 1, 4)
plt.plot(data["time"], np.array(data["volt"]))
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

plt.show()
