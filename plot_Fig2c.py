"""
Filename: plot_Fig2c.py
Author: Naoto Yoshida
Date: 2026-03-14
Description: This script plots Figure 2c of the manuscript.
"""
import matplotlib.pyplot as plt
import json
import numpy as np

file_name_ = "default"
prefix = f"./data/"
file_name = prefix + file_name_ + "/data_all.json"

with open(file_name, "r") as outfile:
    data = json.load(outfile)

print(data.keys())

fig = plt.figure()
fig.set_size_inches(4, 6)  # Width, Height

length = 10_000
end_at = 36714 + 500
start_at = end_at - length
charge_normal = np.array(data["charge_normal"])[start_at:end_at]
temp = np.array(data["temp"]).mean(axis=1)[start_at:end_at]
joints = np.array(data["joint"])[start_at:end_at, :]
motor_actions = np.array(data["motor_action"])[start_at:end_at, :]
prob_cooling = np.array(data["prob_cooling"])[start_at:end_at]
is_food_captured = np.array(data["is_food_captured"])[start_at:end_at]
positions = np.array(data["position"])[start_at:end_at]
time_ = np.array(data["time"])[start_at:end_at]

dt = time_[1:] - time_[:-1]
robot_speed = np.linalg.norm(positions[1:] - positions[:-1], axis=1) / dt

def exponential_moving_average(arr, alpha=0.01):
    ema = np.zeros_like(arr)
    ema[0] = arr[0]

    for i in range(1, len(arr)):
        ema[i] = (1 - alpha) * ema[i-1] + alpha * arr[i]

    return ema


plt.subplot(4,1,1)
ax_left = plt.gca()
ax_right = ax_left.twinx()

# food capture
x_ = np.arange(len(time_))
x_capture = x_[is_food_captured]
y_ = 1.05 * np.array(is_food_captured)[is_food_captured]
ax_left.plot(x_capture, y_, "*k")

# ----- Energy -----
ax_left.plot(x_, charge_normal,
             alpha=0.8, color="g")
ax_left.set_ylabel("Energy", color="g")
ax_left.tick_params(axis='y', colors='g')
ax_left.set_ylim(0.5, 1.1)

ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)
ax_left.set_xlim(0, length)

# ----- Temperature -----
ax_right.plot(x_, temp,
              alpha=0.8, color="r")
ax_right.plot(x_, 40 * np.ones_like(temp), "--", alpha=0.3, color="k")

ax_right.set_ylabel("Temperature", color="r")
ax_right.tick_params(axis='y', colors='r')
ax_right.set_ylim(37.0, 43.0)

ax_right.spines['top'].set_visible(False)
ax_right.spines['left'].set_visible(False)
ax_right.set_xlim(0, length)

plt.xticks([])

plt.subplot(4,1,2)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
speed_ = exponential_moving_average(robot_speed, alpha=0.1)
ax.plot(x_[1:], speed_, alpha=1, c="k")
ax.set_xlim(0, length)
ax.set_ylabel("Torso Speed")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xticks([])

plt.subplot(4,1,3)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(x_, exponential_moving_average(prob_cooling, alpha=0.1), alpha=1, c="k")
plt.xticks([])
ax.set_ylabel("Cooling Pose\n Probability")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, length)

plt.tight_layout()

plt.subplot(4,1,4)
ax = plt.gca()
ax.plot(x_, joints[:, 0], alpha=0.6, label="hip1") # hip1
ax.plot(x_, joints[:, 1], alpha=0.6, label="ankle1") #ankle1
plt.legend(loc='upper left', ncol=2, fontsize=6, frameon=True)
ax.set_ylabel("Joint Angle")
plt.ylim(-1.1, 1.1)
ax.spines['top'].set_visible(False)
ax.set_xlim(0, length)

plt.tight_layout()
plt.show()
