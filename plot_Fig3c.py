import matplotlib.pyplot as plt
import json
import numpy as np

file_name_ = "default"
prefix = f"./data/"
file_name = prefix + file_name_ + "/data_all.json"

with open(file_name, "r") as outfile:
    data = json.load(outfile)

print(data.keys())

fig, ax = plt.subplots()
fig.set_size_inches(4, 6)  # Width, Height

end_at = 36714
start_at = end_at - 10_000
charge_normal = np.array(data["charge_normal"])[start_at:end_at]
temp_normal = np.array(data["temp_normal"]).mean(axis=1)[start_at:end_at]
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
    ema[0] = arr[0]  # 初期値

    for i in range(1, len(arr)):
        ema[i] = (1 - alpha) * ema[i - 1] + alpha * arr[i]

    return ema


plt.subplot(4, 1, 1)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

x_ = np.arange(len(time_))
x_capture = x_[is_food_captured]
y_ = 1.05 * np.array(is_food_captured)[is_food_captured]
plt.plot(x_capture, y_, "*k")
ax.plot(x_, np.ones_like(charge_normal) * 0.8, color="g", linestyle="--", alpha=0.3)
ax.plot(x_, charge_normal, alpha=0.8, color="g", label="Energy")
ax.plot(x_, np.zeros_like(temp_normal), color="r", linestyle="--", alpha=0.3)
ax.plot(x_, temp_normal, alpha=0.8, color="r", label="Temperature")
plt.xticks([])
ax.set_ylabel("Interoception")

plt.subplot(4, 1, 2)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(time_[1:], exponential_moving_average(robot_speed, alpha=0.1), alpha=1, c="k")
ax.set_ylabel("Torso Speed")
plt.xticks([])

plt.subplot(4, 1, 3)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(x_, exponential_moving_average(prob_cooling, alpha=0.1), alpha=1, c="k")
plt.xticks([])
ax.set_ylabel("Cooling Pose\n Probability")

plt.subplot(4, 1, 4)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.plot(x_, joints[:, 0], alpha=0.6, label="hip1")  # hip1
ax.plot(x_, joints[:, 1], alpha=0.6, label="ankle1")  # ankle1
plt.legend(loc='upper left', ncol=2, fontsize=6, frameon=True)
plt.xticks([])
ax.set_ylabel("Joint Angle")
plt.ylim(-1.1, 1.1)
plt.tight_layout()
plt.show()
