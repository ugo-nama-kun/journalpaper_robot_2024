import matplotlib.pyplot as plt
import json
import seaborn as sns
import numpy as np
import pandas as pd

from const import FLAT_POSTURE_ACTION

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_palette('Set1')

flat_action = 0.5 * (FLAT_POSTURE_ACTION + 1)

mode = "hot"  # hot or cold

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


def exponential_moving_variance(arr, alpha=0.01):
    emv = np.zeros(len(arr))
    ema = arr[0]
    emv[0] = 0

    for i in range(1, len(arr)):
        ema = (1 - alpha) * ema + alpha * arr[i]
        emv[i] = (1 - alpha) * emv[i-1] + alpha * (arr[i] - ema)**2

    return emv


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


plot_from = 80  # [sec]
plot_end = 320  # [sec]
impulse_from = 120
impulse_end = 140
data_len = plot_end - plot_from
print(data_len)

data_index = np.zeros((10, 2))
data_time = []
data_temp = []
data_activity = []

points_before_at = plot_from # [sec]
points_after_at = 175 # [sec]

# plotting
plt.figure(figsize=(10, 5), dpi=100)
plt.clf()

data_all = {
    "hot": {},
    "cold": {},
}

for fig_id, mode in enumerate(["cold", "hot"]):
    file_list = None
    prefix = None
    if mode == "hot":
        file_list = hot_file_list
        prefix = hot_prefix
    elif mode == "cold":
        file_list = cold_file_list
        prefix = cold_prefix

    data_before = []
    data_after = []
    for i in range(10):
        file_name_ = file_list[i]
        file_name = prefix + file_name_ + "/data_all.json"

        with open(file_name, "r") as outfile:
            data_ = json.load(outfile)

        # mask
        mask = np.array(data_["time"]) < plot_from
        index_start = np.argmax(data_["time"] * mask)
        mask = np.array(data_["time"]) < points_after_at
        index_end = np.argmax(data_["time"] * mask)

        print("TIME [sec]: ", plot_from, plot_from + points_after_at)

        data_index[i] = [index_start, index_end]

        activity = np.sqrt(((np.array(data_["motor_action"]) - flat_action) ** 2).mean(axis=1))  # RMS
        activity_emv = exponential_moving_variance(activity)  # exponentially moving variance

        data_before.append(activity_emv[index_start])
        data_after.append(activity_emv[index_end])

    print(data_before)
    ax = plt.subplot(1, 2, fig_id+1)

    data_all[mode]["before"] = np.array(data_before)
    data_all[mode]["after"] = np.array(data_after)

    data_ba = [np.array(data_before), np.array(data_after)]
    df = pd.DataFrame({
        'before': np.array(data_before),
        'after': np.array(data_after),
    })

    df_melt = pd.melt(df)

    if mode == "hot":
        palette = ['#ffffff', '#ff7f50']
    else:
        palette = ['#ffffff', '#4169e1']

    sns.boxplot(x='variable', y='value', data=df_melt, showfliers=False, width=0.4, ax=ax,
                palette=palette)
    plt.scatter(np.zeros_like(data_before), data_before, c="k", alpha=0.5, s=10)
    plt.scatter(np.ones_like(data_after), data_after, c=palette[1], alpha=0.5, s=10)
    for y_b, y_a in zip(data_before, data_after):
        plt.plot([0, 1], [y_b, y_a], c="k", alpha=0.3)

    plt.yticks([0.0, 0.01, 0.02, 0.03])
    plt.xticks(fontsize=20)
    plt.xlabel("")
    if fig_id == 0:
        plt.ylabel("Motor Activity (EMV)", fontsize=20)
    else:
        plt.ylabel("")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

print(data_all)
print("Statistical tests")

# normality test
from scipy import stats
# 1) Shapiro-Wilk test
for mode in ["cold", "hot"]:
    for timing in ["before", "after"]:
        stat, p_value = stats.shapiro(data_all[mode][timing])
        print(f"Stat ({mode}): {stat}")
        print(f"p-value ({mode}): {p_value}")
        alpha = 0.05  # 有意水準
        if p_value > alpha:
            print(f"({timing}-{mode}) follows normal dist.")
        else:
            print(f"({timing}-{mode}) doesn't follows normal dist. ")
        print("---")


# hot before vs cold before
# Test for Homoscedasticity
stat, p_value = stats.levene(data_all["hot"]["before"], data_all["cold"]["before"])

print(f"Levene-test statistics: {stat}, p値: {p_value}")

alpha = 0.05  # significance level
if p_value > alpha:
    print("There is no significant difference in the variances of the two datasets (the null hypothesis is accepted).")
else:
    print("There is a significant difference in the variances of the two datasets (the null hypothesis is rejected).")

# Performing an Independent Samples t-Test
stat, p_value = stats.ttest_ind(data_all["hot"]["before"], data_all["cold"]["before"])

print("hot before vs cold before")
print(f"statistics: {stat}")
print(f"p^value: {p_value}")
print(f'Chens d: {(np.mean(data_all["hot"]["before"]) - np.mean(data_all["cold"]["before"])) / np.sqrt((np.std(data_all["hot"]["before"])**2 + np.std(data_all["cold"]["before"])**2) / 2)}')

alpha = 0.05  # significance level
if p_value > alpha:
    print("There is no significant difference in the means of the two datasets (the null hypothesis is accepted).")
else:
    print("There is a significant difference in the means of the two datasets (the null hypothesis is rejected).")

print("---")

# cold before vs cold after
stat, p_value = stats.ttest_rel(data_all["cold"]["before"], data_all["cold"]["after"])

print("cold before vs cold after")
print(f"statistics: {stat}")
print(f"p-value: {p_value}")

alpha = 0.05  # significance level
if p_value > alpha:
    print("There is no significant difference between the two conditions (the null hypothesis is accepted).")
else:
    print("There is a significant difference between the two conditions (the null hypothesis is rejected).")

print("---")

# hot before vs hot after
stat, p_value = stats.ttest_rel(data_all["hot"]["before"], data_all["hot"]["after"])

print("hot before vs hot after")
print(f"statistics: {stat}")
print(f"p-value: {p_value}")

alpha = 0.05  # significance level
if p_value > alpha:
    print("There is no significant difference between the two conditions (the null hypothesis is accepted).")
else:
    print("There is a significant difference between the two conditions (the null hypothesis is rejected).")

plt.show()
