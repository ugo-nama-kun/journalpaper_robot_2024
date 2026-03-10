
"""
Filename: plot_FigS1.py
Author: Naoto Yoshida
Date: 2026-03-14
Description: This script plots Figure S1 of the manuscript.
"""

import pandas as pd
import matplotlib.pyplot as plt
import re

# File paths
files = {
    "episode_length": "./data/data_train/wandb_export_2026-02-14T00_53_10.012+09_00.csv",
    "intero_error": "./data/data_train/wandb_export_2026-02-14T00_53_27.625+09_00.csv",
    "average_reward": "./data/data_train/wandb_export_2026-02-14T00_53_43.527+09_00.csv"
}


# Clean dataframe
def clean_df(path):
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df["global_step"], errors="coerce").notnull()]
    df["global_step"] = pd.to_numeric(df["global_step"])
    return df


data = {k: clean_df(v) for k, v in files.items()}


# Extract seed id
def extract_seed(col):
    match = re.search(r"__([0-9]+)__", col)
    return match.group(1) if match else None


# Get consistent seed order from episode_length
episode_cols = [
    col for col in data["episode_length"].columns
    if "episodic_length" in col
       and "MIN" not in col
       and "MAX" not in col
       and not col.endswith("_step")
]

seed_ids = sorted({extract_seed(col) for col in episode_cols})

# Plot function (no color指定、matplotlibデフォルトを使用)
def plot_metric(metric_key, metric_name_in_column, ylabel):
    df = data[metric_key]

    metric_cols = [
        col for col in df.columns
        if metric_name_in_column in col
           and "MIN" not in col
           and "MAX" not in col
           and not col.endswith("_step")
    ]

    # seed順で並べて描画（色順を揃えるため）
    for seed in seed_ids:
        for col in metric_cols:
            if f"__{seed}__" in col:
                print(seed, isinstance(seed, str), isinstance(seed, int))
                if seed == str(9):
                    print("That it!")
                    print(f"SEED : {seed}")
                    plt.plot(
                        df["global_step"],
                        pd.to_numeric(df[col], errors="coerce"),
                        label=f"seed_{seed}",
                        color="r",
                        alpha=1.0,
                        linewidth=3,
                    )
                else:
                    plt.plot(
                        df["global_step"],
                        pd.to_numeric(df[col], errors="coerce"),
                        label=f"seed_{seed}",
                        alpha=0.3,
                    )

    plt.xlabel("Step")
    plt.ylabel(ylabel)

# Generate three plots
plt.figure(figsize=(15, 5))
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

plt.subplot(1, 3, 1)
plot_metric("episode_length", "episodic_length", "Episode Length")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.subplot(1, 3, 2)
plot_metric("intero_error", "intero_error", "Intero Error")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

plt.subplot(1, 3, 3)
plot_metric("average_reward", "average_reward", "Average Reward")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
