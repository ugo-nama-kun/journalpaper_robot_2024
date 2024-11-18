import matplotlib.pyplot as plt
import json
import seaborn as sns
import numpy as np

file_name_ = "default"
prefix = f"./data/"
file_name = prefix + file_name_ + "/data_all.json"
CHARGE_ONLY = "energy" in file_name

fig = plt.figure(figsize=(8, 4), dpi=100)

with open(file_name, "r") as outfile:
    data = json.load(outfile)

plt.clf()
if CHARGE_ONLY:
    plt.subplot(3, 1, 1)
    plt.plot(data["time"], data["charge_normal"])
    plt.plot(data["time"], 0.8 * np.ones_like(data["charge_normal"]), "--k", alpha=0.5)
    plt.xlabel("Time [sec]")
    plt.ylabel("Normalized Energy")

    plt.subplot(3, 1, 2)
    plt.plot(data["time"], np.array(data["temp"]).mean(axis=1))
    plt.plot(data["time"], 40 + np.zeros_like(data["temp"])[:, 0], "--k", alpha=0.5)
    plt.xlabel("Time [sec]")
    plt.ylabel("Average Temperature")

    plt.subplot(3, 1, 3)
    plt.plot(data["time"], data["temp"])
    plt.plot(data["time"], 40 + np.zeros_like(data["temp"])[:, 0], "--k", alpha=0.5)
    plt.legend(["hip1", "ankle1", "hip2", "ankle2", "hip3", "ankle3", "hip4", "ankle4"], loc="upper right", fontsize=8)
    plt.xlabel("Time [sec]")
    plt.ylabel("Temperature [$^\circ C$]")

    plt.tight_layout()
    plt.savefig(prefix + f"{file_name_}/{file_name_}_nature.pdf")

else:
    plt.subplot(2, 1, 1)
    plt.plot(data["time"], data["charge_normal"], "r")
    plt.plot(data["time"], 0.8 * np.ones_like(data["charge_normal"]), "--k", alpha=0.5)
    plt.ylabel("Normalized Energy")

    x_ = np.array(data["time"])[data["is_food_captured"]]
    y_ = 1.05 * np.array(data["is_food_captured"])[data["is_food_captured"]]
    plt.plot(x_, y_, "*k")
    plt.ylim(0.4, 1.1)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplot(2, 1, 2)
    plt.plot(data["time"], np.array(data["temp"]).mean(axis=1), "r")
    plt.plot(data["time"], 40 + np.zeros_like(data["temp"])[:, 0], "--k", alpha=0.5)
    plt.xlabel("Time [sec]")
    plt.ylabel("Average Temperature")
    plt.ylim([34, 47])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(prefix + f"{file_name_}/{file_name_}_v_nature.pdf")

plt.show()

print(f"saved at : {file_name}")
