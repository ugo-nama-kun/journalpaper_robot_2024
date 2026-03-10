
"""
Filename: plot_Fig4_S4.py
Author: Naoto Yoshida
Date: 2026-03-14
Description: This script plots Figure 4 and S4 of the manuscript.
"""

import matplotlib.pyplot as plt
import json
import numpy as np

prefix = "./data/"
file_name = prefix + "motion_analysis/data_all.json"
with open(file_name, "r") as outfile:
    data = json.load(outfile)

half_window = 1000

data_time_all = np.array(data["time"])

data_joint_all = np.array(data["joint"])
class_joints = np.load(prefix + "motion_analysis/class_joints_alldata.npy")

data_charge_normal_all = np.atleast_2d(np.array(data["charge_normal"])).transpose()
class_charge = np.load(prefix + "motion_analysis/class_charge_alldata.npy")

# Compute all classification for all data
data_temp_all = np.atleast_2d(np.array(data["temp_normal"]).mean(axis=1)).transpose()
class_temp = np.load(prefix + "motion_analysis/class_temp_alldata.npy")

tfe_charge_joint = np.load(prefix + "motion_analysis/tfe_charge_joint_alldata.npy")
tfe_temp_joint = np.load(prefix + "motion_analysis/tfe_temp_joint_alldata.npy")
tfe_joint_charge = np.load(prefix + "motion_analysis/tfe_joint_charge_alldata.npy")
tfe_joint_temp = np.load(prefix + "motion_analysis/tfe_joint_temp_alldata.npy")


t_ = data_time_all[half_window:len(data_time_all)-half_window]

plt.figure()
plt.subplot(3, 1, 1)
plt.ylabel("Transfer Entropy")
plt.xlabel("Time [sec]")
plt.plot(t_, tfe_charge_joint, "r", label="charge->joint")
plt.plot(t_, tfe_temp_joint, "b", label="temp->joint")
plt.legend(loc="upper right")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.subplot(3, 1, 3)
plt.plot(t_[::10], class_joints[half_window:len(data_time_all)-half_window][::10], ".k", alpha=0.5)
plt.ylabel("Joint CLASS")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(3, 1, 2)
plt.plot(t_, data_charge_normal_all[half_window:len(data_time_all)-half_window], "r", label="charge")
plt.plot(t_, data_temp_all[half_window:len(data_time_all)-half_window], "b", label="temp")
plt.legend(loc="upper right")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.pause(0.1)

plt.figure()
plt.subplot(2, 1, 1)
plt.ylabel("Transfer Entropy")
plt.xlabel("Time [sec]")
plt.plot(t_, tfe_charge_joint, "r", label="charge->joint")
plt.plot(t_, tfe_joint_charge, "--r", alpha=0.5, label="joint->charge")
plt.legend(loc="upper right")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(2, 1, 2)
plt.ylabel("Transfer Entropy")
plt.xlabel("Time [sec]")
plt.plot(t_, tfe_temp_joint, "b", label="temp->joint")
plt.plot(t_, tfe_joint_temp, "--b", alpha=0.5, label="joint->temp")
plt.legend(loc="upper right")
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
# print(f"saved at : {file_name}")

print("finish.")
