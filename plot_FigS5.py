
"""
Filename: plot_FigS5.py
Author: Naoto Yoshida
Date: 2026-03-14
Description: This script plots Figure S5 of the manuscript. The results may differ slightly from those in the manuscript due to the stochasticity of the method.
"""
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.mixture import GaussianMixture

file_name = "./data/default/data_all.json"
with open(file_name, "r") as outfile:
    data = json.load(outfile)

data_time = np.array(data["time"])[::10]
data_charge_normal = np.atleast_2d(np.array(data["charge_normal"])[::10]).transpose()
data_temp = np.atleast_2d(np.array(data["temp_normal"]).mean(axis=1)[::10]).transpose()
data_joints = np.array(data["joint"])[::10]

# joint correlation
thresh = 0.01
corr_matrix = np.load("./data/motion_analysis/corr_matrix_joint.npy")
corr_filter = corr_matrix < thresh

# resampling
data_time_resample = []
data_charge_normal_resample = []
data_temp_resample = []
data_joints_resample = []

next_t = 0
while next_t < corr_filter.shape[0]:
    data_time_resample.append(data_time[next_t])
    data_charge_normal_resample.append(data_charge_normal[next_t])
    data_temp_resample.append(data_temp[next_t])
    data_joints_resample.append(data_joints[next_t])
    for dt in range(corr_filter.shape[0]):
        if next_t + dt == corr_filter.shape[0] or corr_filter[next_t, next_t + dt] == False:
            next_t = next_t + dt
            break

data_time_resample = np.array(data_time_resample)
data_charge_normal_resample = np.array(data_charge_normal_resample)
data_temp_resample = np.array(data_temp_resample)
data_joints_resample = np.array(data_joints_resample)

print(data_time_resample.shape, data_charge_normal_resample.shape, data_temp_resample.shape, data_joints_resample.shape)

# Calculating BIC of joint data for each number of classes
n_class_list = 2 + np.arange(20)
bic_list = []
for n_class in n_class_list:
    gmm = GaussianMixture(n_components=n_class, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=1000, n_init=1,
                          init_params='kmeans')
    gmm.fit(data_joints_resample)
    bic = gmm.bic(data_joints_resample)
    bic_list.append(bic)
    # print(f"JOINT {n_class}: {bic}")

plt.subplot(131)
print(f"JOINT Best bic: {np.min(bic_list)} @ n_class={n_class_list[np.argmin(bic_list)]}")
plt.plot(n_class_list, bic_list, "r")
ax = plt.gca()
ax.set_title("JOINT BIC")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.pause(0.001)

# Calculating BIC of normalized-charge data for each number of classes
n_class_list = 2 + np.arange(20)
bic_list = []
for n_class in n_class_list:
    gmm = GaussianMixture(n_components=n_class, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=1000, n_init=1,
                          init_params='kmeans')
    gmm.fit(data_charge_normal_resample)
    bic = gmm.bic(data_charge_normal_resample)
    bic_list.append(bic)
    # print(f"CHARGE_NORMAL {n_class}: {bic}")

plt.subplot(132)
print(f"CHARGE Best bic: {np.min(bic_list)} @ n_class={n_class_list[np.argmin(bic_list)]}")
plt.plot(n_class_list, bic_list, "r")
ax = plt.gca()
ax.set_title("CHARGE NORMAL BIC")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.pause(0.001)

# Calculating BIC of temperature data for each number of classes
n_class_list = 2 + np.arange(20)
bic_list = []
for n_class in n_class_list:
    gmm = GaussianMixture(n_components=n_class, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=1000, n_init=1,
                          init_params='kmeans')
    gmm.fit(data_temp_resample)
    bic = gmm.bic(data_temp_resample)
    bic_list.append(bic)
    # print(f"TEMP NORMAL {n_class}: {bic}")

plt.subplot(133)
print(f"TEMP Best bic: {np.min(bic_list)} @ n_class={n_class_list[np.argmin(bic_list)]}")
plt.plot(n_class_list, bic_list, "r")
ax = plt.gca()
ax.set_title("TEMP NOTMAL BIC")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

print("finish.")