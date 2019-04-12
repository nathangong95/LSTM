import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def percentage_in_range(GT, PD, r):
    rate = []
    GT = np.reshape(GT, (GT.shape[0], 7, 2))
    PD = np.reshape(PD, (PD.shape[0], 7, 2))
    for i in range(7):
        hit = 0
        hit_navie = 0
        for j in range(1, GT.shape[0]):
            distance = ((GT[j, i, 0] - PD[j, i, 0]) ** 2 + (GT[j, i, 1] - PD[j, i, 1]) ** 2) ** 0.5
            if distance <= r:
                hit += 1
        rate.append(hit / (GT.shape[0] - 1))
    return rate  # list with len 7


def plot_result(GT, data, names):
    r = np.arange(0, 30, 0.1)
    J = []
    name = ['head', 'l_hand', 'r_hand', 'l_elbow', 'r_elbow', 'l_shoulder', 'r_shoulder']
    for dat in data:
        joints = [[], [], [], [], [], [], []]
        for i in range(len(r)):
            print(i)
            rate = percentage_in_range(GT, dat, r[i])
            for j in range(7):
                joints[j].append(rate[j])
        for i in range(7):
            joints[i] = np.asarray(joints[i])
        J.append(joints)
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        for j, nam in zip(J, names):
            plt.plot(r, j[i], label = nam)
        plt.legend()
        #plt.gca().legend(('HV', 'KF', 'PF with a with 1k particles using median'))
        plt.ylabel(name[i] + ' accuracy')
        plt.xlabel('distance')
    plt.show()


GT = np.load('Data/joints_manual.npy')
#HM = np.load('Data/jointsfromheatmap.npy')
KF = np.load('Data/joints_kalman.npy')
PF0 = np.load('joints_particle0.1.npy')
PF1 = np.load('joints_particle0.01.npy')
PF2 = np.load('joints_particle0.001.npy')
PF3 = np.load('joints_particle0.0001.npy')
data = [KF, PF0, PF1, PF2, PF3]
names = ['Patient-Pose', 'PF a = 0.1', 'PF a = 0.01', 'PF a = 0.001', 'PF a = 0.0001']
plot_result(GT, data, names)
# r = np.arange(0, 30, 0.1)
# joints = [[], [], [], [], [], [], []]
# name = ['head', 'l_hand', 'r_hand', 'l_elbow', 'r_elbow', 'l_shoulder', 'r_shoulder']
# for i in range(len(r)):
#     print(i)
#     rate = percentage_in_range(GT1, PD1, r[i])
#     for j in range(7):
#         joints[j].append(rate[j])
# for i in range(7):
#     joints[i] = np.asarray(joints[i])
#
# joints_kf = [[], [], [], [], [], [], []]
# for i in range(len(r)):
#     print(i)
#     rate = percentage_in_range(GT1, KF, r[i])
#     for j in range(7):
#         joints_kf[j].append(rate[j])
# for i in range(7):
#     joints_kf[i] = np.asarray(joints_kf[i])
#
# joints_pf = [[], [], [], [], [], [], []]
# for i in range(len(r)):
#     print(i)
#     rate = percentage_in_range(GT1, PF, r[i])
#     for j in range(7):
#         joints_pf[j].append(rate[j])
# for i in range(7):
#     joints_pf[i] = np.asarray(joints_pf[i])
#
# for i in range(7):
#     plt.subplot(4, 2, i + 1)
#     plt.plot(r, joints[i], r, joints_kf[i], r, joints_pf[i])
#     plt.gca().legend(('HV', 'KF', 'PF with a with 1k particles using median'))
#     plt.ylabel(name[i] + ' accuracy')
#     plt.xlabel('distance')
# plt.show()
