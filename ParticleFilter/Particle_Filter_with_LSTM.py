import sys
import os
sys.path.insert(0, os.getcwd() + '/../Modules/')
import utils
from keras.models import load_model
import numpy as np
import particle
from matplotlib import pyplot as plt


def custom_lstm(ht, ct, data, W, U, b):
    """
    :param ht: previous hidden state (h, 1)
    :param ct: Previous cell state (h, 1)
    :param data: new input data (d, 1)
    :param W: parameter for ht (h, 4*h)
    :param U: parameter for data (d, 4*h)
    :param b: bias (4*h, )
    :return: current hidden state and cell state
    """
    global ht_, ct_
    h, _ = ht.shape
    W_i = W[:, :h]
    W_f = W[:, h: h * 2]
    W_c = W[:, h * 2: h * 3]
    W_o = W[:, h * 3:]

    U_i = U[:, :h]
    U_f = U[:, h: h * 2]
    U_c = U[:, h * 2: h * 3]
    U_o = U[:, h * 3:]

    b_i = b[:h]
    b_f = b[h: h * 2]
    b_c = b[h * 2: h * 3]
    b_o = b[h * 3:]

    ft = hard_sigmoid(W_f.T.dot(ht)+U_f.T.dot(data)+b_f.reshape((h, 1)))
    it = hard_sigmoid(W_i.T.dot(ht) + U_i.T.dot(data) + b_i.reshape((h, 1)))
    ct_bar = np.tanh(W_c.T.dot(ht) + U_c.T.dot(data) + b_c.reshape((h, 1)))
    ct_ = ft*ct + it*ct_bar
    ot = hard_sigmoid(W_o.T.dot(ht) + U_o.T.dot(data) + b_o.reshape((h, 1)))
    ht_ = ot*np.tanh(ct_)
    return ht_, ct_


def hard_sigmoid(x):
    output = 0.2 * x + 0.5
    output[x < -2.5] = 0
    output[x > 2.5] = 1
    return output

def custom_dense(ht, Dw, Db):
    """
    :param ht: previous hidden state (h, 1)
    :param Dw: Dense weight (h, d)
    :param Db: Dense bias (d, )
    :return: output: (d,)
    """
    h, d = Dw.shape
    return Dw.T.dot(ht).reshape((d,)) + Db

def normalize_heatmap(heatmap):
    """
        heatmap:256*256
    """
    minval = min(heatmap.flatten())
    heatmap = heatmap - minval
    s = heatmap.sum(axis=0).sum()
    normalized_heatmap = heatmap / float(s)
    return normalized_heatmap


def sampling_from_heatmap(normalized_heatmap):
    """
        heatmap:256*256
    """
    col_prob = np.sum(normalized_heatmap, axis=0)
    thres = float(np.random.uniform(0, 1, 1))
    sum_prob = 0
    y = 0
    for i in range(normalized_heatmap.shape[0]):
        sum_prob += col_prob[i]
        if sum_prob >= thres:
            y = i
            break
    row_prob = np.sum(normalized_heatmap, axis=1)
    thres = float(np.random.uniform(0, 1, 1))
    sum_prob = 0
    x = 0
    for i in range(normalized_heatmap.shape[0]):
        sum_prob += row_prob[i]
        if sum_prob >= thres:
            x = i
            break
    return [x, y]


def importance_sampling(particles, weight):
    """
        particles: list of num_particles
        weight: list of num_particles
    """
    thres = float(np.random.uniform(0, 1, 1))
    sum_prob = 0
    for i in range(len(weight)):
        sum_prob += weight[i]
        if sum_prob >= thres:
            return particles[i]
    return particles[len(particles)-1]


def particle_filter_with_LSTM(joints, heatmap_path, model, Q, GT, num_particles=100):
    """
        joints: 3000*7*2
        GT: 3000*7*2
        Q: 2*2*7
    """
    global particles, weights
    window_size = model.layers[0].output_shape[1]
    joints_particle = np.zeros((3000, 7, 2))
    U, W, b, Dw, Db = model.get_weights()
    h, _ = W.shape
    for i in range(joints.shape[0]):
        print(i)
        heatmaps = np.load(heatmap_path + str(i + 1) + '.npy')  # 7*256*256
        for j in range(7):
            heatmaps[j, :, :] = normalize_heatmap(heatmaps[j, :, :])
        if i == 0:
            particles = []
            weights = np.zeros((num_particles,))
            for k in range(num_particles):
                position = np.zeros((14,))
                weight = np.zeros((7,))
                for j in range(7):
                    pos = sampling_from_heatmap(heatmaps[j, :, :])
                    position[2 * j] = pos[0]
                    position[2 * j + 1] = pos[1]
                    weight[j] = heatmaps[j, pos[0], pos[1]]
                particles.append(particle.Particle(parent=None, pos=position, ht=np.zeros((h, 1)), ct=np.zeros((h, 1))))
                weights[k] = sum(weight)/7.0  # might need change here
            weights = weights/sum(weights)
            joints_particle[0, :, :] = joints[0, :, :]
        else:
            new_particles = []
            for k in range(num_particles):
                # particle_k = importance_sampling(particles, weights)
                # model_input = np.zeros((1, window_size, 14))
                # model_input[0, window_size - 1, :] = particle_k.pos
                # for a in range(window_size - 1):
                #     parent = particle_k.parent
                #     if parent is None:
                #         break
                #     model_input[0, window_size - a - 2, :] = parent.pos
                # model_input, _ = utils.normalize([model_input], [model_input])
                # predict = model.predict(model_input[0])
                # predict = utils.recover_from_normalize(predict[0, window_size - 1, :])

                particle_k = importance_sampling(particles, weights)
                ht, ct = custom_lstm(particle_k.ht, particle_k.ct, (2 * particle_k.pos / 255 - 1).reshape((14, 1)), W, U, b)
                pos = (custom_dense(ht, Dw, Db).T + 1) * 255 / 2
                updated = np.random.multivariate_normal(np.reshape(pos, (14,)), Q, 1)
                updated = np.reshape(updated, (14,))
                new_particles.append(particle.Particle(parent=particle_k, pos=updated, ht=ht, ct=ct))
            particles = new_particles
            for k in range(num_particles):
                pos = particles[k].pos
                if not (np.all(pos < 250) and np.all(pos > 5)):
                    weights[k] = 0
                    continue
                weight = np.zeros((7,))
                for j in range(7):
                    weight[j] = heatmaps[j, int(pos[2 * j]), int(pos[2 * j + 1])]
                weights[k] = sum(weight)/7.0  # same will change here
            weights = np.abs(weights)/sum(np.abs(weights))
            sum_pos = np.zeros((14,))
            for part in particles:
                sum_pos += part.pos
            joints_particle[i, :, :] = np.reshape(sum_pos / float(num_particles), (7, 2))

            # uncomment to visualize each step
            # which_joint = 0
            # print('joints is')
            # print(joints_particle[i, which_joint, :])
            # print('heatmap is')
            # print(np.where(heatmaps[which_joint, :, :] == heatmaps[which_joint, :, :].max()))
            # plt.imshow(heatmaps[which_joint, :, :])
            # for k in range(num_particles):
            #     plt.scatter(particles[k].pos[2*which_joint+1], particles[k].pos[2*which_joint], marker='o', color="red")
            # plt.scatter(GT[i, which_joint, 1], GT[i, which_joint, 0], marker='o', color="green")
            # plt.title('Right Hand ' + str(i))
            # plt.pause(0.0001)
            # plt.clf()

    return joints_particle


which_model = '2019414352'
model_path = '../Result/' + which_model + '/model.h5'
model = load_model(model_path)
window_size = model.layers[0].output_shape[1]
joints = np.load('Data/QR_train_GT.npy')
joints = np.reshape(joints, (joints.shape[0], 14))
print(joints.shape)
joints, label = utils.stack_data_one_step_prediction([joints], 1, window_size, False)
joints, label = utils.normalize(joints, label)
print(joints[0].shape)
print(label[0].shape)
predict = model.predict(joints[0])
print(predict.shape)
GT = np.zeros((label[0].shape[0], 14))
PD = np.zeros((label[0].shape[0], 14))
for i in range(label[0].shape[0]):
    GT[i, :] = label[0][i, window_size - 1, :]
    PD[i, :] = predict[i, window_size - 1, :]
GT = np.reshape(GT, (GT.shape[0], 7, 2))
PD = np.reshape(PD, (PD.shape[0], 7, 2))
GT = utils.recover_from_normalize(GT)
PD = utils.recover_from_normalize(PD)
print(GT[0, :])
print(PD[0, :])
R = utils.estimateR_with_LSTM(GT, PD)
print(R)

heatmap_path = '../../Heatmaps/'
joints = np.load('Data/jointsfromheatmap.npy')
GT = np.load('Data/joints_manual.npy')

joints_particlefilter = particle_filter_with_LSTM(joints, heatmap_path, model, R, GT, num_particles=1000)
np.save('joints_particle.npy', joints_particlefilter)