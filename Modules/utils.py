from scipy.stats import pearsonr
import numpy as np
import os
import sys
import keras
import datetime
from keras.utils import to_categorical
from keras import optimizers

sys.path.insert(0, os.getcwd() + '/Modules/')
import simjoints as sj
from numpy import linalg as LA
from matplotlib import pyplot as plt

# add feature part
data_path = os.getcwd() + '/Data/'


def make_22D_data(data_path):
    filelist = os.listdir(data_path)
    j = 0
    for file in filelist:
        Data = np.genfromtxt(data_path + file, delimiter=',')
        _, s = Data.shape
        Data = Data.T
        newData = np.zeros((s, 24))
        newData[:, :14] = Data[:, :14]
        for i in range(s):
            newData[i, 14] = ((Data[i, 2] - Data[i, 6]) ** 2 + (Data[i, 3] - Data[i, 7]) ** 2) ** 0.5  # Left lower
            newData[i, 15] = ((Data[i, 4] - Data[i, 8]) ** 2 + (Data[i, 5] - Data[i, 9]) ** 2) ** 0.5  # Right lower
            newData[i, 16] = ((Data[i, 6] - Data[i, 10]) ** 2 + (Data[i, 7] - Data[i, 11]) ** 2) ** 0.5  # Left Upper
            newData[i, 17] = ((Data[i, 8] - Data[i, 12]) ** 2 + (Data[i, 9] - Data[i, 13]) ** 2) ** 0.5  # Right Upper

            v1 = [Data[i, 2] - Data[i, 6], Data[i, 3] - Data[i, 7]]
            v2 = [Data[i, 10] - Data[i, 6], Data[i, 11] - Data[i, 7]]
            newData[i, 18] = sj.get_elbow_angle(v1, v2)  # left elbow
            if np.isnan(newData[i, 18]):
                newData[i, 18] = newData[i - 1, 18]

            v1 = [Data[i, 4] - Data[i, 8], Data[i, 5] - Data[i, 9]]
            v2 = [Data[i, 12] - Data[i, 8], Data[i, 13] - Data[i, 9]]
            newData[i, 19] = sj.get_elbow_angle(v1, v2)  # right elbow
            if np.isnan(newData[i, 19]):
                newData[i, 19] = newData[i - 1, 19]

            v1 = [Data[i, 12] - Data[i, 10], Data[i, 13] - Data[i, 11]]
            v2 = [Data[i, 6] - Data[i, 10], Data[i, 7] - Data[i, 11]]
            newData[i, 20] = sj.get_axillary_angle(v1, v2)  # left shoulder
            if np.isnan(newData[i, 20]):
                newData[i, 20] = newData[i - 1, 20]

            v1 = [Data[i, 10] - Data[i, 12], Data[i, 11] - Data[i, 13]]
            v2 = [Data[i, 8] - Data[i, 12], Data[i, 9] - Data[i, 13]]
            newData[i, 21] = sj.get_axillary_angle(v1, v2)  # right shoulder
            if np.isnan(newData[i, 21]):
                newData[i, 21] = newData[i - 1, 21]

            newData[i, 22] = Data[i, 14]
            newData[i, 23] = Data[i, 15]
        np.savetxt(data_path + '/part' + str(j + 1) + '(22D).csv', newData.T, delimiter=",")
        j += 1


def load_moving_data(data_path, threshold):
    train_data, _ = load_data_one_step_prediction(data_path, step_size=1, window_size=1, moving_only=False)
    train_data = np.concatenate(train_data)
    s, _, _ = train_data.shape
    train_data = np.reshape(train_data, (s, 14))
    data = []
    for i in range(7):
        data.append(train_data[:, i * 2:(i + 1) * 2])
    # print(data[0].shape)
    # print(len(data))
    speed = []
    for i in range(7):
        temp = []
        for j in range(1, data[i].shape[0]):
            temp.append(((data[i][j, 0] - data[i][j - 1, 0]) ** 2 + (data[i][j, 1] - data[i][j - 1, 1]) ** 2) ** 0.5)
        speed.append(np.asarray(temp))
    moving_list = []
    tmp = []
    for i in range(speed[0].shape[0]):
        if not any_moving(speed, i, threshold):
            if len(tmp) > 2:
                moving_list.append(np.asarray(tmp))
            tmp = []
        else:
            if len(tmp) == 0:
                tmp.append(train_data[i, :])
            tmp.append(train_data[i + 1, :])
        if i == speed[0].shape[0] - 1:
            if len(tmp) > 2:
                moving_list.append(tmp)
    return moving_list


def any_moving(speed, which_i, threshold):
    for i in range(7):
        if speed[i][which_i] > threshold:
            return True
    return False


def load_data(data_path, step_size=5, window_size=20):
    filelist = os.listdir(data_path)
    train_data = []
    train_label = []
    filelist.sort()
    for file in filelist:
        data = np.genfromtxt(data_path + file, delimiter=',')
        a, s = data.shape
        train_data.append(np.reshape(data[:a - 2, :].T, (s, a - 2)))
        train_label.append(to_categorical(np.reshape(data[a - 2, :], (s,))))
    # deal with the step size here
    print(train_data[0].shape)
    return stack_data(train_data, train_label, step_size, window_size)


def load_data_one_step_prediction(data_path, step_size=5, window_size=20, moving_only=False):
    filelist = os.listdir(data_path)
    train_data = []
    filelist.sort()
    for file in filelist:
        data = np.genfromtxt(data_path + file, delimiter=',')
        a, s = data.shape
        train_data.append(np.reshape(data[:14, :].T, (s, 14)))
    # deal with the step size here
    # print(train_data[0].shape)
    return stack_data_one_step_prediction(train_data, step_size, window_size, moving_only)


def stack_data_one_step_prediction(train_data, step_size, window_size, moving_only):
    """ returns (a list of n*window_size*14, a list of n*14)
    """
    new_train_data = []
    new_train_label = []
    for train_dat in train_data:
        s, d = train_dat.shape
        new_train_dat = []
        new_train_lab = []
        for i in range(s - 1):
            if i % step_size == 0:
                if i >= (window_size - 1):
                    window = []
                    for j in range(window_size):
                        window.append(train_dat[i - window_size + j + 1, :])
                    window = np.asarray(window)
                    if moving_only:
                        if not (window[window.shape[0] - 1, :] == train_dat[i + 1]).all():
                            new_train_dat.append(window)
                            new_train_lab.append(train_dat[i + 1])
                    else:
                        new_train_dat.append(window)
                        tmp = np.zeros(window.shape)
                        tmp[:window.shape[0] - 1, :] = window[1:, :]
                        tmp[window.shape[0] - 1, :] = train_dat[i + 1]
                        new_train_lab.append(tmp)
        new_train_data.append(np.asarray(new_train_dat))
        new_train_label.append(np.asarray(new_train_lab))
    return new_train_data, new_train_label


def stack_data(train_data, train_label, step_size, window_size):
    """ returns (a list of n*window_size*14, a list of n*4)
    """
    new_train_data = []
    new_train_label = []
    for train_dat, train_lab in zip(train_data, train_label):
        s, d = train_dat.shape
        new_train_dat = []
        new_train_lab = []
        for i in range(s):
            if i % step_size == 0:
                if i >= (window_size - 1):
                    window = []
                    for j in range(window_size):
                        window.append(train_dat[i - window_size + j + 1, :])
                    window = np.asarray(window)
                    new_train_dat.append(window)
                    new_train_lab.append(train_lab[i])
        new_train_data.append(np.asarray(new_train_dat))
        new_train_label.append(np.asarray(new_train_lab))
    return new_train_data, new_train_label


def normalize(data, label):
    """ for one step prediction
    data: list of n*ws*14
    label: list of n*14
    """
    output_data = []
    output_label = []
    for dat, lab in zip(data, label):
        output_data.append(2 * dat / 255 - 1)
        output_label.append(2 * lab / 255 - 1)
    return output_data, output_label


def recover_from_normalize(data):
    # data:14*n
    return (data + 1) * 255 / 2


def recover_from_stack(stacked_data):
    a, _, b = stacked_data.shape
    return np.reshape(stacked_data[:, 0, :], (a, b))


def plot_speed_hist(data_path):
    """ takes in n*1*14
    """
    stacked_data = load_data(data_path, 1, 2)
    speeds = []
    for data in stacked_data[0]:
        speed = []
        s, _, _ = data.shape
        for i in range(s):
            speed.append(LA.norm(data[i, 1, :] - data[i, 0, :]))
        speeds.append(np.asarray(speed))
    bins = np.arange(0, 40, 2)
    for i in range(len(speeds)):
        if i == 2:
            plt.title('Histogram of the speeds of each folder')
        plt.subplot(2, 3, i + 1)
        plt.xlim([0, 40])
        plt.hist(speeds[i], bins=bins, alpha=0.5)

    plt.show()


def train_model(model, train_data, train_label, batch_s, epo, cross_val=False):
    call_back = []
    if cross_val:
        model_copy = keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        model_copy.compile(loss='mean_squared_error',
                           optimizer=sgd,
                           metrics=['mse'])
        for i in range(len(train_data)):
            val_X = train_data[i]
            val_Y = train_label[i]
            train_X = [train_data[j] for j in range(len(train_data)) if j != i]
            train_Y = [train_label[j] for j in range(len(train_data)) if j != i]
            train_x = np.vstack(train_X)
            train_y = np.vstack(train_Y)
            call_back.append(
                model_copy.fit(train_x, train_y, batch_size=batch_s, epochs=epo, validation_data=(val_X, val_Y)))
        del model_copy
    train_x = np.vstack(train_data)
    train_y = np.vstack(train_label)
    call_back.append(model.fit(train_x, train_y, batch_size=batch_s, epochs=epo))
    return model, call_back


# def predict(model, test_data):
#    return model.predict(test_data)
# def save_model(model, path):
#    model.save(path)
def toIntegerLabel(l):
    '''
    low level helper function to transfer one hot label to integer label
    input: one hot label
    output: integer label
    '''
    # for ele in l:
    #    if ele!=[0,0,0,1] and ele!=[0,0,1,0] and ele!=[0,1,0,0] and ele!=[1,0,0,0]:
    #        print(ele)
    label = []
    s, _ = l.shape
    for i in range(s):
        label.append(l[i].tolist().index(max(l[i].tolist())))
    return label


def correlation(label, predict):
    label = toIntegerLabel(label)
    predict = toIntegerLabel(predict)
    pearson = pearsonr(label, predict)
    return pearson


def mse(test_labe, predic):
    """ both input: n*14
        output:1*1 const
    """
    error = 0
    for i in range(test_labe.shape[0]):
        e = test_labe[i, :] - predic[i, :]
        e = e ** 2
        error += e.sum() ** (1 / 2)
    return error


def recover_from_stack(label):
    # label: N*window_size*14
    window_size = label.shape[1]
    recoverd_label = np.zeros((label.shape[0], label.shape[2]))
    for i in range(label.shape[0]):
        recoverd_label[i, :] = label[i, window_size - 1, :]
    return recoverd_label


def save_result_one_step_prediction(model, test_data, test_label, callback, cross_val=False):
    predict = []
    for test_dat in test_data:
        predict.append(model.predict(test_dat))
    score = []
    for test_labe, predic in zip(test_label, predict):
        score.append(mse(recover_from_stack(test_labe), recover_from_stack(predic)))
    now = datetime.datetime.now()

    path = os.getcwd() + '/Result/' + str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    os.makedirs(path)
    for i in range(len(test_data)):
        label = test_label[i]
        pre = predict[i]
        np.savetxt(path + '/GroundTruth' + str(i + 1) + '.csv',
                   recover_from_normalize(recover_from_stack(label)), delimiter=",")
        np.savetxt(path + '/Prediction' + str(i + 1) + '.csv',
                   recover_from_normalize(recover_from_stack(pre)), delimiter=",")
    model.save(path + '/model.h5')
    i = 0
    for call in callback:
        loss = call.history["loss"]
        mse_ = call.history["mean_squared_error"]
        with open(path + "/loss_part" + str(i + 1) + ".txt", 'w') as f:
            for s in loss:
                f.write(str(s) + '\n')
        with open(path + "/mse_part" + str(i + 1) + ".txt", 'w') as f:
            for s in mse_:
                f.write(str(s) + '\n')

        if cross_val:
            val_loss = call.history["val_loss"]
            val_mse = call.history["val_mean_squared_error"]
            with open(path + "/val_loss_part" + str(i + 1) + ".txt", 'w') as f:
                for s in val_loss:
                    f.write(str(s) + '\n')
            with open(path + "/val_mse_part" + str(i + 1) + ".txt", 'w') as f:
                for s in val_mse:
                    f.write(str(s) + '\n')
        i += 1
    with open(path + "/score_mse.txt", 'w') as f:
        f.write('\n'.join('%f' % x for x in score))
    return None


def save_result(model, test_data, test_label, callback):
    predict = []
    for test_dat in test_data:
        predict.append(model.predict(test_dat))
    score = []
    for test_labe, predic in zip(test_label, predict):
        score.append(correlation(test_labe, predic))
    now = datetime.datetime.now()

    path = os.getcwd() + '/Result/' + str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute)
    os.makedirs(path)
    for i in range(len(test_data)):
        data = recover_from_stack(test_data[i])
        label = np.asarray(toIntegerLabel(test_label[i]))
        pre = np.asarray(toIntegerLabel(predict[i]))
        label = np.reshape(label, (label.shape[0], 1))
        pre = np.reshape(pre, (pre.shape[0], 1))
        save = np.concatenate((data, label, pre), axis=1)
        np.savetxt(path + '/result_part' + str(i + 1) + '.csv', save, delimiter=",")
    model.save(path + '/model.h5')
    i = 0
    for call in callback:
        loss = call.history["loss"]
        acc = call.history["acc"]
        val_loss = call.history["val_loss"]
        val_acc = call.history["val_acc"]
        with open(path + "/loss_part" + str(i + 1) + ".txt", 'w') as f:
            for s in loss:
                f.write(str(s) + '\n')
        with open(path + "/acc_part" + str(i + 1) + ".txt", 'w') as f:
            for s in acc:
                f.write(str(s) + '\n')
        with open(path + "/val_loss_part" + str(i + 1) + ".txt", 'w') as f:
            for s in val_loss:
                f.write(str(s) + '\n')
        with open(path + "/val_acc_part" + str(i + 1) + ".txt", 'w') as f:
            for s in val_acc:
                f.write(str(s) + '\n')
        i += 1
    with open(path + "/score.txt", 'w') as f:
        f.write('\n'.join('%f %f' % x for x in score))
    return None


def estimateQR_with_a(joints, est):
    # 13494,7,2
    # print(joints[0,:,:])
    # print(est[0,:,:])
    s = joints.shape[0]
    F = np.array([[1.0, 0.0, 1.0, 0.0, 0.5, 0.0],
                  [0.0, 1.0, 0.0, 1.0, 0.0, 0.5],
                  [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

    X = np.zeros((6, 7, s))
    for i in range(2, s):
        for j in range(7):
            X[0, j, i] = joints[i, j, 0]
            X[1, j, i] = joints[i, j, 1]
            X[2, j, i] = joints[i, j, 0] - joints[i - 1, j, 0]
            X[3, j, i] = joints[i, j, 1] - joints[i - 1, j, 1]
            X[4, j, i] = joints[i, j, 0] - joints[i - 1, j, 0] - joints[i - 1, j, 0] + joints[i - 2, j, 0]
            X[5, j, i] = joints[i, j, 1] - joints[i - 1, j, 1] - joints[i - 1, j, 1] + joints[i - 2, j, 1]
    for j in range(7):
        X[0, j, 0] = joints[0, j, 0]
        X[1, j, 0] = joints[0, j, 1]

        X[0, j, 1] = joints[1, j, 0]
        X[1, j, 1] = joints[1, j, 1]

        X[2, j, 1] = joints[1, j, 0] - joints[0, j, 0]
        X[3, j, 1] = joints[1, j, 1] - joints[0, j, 1]

    R = np.zeros((6, 6, 7))
    Q = np.zeros((2, 2, 7))
    for j in range(7):
        for i in range(2, s):
            # print(F.shape)
            # print(X[:,j,i].shape)
            jointji = np.reshape(X[:, j, i], (6, 1))
            jointji_1 = np.reshape(X[:, j, i - 1], (6, 1))
            tmp = jointji - F @ jointji_1
            # print(tmp.shape)
            R[:, :, j] += tmp @ tmp.T
        R[:, :, j] /= (s - 2)
        for i in range(s):
            Q[:, :, j] += np.reshape((est[i, j, :] - joints[i, j, :]), (2, 1)) @ np.reshape(
                (est[i, j, :] - joints[i, j, :]), (1, 2))
        Q[:, :, j] = Q[:, :, j] / float(s)
    return R, Q


def estimateR_with_LSTM(joints, est):
    # 13494,7,2
    # print(joints[0,:,:])
    # print(est[0,:,:])
    s = joints.shape[0]
    R = np.zeros((2, 2, 7))
    for i in range(7):
        for j in range(s):
            tmp = np.reshape(joints[j, i, :] - est[j, i, :], (2, 1))
            R[:, :, i] += tmp @ tmp.T
        R[:, :, i] /= s

    return R
# data_path=os.getcwd()+'/../Data/'
# plot_speed_hist(data_path)
