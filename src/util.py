import numpy as np
from operator import itemgetter


def extract_humanEva_label(file_data, which_subject, which_action, which_cam, selected_joints):
    joints_names = ["head", "neck", "thorax", "pelvis", "l_shoulder", "l_elbow", "l_wrist",
                    "r_shoulder", "r_elbow", "r_wrist", "l_knee", "l_ankle", "r_knee", "r_ankle"]
    idx = [10, 11, 15, 16, 20, 21, 25, 26, 30, 31, 35, 36, 40, 41, 45, 46, 50, 51, 55, 56, 60, 61, 65, 66, 70, 71, 75,
           76]
    file_data = file_data.split('\n')
    joints_data = []
    image_id = []
    for dat in file_data:
        splited = dat.split(',')
        names = splited[0].split('/')
        if len(names) == 7:
            if names[3] == which_subject and names[4] == which_action and names[5] == which_cam:
                joints = itemgetter(*idx)(splited)
                joints = [float(i) for i in joints]
                joints = np.asarray(joints)
                joints = np.reshape(joints, (28, 1))
                joints = np.reshape(joints, (14, 2))
                sort_joints = np.zeros((len(selected_joints), 2))
                for i in range(len(selected_joints)):
                    sort_joints[i, :] = joints[joints_names.index(selected_joints[i]), :]
                joints_data.append(sort_joints)
                image_id.append(int(names[6].split('.')[0]))
    # joints_data = np.asarray(joints_data)
    return joints_data, image_id
