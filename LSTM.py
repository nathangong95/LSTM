import scipy.io as spio
import numpy as np
import keras.utils
from keras import utils as np_utils
from keras.utils import to_categorical
import h5py
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def loadData(data_path):
    '''
    function to load training data, training label, and time stamp
    input: data path
    output: training data, training label, and time stamp
    '''
    Data = spio.loadmat(data_path+'/Folder.mat', squeeze_me=True,struct_as_record=True)
    train_data=[]
    train_data.append(Data['Folder1'])
    train_data.append(Data['Folder2'])
    train_data.append(Data['Folder3'])
    train_data.append(Data['Folder4'])
    train_data.append(Data['Folder5'])
    train_data.append(Data['Folder6'])
    train_label=np.load(data_path+'/train_label.npy')
    for i in range(6):
        a,b,s=train_data[i].shape
        train_data[i]=np.reshape(train_data[i],(14,s))
        train_data[i]=train_data[i].T
        train_data[i]=np.reshape(train_data[i],(s,1,14))
    for i in range(6):
        train_label[i]=to_categorical(train_label[i])

    f=[]
    f.append(h5py.File(data_path+'/t1.mat'))
    f.append(h5py.File(data_path+'/t2.mat'))
    f.append(h5py.File(data_path+'/t3.mat'))
    f.append(h5py.File(data_path+'/t4.mat'))
    f.append(h5py.File(data_path+'/t5.mat'))
    f.append(h5py.File(data_path+'/t6.mat'))
    f.append(h5py.File(data_path+'/t7.mat'))
    return train_data, train_label, f


def trainLSTM(train_data,train_label,Hidden_unit,batch_s,epoch):
    '''
    low level train function
    input: training data, training label, params
    output: model
    '''
    model = Sequential()
    model.add(LSTM(Hidden_unit, input_shape=(1, 14)))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    for i in range(len(train_data)):
        model.fit(train_data[i],train_label[i], batch_size=batch_s, epochs=epoch)
    return model



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=2.0)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    sns.set(font_scale=2)
    
    fig, ax = plt.subplots(1)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=24,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ax.grid(False)
    
    
    
    return fig
def toIntegerLabel(l):
    '''
    low level helper function to transfer one hot label to integer label
    input: one hot label
    output: integer label
    '''
    label=[]
    s,_=l.shape
    for i in range(s):
        label.append(l[i].tolist().index(max(l[i].tolist())))
    return label
def toOnOffSet(integerLabel):
    '''
    low level helper function to transfer integer label to on/off set
    input: integer label
    output: list of on/off set
    '''
    onoff=[]
    lmsk = [(el==1) or (el==3) for el in integerLabel]
    Rmsk = [(el==2) or (el==3) for el in integerLabel]
    i=0
    while i<len(lmsk):
        if lmsk[i]:
            temp=[]
            temp.append('l')
            temp.append(i)
            for j in range(i,len(lmsk)):
                if not lmsk[j]:
                    temp.append(j-1)
                    onoff.append(temp)
                    i=j
                    break
        i+=1
    i=0
    while i<len(Rmsk):
        if Rmsk[i]:
            temp=[]
            temp.append('r')
            temp.append(i)
            for j in range(i,len(Rmsk)):
                if not Rmsk[j]:
                    temp.append(j-1)
                    onoff.append(temp)
                    i=j
                    break
        i+=1
    return onoff
def toPanda_test(model,train_data,whichFolder,f):
    '''
    function to take testing data and model and output the panda prediction
    input: model, testing data, folder and time stamp information
    output: pandas data frame with the prediction result
    '''
    IntegerLabel=toIntegerLabel(model.predict(train_data[whichFolder]))
    OnOff=toOnOffSet(IntegerLabel)
    df=[]
    for i in range(len(OnOff)):
        OnOff[i][1]=f[whichFolder]['color']['kT'][0][OnOff[i][1]]/1000
        OnOff[i][2]=f[whichFolder]['color']['kT'][0][OnOff[i][2]]/1000
        df.append(pd.DataFrame(data={'onset':[OnOff[i][1]],  'offset':[OnOff[i][2]],  'label':[OnOff[i][0]]}))
    df = pd.concat(df, axis=0)
    df.sort_values(by=('onset'), inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def toPanda_train(train_label,whichFolder,f):
    '''
    function that find the pandas data frame for training data
    input: training label and folder and time stamp
    output: pandas data frame of training data
    '''
    IntegerLabel=toIntegerLabel(train_label[whichFolder])
    OnOff=toOnOffSet(IntegerLabel)
    df=[]
    for i in range(len(OnOff)):
        OnOff[i][1]=f[whichFolder]['color']['kT'][0][OnOff[i][1]]/1000
        OnOff[i][2]=f[whichFolder]['color']['kT'][0][OnOff[i][2]]/1000
        df.append(pd.DataFrame(data={'onset':[OnOff[i][1]],  'offset':[OnOff[i][2]],  'label':[OnOff[i][0]]}))
    df = pd.concat(df, axis=0)
    df.sort_values(by=('onset'), inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def score_periods(df_manual, df_pred, params=None, err1thresh=2, err2thresh=5):
    '''
    function to find df_result
    input: pandas training and testing data frames and threshold for err1 and err2
    output: df_result
    '''
    # initialize storage variables/arrays
    err1 = np.empty(len(df_manual)) # err1
    err2 = np.empty(len(df_manual)) # err2
    detected = np.zeros(len(df_manual), dtype=bool) # detected
    
    # Loop through each manual
    for i, (label, offset, onset) in df_manual.iterrows():
                
        # search for nearest neighbor in df_pred
        pred_onsets = df_pred['onset'].values
        pred_offsets = df_pred['offset'].values
        
        i_onset_match = np.argmin(np.abs(pred_onsets-onset))
        i_offset_match = np.argmin(np.abs(pred_offsets - offset))
        
        if label=='b' or i_onset_match != i_offset_match:
            err1[i] = np.nan
            err2[i] = np.nan
            detected[i] = False
            continue
        
        # calculate onset and offset error (err1, err2)
        err1[i] = onset - pred_onsets[i_onset_match]
        err2[i] = offset - pred_offsets[i_offset_match]
        if np.abs(err1[i])>err1thresh or np.abs(err2[i])>err2thresh:
            detected[i] = False
        else:
            detected[i]=True

    df_return = df_manual.copy()
    df_return['err1'] = err1
    df_return['err2'] = err2
    df_return['detected'] = detected
    
    return df_return

def trainning(train_data, train_label,f,params):
    '''
    high level training function
    input: training data, training label, time stamps, training params
    output: df_result and df_pred
    '''
    result=[]
    pred=[]
    for i in range(4):
        train_set=[]
        train_set_label=[]
        test_folder=i
        for j in range(4):
            if j!=i:
                train_set.append(train_data[j])
                train_set_label.append(train_label[j])
        model=trainLSTM(train_set, train_set_label, params[0],params[1],params[2])
        df_manual=toPanda_train(train_label,test_folder,f)
        df_pred=toPanda_test(model,train_data,test_folder,f)
        df_results = score_periods(df_manual, df_pred)
        pred.append(df_pred)
        result.append(df_results)
    return result,pred
def evaResult(results):
    '''
    high level evaluation function
    input: df_result
    output: hit rate, err1, err2
    '''
    total_event=0.0
    total_detected=0.0
    total_err1=0.0
    total_err2=0.0
    for result in results:
        total_event+=len(result)
        total_detected+=sum(result['detected'])
        for i in range(len(result)):
            if result['detected'][i]:
                total_err1+=np.abs(result['err1'][i])
                total_err2+=np.abs(result['err2'][i])
    return (total_detected/total_event,total_err1/total_event,total_err2/total_event)

def demo():
    return None
if __name__ == "__main__":
    demo()