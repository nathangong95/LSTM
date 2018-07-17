import LSTM
data_path='/Users/user/Desktop/LSTMTrain'
train_data, train_label, f=LSTM.loadData(data_path)
params=[64,15,1]
result,pred=LSTM.trainning(train_data,train_label,f,params)
print(LSTM.evaResult(result))
for i in range(4):
    print(LSTM.evaResult([result[i]]))