import sys
import os
sys.path.insert(0,os.getcwd()+'/Modules/')
import LSTM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str,
                    help="Path to the data folder")
args = parser.parse_args()
data_path=args.data_path
train_data, train_label, f=LSTM.loadData(data_path)
params=[64,15,1]
result,pred=LSTM.trainning(train_data,train_label,f,params)
print(LSTM.evaResult(result))
for i in range(4):
    print(LSTM.evaResult([result[i]]))