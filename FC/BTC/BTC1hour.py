# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 22:04:03 2018

@author: wang
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 14:03:19 2018

@author: wang
"""
from sklearn.cross_validation import train_test_split
from urllib import request
from urllib.error import HTTPError
from http.client import IncompleteRead
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn.functional as F
from torch import optim
import requests
from sendEmail import sendMail
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.externals import joblib
import pytz

true = True
false = False

input_length = 96  #每一次输入多少个价格向量
last_length = 16     #对输出进行判断，最后的多少个价格向量决定输出是否超过最高价/最低价
predict_length = 4  #判断未来出现多少个K线进行输出判断
LONG_INPUT = True
compare_method = "2" #1是三分类，2是二分类，3是排序,4是0.618
ERROR_RATE = 0
COIN = "BTC"
MINUTE = 15
Timeblock = 8 * 3600



class FCNN(torch.nn.Module):
    def __init__(self,input_size):
        super(FCNN, self).__init__()
        self.hidden = 1000
        self.input_size = input_size
        self.ln1 = torch.nn.Linear(self.input_size,self.hidden)
        self.ln2 = torch.nn.Linear(self.hidden,self.hidden)
        self.bn = torch.nn.BatchNorm1d(self.hidden)
        self.out = torch.nn.Linear(self.hidden,2)
    
    def forward(self,x):
        x = self.ln1(x)
        x = F.sigmoid(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.bn(x)#BatchNormal
        x = self.out(x)
        x = F.dropout(x, 0.5, self.training)
        return x


def getData(start,end,plateform,coin,timespan,aggregate):
    if type(start)== int and type(end)== int:
        startTime = start
        endTime = end
    else:
        startTime = int(time.mktime(time.strptime(start, '%Y-%m-%d %H:%M:%S')))
        endTime = int(time.mktime(time.strptime(end, '%Y-%m-%d %H:%M:%S')))
    #print(startTime)
    #print(endTime)
    to = ""
    if coin == 'BTC':
        to = "USDT"
    if coin == "ETC":
        to = "USDT"
    params="http://dev.s20180502.shareted.com/api/coin/price?con_date_start="+str(startTime)+"&con_date_end="+str(endTime)+"&con_plateform="+plateform+"&con_coin_from="+coin+"&con_coin_to="+to+"&con_timespan="+timespan+"&con_aggregat="+str(aggregate)+"&con_regular=002"
    
    #url = "http://dev.s20180502.shareted.com/api/coin/price?con_date_start=1420041600&con_date_end=1546272000&con_plateform=CCCAGG&con_coin_from=ETC&con_coin_to=USD&con_timespan=HOUR&con_aggregat=1&con_regular=002&con_simple=001"
    print(params)
    while True:
        try:
            print("start")
            req = request.Request(params)  # GET方法
            print(1)
            reponse = request.urlopen(req)
            print(2)
            data = reponse.read()
            #print(data)
            reponse.close()
            print("ok")
            break
        except Exception:
            print("error")
            continue
    return parseData(data)
    

def parseData(data):
    data = eval(data)
    #print(data)
    total = data["total"]
    print(total)
    rows = data["rows"]
    dataFrame = pd.DataFrame(np.zeros((total,4)),columns=["Open","High","Low","Close"])
    timeStamp = []
    for i in range(len(rows)):
        dataFrame.iloc[i,0] = rows[i]["open"]
        dataFrame.iloc[i,1] = rows[i]["high"]
        dataFrame.iloc[i,2] = rows[i]["low"]
        dataFrame.iloc[i,3] = rows[i]["close"]
        timeStamp.append(rows[i]["time"])
    dataFrame.index = timeStamp
    return dataFrame
        
def stampToTime(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
    return otherStyleTime    
    
    
def sendMessage(string,coin = COIN,to = "USDT"):
    message = {}
    message["time"] = int(time.time())
    message["type"] = "alarm"
    message["degree"] = 1
    message["coin_from"] = coin
    message["coin_to"] = to
    message["signal"] = "高低预测"
    message["message"] = string
    return message
    
def reNew(path,svmPath,compare,train_acc = 0.85,test_acc = 0.77):
    
    
        
    def higher(x0,x1,index,eRROR_RATE = ERROR_RATE):
        x0_max = x0.loc[:,index].max()
        x1_max = x1.loc[:,index].max()
        if (x1_max - x0_max)/x0_max > eRROR_RATE:
            return 1
        elif (x1_max - x0_max)/x0_max <  -eRROR_RATE:
            return 2
        return 0
    
    def higher2(x0,x1,index,eRROR_RATE = ERROR_RATE):
        x0_max = x0.loc[:,index].max()
        x1_max = x1.loc[:,index].max()
        if (x1_max - x0_max)/x0_max > eRROR_RATE:
            return 1
        return 0
    
    def higher4(x0,x1,index,eRROR_RATE = ERROR_RATE):      
        x0_max = x0.loc[:,index].max()
        x0_min = x0.loc[:,index].min()
        x0_median = (x0_max - x0_min) * 0.618 + x0_min
        x1_max = x1.loc[:,index].max()
        if x1_max> x0_median:
            return 1
        else:
            return 0
    
    def higher3(x0,x1,index,eRROR_RATE = ERROR_RATE):
        x0_max = x0.loc[:,index].max()
        x0_min = x0.loc[:,index].min()
        x0_3 = (x0_max - x0_min) * 0.618 + x0_min
        x0_2 = (x0_max - x0_3) * 0.618 + x0_3
        x1_max = x1.loc[:,index].max()
        if (x1_max - x0_max)/x0_max > eRROR_RATE:
            return 0
        elif (x1_max - x0_max)/x0_max <= eRROR_RATE and x1_max > x0_2:
            return 1
        elif x1_max > x0_3 and x1_max <= x0_2:
            return 2
        else:
            return 3
    
    
    def lower(x0,x1,index,eRROR_RATE = ERROR_RATE):
        x0_min = x0.loc[:,index].min()
        x1_min = x1.loc[:,index].min()
        if (x0_min - x1_min)/x0_min > eRROR_RATE:
            return 1
        elif (x0_min - x1_min)/x0_min <  -eRROR_RATE:
            return 2
        return 0
    
    
    def lower2(x0,x1,index,eRROR_RATE = ERROR_RATE):
        x0_min = x0.loc[:,index].min()
        x1_min = x1.loc[:,index].min()
        if (x0_min - x1_min)/x0_min > eRROR_RATE:
            return 1
        return 0
    
    def lower4(x0,x1,index,eRROR_RATE = ERROR_RATE):
        x0_max = x0.loc[:,index].max()
        x0_min = x0.loc[:,index].min()
        x0_median = x0_min + (x0_max - x0_min) *(1 - 0.618)
        x1_min = x1.loc[:,index].min()
        if x1_min > x0_median:
            return 0
        else:
            return 1

    def lower3(x0,x1,index,eRROR_RATE = ERROR_RATE):
        x0_max = x0.loc[:,index].max()
        x0_min = x0.loc[:,index].min()
        x0_3 = x0_min + (x0_max - x0_min) * 0.618 
        x0_2 = x0_min + (x0_3 - x0_min) * 0.618
        x1_min = x1.loc[:,index].min()
        if (x0_min - x1_min)/x0_min > eRROR_RATE:
            return 0
        elif (x0_min - x1_min)/x0_min <= eRROR_RATE and x1_min <= x0_2:
            return 1
        elif x1_min <= x0_3 and x1_min > x0_2:
            return 2
        else:
            return 3
    
    train_data = getData(0,int(time.time()),"HUOBI",COIN,"MINUTE",MINUTE)
    train = pd.DataFrame(np.zeros((len(train_data.index),16)))
    train.columns = ["OO","OH","OL","OC","HO","HH","HL","HC",
                     "LO","LH","LL","LC","CO","CH","CL","CC"]
    train.index = train_data.index
    

    train.loc[train.index[1]:train.index[-1],'OO'] = (train_data.loc[train.index[1]:,"Open"].values - train_data.loc[:train.index[-2],"Open"].values)/train_data.loc[:train.index[-2],"Open"].values
    train.loc[train.index[1]:train.index[-1],'OH'] = (train_data.loc[train.index[1]:,"Open"].values - train_data.loc[:train.index[-2],"High"].values)/train_data.loc[:train.index[-2],"High"].values
    train.loc[train.index[1]:train.index[-1],'OL'] = (train_data.loc[train.index[1]:,"Open"].values - train_data.loc[:train.index[-2],"Low"].values)/train_data.loc[:train.index[-2],"Low"].values
    train.loc[train.index[1]:train.index[-1],'OC'] = (train_data.loc[train.index[1]:,"Open"].values - train_data.loc[:train.index[-2],"Close"].values)/train_data.loc[:train.index[-2],"Close"].values 
    train.loc[train.index[1]:train.index[-1],'HO'] = (train_data.loc[train.index[1]:,"High"].values - train_data.loc[:train.index[-2],"Open"].values)/train_data.loc[:train.index[-2],"Open"].values
    train.loc[train.index[1]:train.index[-1],'HH'] = (train_data.loc[train.index[1]:,"High"].values - train_data.loc[:train.index[-2],"High"].values)/train_data.loc[:train.index[-2],"High"].values
    train.loc[train.index[1]:train.index[-1],'HL'] = (train_data.loc[train.index[1]:,"High"].values - train_data.loc[:train.index[-2],"Low"].values)/train_data.loc[:train.index[-2],"Low"].values
    train.loc[train.index[1]:train.index[-1],'HC'] = (train_data.loc[train.index[1]:,"High"].values - train_data.loc[:train.index[-2],"Close"].values)/train_data.loc[:train.index[-2],"Close"].values
    train.loc[train.index[1]:train.index[-1],'LO'] = (train_data.loc[train.index[1]:,"Low"].values - train_data.loc[:train.index[-2],"Open"].values)/train_data.loc[:train.index[-2],"Open"].values
    train.loc[train.index[1]:train.index[-1],'LH'] = (train_data.loc[train.index[1]:,"Low"].values - train_data.loc[:train.index[-2],"High"].values)/train_data.loc[:train.index[-2],"High"].values
    train.loc[train.index[1]:train.index[-1],'LL'] = (train_data.loc[train.index[1]:,"Low"].values - train_data.loc[:train.index[-2],"Low"].values)/train_data.loc[:train.index[-2],"Low"].values
    train.loc[train.index[1]:train.index[-1],'LC'] = (train_data.loc[train.index[1]:,"Low"].values - train_data.loc[:train.index[-2],"Close"].values)/train_data.loc[:train.index[-2],"Close"].values 
    train.loc[train.index[1]:train.index[-1],'CO'] = (train_data.loc[train.index[1]:,"Close"].values - train_data.loc[:train.index[-2],"Open"].values)/train_data.loc[:train.index[-2],"Open"].values
    train.loc[train.index[1]:train.index[-1],'CH'] = (train_data.loc[train.index[1]:,"Close"].values - train_data.loc[:train.index[-2],"High"].values)/train_data.loc[:train.index[-2],"High"].values
    train.loc[train.index[1]:train.index[-1],'CL'] = (train_data.loc[train.index[1]:,"Close"].values - train_data.loc[:train.index[-2],"Low"].values)/train_data.loc[:train.index[-2],"Low"].values
    train.loc[train.index[1]:train.index[-1],'CC'] = (train_data.loc[train.index[1]:,"Close"].values - train_data.loc[:train.index[-2],"Close"].values)/train_data.loc[:train.index[-2],"Close"].values
    train[np.abs(train)<ERROR_RATE] = 0
    train[train>ERROR_RATE] = 1
    train[train<-ERROR_RATE] = -1
    
    
    train = train.iloc[1:,:]
    
    #train = train.astype(np.int)

    


    train_x,train_y = [],[]
    for i in range(train.shape[0] - input_length - predict_length + 1):
        #print(train.iloc[i:i+input_length,:].values.shape)
        if LONG_INPUT:
            train_x.append(train.iloc[i:i+input_length,:].values.reshape(1,-1))
        else:
            train_x.append(train.iloc[i:i+input_length,:].values)
        if compare == "High":
            if compare_method == "2":
                train_y.append([higher2(train_data.iloc[i+input_length-last_length:i+input_length,:],train_data.iloc[i+input_length:i+input_length+predict_length,:],compare,ERROR_RATE)])
            elif compare_method == "1":
                train_y.append([higher(train_data.iloc[i+input_length-last_length:i+input_length,:],train_data.iloc[i+input_length:i+input_length+predict_length,:],compare,ERROR_RATE)])
            elif compare_method == "3":
                train_y.append([higher3(train_data.iloc[i+input_length-last_length:i+input_length,:],train_data.iloc[i+input_length:i+input_length+predict_length,:],compare,ERROR_RATE)])
            elif compare_method == "4": 
                train_y.append([higher4(train_data.iloc[i+input_length-last_length:i+input_length,:],train_data.iloc[i+input_length:i+input_length+predict_length,:],compare,ERROR_RATE)])

        if compare == "Low":
            if compare_method == "2":
                train_y.append([lower2(train_data.iloc[i+input_length-last_length:i+input_length,:],train_data.iloc[i+input_length:i+input_length+predict_length,:],compare,ERROR_RATE)])
            elif compare_method == "1":
                train_y.append([lower(train_data.iloc[i+input_length-last_length:i+input_length,:],train_data.iloc[i+input_length:i+input_length+predict_length,:],compare,ERROR_RATE)])
            elif compare_method == "3":
                train_y.append([lower3(train_data.iloc[i+input_length-last_length:i+input_length,:],train_data.iloc[i+input_length:i+input_length+predict_length,:],compare,ERROR_RATE)])
            elif compare_method == "4":
                train_y.append([lower4(train_data.iloc[i+input_length-last_length:i+input_length,:],train_data.iloc[i+input_length:i+input_length+predict_length,:],compare,ERROR_RATE)])

    train_x = np.array(train_x)
    if LONG_INPUT:
        train_x = train_x.reshape(train_x.shape[0],train_x.shape[2])
    train_y = np.array(train_y)
    train_y = train_y.astype(np.int).reshape(-1)
    print(pd.Series(train_y).value_counts())
    if not (compare_method == "4") or compare_method == "2":
        clf = SVC()
        clf.fit(train_x,train_y)
        joblib.dump(clf, svmPath)
    
    
    
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.33)
    if compare_method == "4":
        train_acc = 0.85
        test_acc = 0.75
        
    continue_train = True
    while continue_train:
        net = FCNN(train_x.shape[1])
        cost = torch.nn.CrossEntropyLoss(torch.Tensor([0.8,0.2]))
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        mse  = []
        epochnum = 100
        for epoch in range(epochnum):
            inputs, labels = torch.autograd.Variable(torch.Tensor(train_x)), torch.autograd.Variable(torch.LongTensor(train_y))
            optimizer.zero_grad()
    
            outputs = net(inputs)
            loss = cost(outputs,labels)
            train_accuracy = f1_score(labels.numpy(), torch.max(outputs.data,1)[1].numpy(), average='micro')
            print("f1:",train_accuracy,end = "    ")
            mse.append(train_accuracy)
            loss.backward()
            optimizer.step()
        
            net.eval()     
            inputs = torch.autograd.Variable(torch.Tensor(val_x))
            true_y = torch.autograd.Variable(torch.LongTensor(val_y))
            predict_y = net(inputs)
            #print(pd.Series(torch.max(predict_y.data,1)[1].numpy()).value_counts()[1]/pd.Series(val_y).value_counts()[1])
            pre_accuracy = f1_score(true_y.numpy(), torch.max(predict_y.data,1)[1].numpy(), average='micro')
            print("validation f1:",pre_accuracy)    
            net.train()
            if train_accuracy > train_acc and pre_accuracy > test_acc:
                continue_train = False
                break 
    torch.save(net, path)
    return (path,svmPath)


if __name__ == '__main__':

# =============================================================================
#     class FCNN(torch.nn.Module):
#         def __init__(self,input_size):
#             super(FCNN, self).__init__()
#             self.hidden = 1000
#             self.input_size = input_size
#             self.ln1 = torch.nn.Linear(self.input_size,self.hidden)
#             self.bn = torch.nn.BatchNorm1d(self.hidden)
#             self.out = torch.nn.Linear(self.hidden,2)
#         
#         def forward(self,x):
#             x = self.ln1(x)
#             x = F.relu(x)
#             x = self.bn(x)#BatchNormal
#             x = self.out(x)
#             x = F.dropout(x, 0.3, self.training)
#             return x
# =============================================================================
        
        
    if compare_method == "2":  
        try:
            high = torch.load(COIN+r"1hourHigh/FC"+COIN+"High"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre"+str(ERROR_RATE)+"error"+str(input_length)+"input2K.pth")
            low = torch.load(COIN+r"1hourLow/FC"+COIN+"Low"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre"+str(ERROR_RATE)+"error"+str(input_length)+"input2K.pth")    
            svmHigh = joblib.load(COIN+r"1hourHigh/SVM"+COIN+"High"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre0.005error"+str(input_length)+"input2K.m")
            svmLow = joblib.load(COIN+r"1hourLow/SVM"+COIN+"Low"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre0.005error"+str(input_length)+"input2K.m")
        except FileNotFoundError:
            highPath = reNew(COIN+r"1hourHigh/FC"+COIN+"High"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre"+str(ERROR_RATE)+"error"+str(input_length)+"input2K.pth",COIN+r"1hourHigh/SVM"+COIN+"High"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre0.005error"+str(input_length)+"input2K.m","High")
            lowPath = reNew(COIN+r"1hourLow/FC"+COIN+"Low"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre"+str(ERROR_RATE)+"error"+str(input_length)+"input2K.pth",COIN+r"1hourLow/SVM"+COIN+"Low"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre0.005error"+str(input_length)+"input2K.m","Low",test_acc = 0.77)
            
            low = torch.load(lowPath[0])
            high = torch.load(highPath[0])
            svmHigh = joblib.load(highPath[1])
            svmLow = joblib.load(lowPath[1])

    
    elif compare_method == "3":       
        high = torch.load(COIN+r"1hourHigh/FC"+COIN+"High"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre0.005error"+str(input_length)+"input3K.pth")
        low = torch.load(COIN+r"1hourLow/FC"+COIN+"Low"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre0.005error"+str(input_length)+"input3K.pth")    

    
    
    while True:
        
        timeNow = datetime.datetime.now()
        timeNow = timeNow.astimezone(pytz.timezone('Asia/Shanghai'))
        if (timeNow.minute + 5) % 15 == 0:
            
            test_data = getData(int(time.time()) - (input_length+110) * MINUTE * 60 ,int(time.time()),"HUOBI",COIN,"MINUTE",MINUTE)
            test_time = test_data.index[-last_length:]
            for i in test_time:
                print(stampToTime(i))
            test = pd.DataFrame(np.zeros((len(test_data.index),16)))
            test.columns = ["OO","OH","OL","OC","HO","HH","HL","HC",
                             "LO","LH","LL","LC","CO","CH","CL","CC"]
            test.index = test_data.index
            test.loc[test.index[1]:test.index[-1],'OO'] = (test_data.loc[test.index[1]:,"Open"].values - test_data.loc[:test.index[-2],"Open"].values)/test_data.loc[:test.index[-2],"Open"].values
            test.loc[test.index[1]:test.index[-1],'OH'] = (test_data.loc[test.index[1]:,"Open"].values - test_data.loc[:test.index[-2],"High"].values)/test_data.loc[:test.index[-2],"High"].values
            test.loc[test.index[1]:test.index[-1],'OL'] = (test_data.loc[test.index[1]:,"Open"].values - test_data.loc[:test.index[-2],"Low"].values)/test_data.loc[:test.index[-2],"Low"].values
            test.loc[test.index[1]:test.index[-1],'OC'] = (test_data.loc[test.index[1]:,"Open"].values - test_data.loc[:test.index[-2],"Close"].values)/test_data.loc[:test.index[-2],"Close"].values
            test.loc[test.index[1]:test.index[-1],'HO'] = (test_data.loc[test.index[1]:,"High"].values - test_data.loc[:test.index[-2],"Open"].values)/test_data.loc[:test.index[-2],"Open"].values
            test.loc[test.index[1]:test.index[-1],'HH'] = (test_data.loc[test.index[1]:,"High"].values - test_data.loc[:test.index[-2],"High"].values)/test_data.loc[:test.index[-2],"High"].values
            test.loc[test.index[1]:test.index[-1],'HL'] = (test_data.loc[test.index[1]:,"High"].values - test_data.loc[:test.index[-2],"Low"].values)/test_data.loc[:test.index[-2],"Low"].values
            test.loc[test.index[1]:test.index[-1],'HC'] = (test_data.loc[test.index[1]:,"High"].values - test_data.loc[:test.index[-2],"Close"].values)/test_data.loc[:test.index[-2],"Close"].values
            test.loc[test.index[1]:test.index[-1],'LO'] = (test_data.loc[test.index[1]:,"Low"].values - test_data.loc[:test.index[-2],"Open"].values)/test_data.loc[:test.index[-2],"Open"].values
            test.loc[test.index[1]:test.index[-1],'LH'] = (test_data.loc[test.index[1]:,"Low"].values - test_data.loc[:test.index[-2],"High"].values)/test_data.loc[:test.index[-2],"High"].values
            test.loc[test.index[1]:test.index[-1],'LL'] = (test_data.loc[test.index[1]:,"Low"].values - test_data.loc[:test.index[-2],"Low"].values)/test_data.loc[:test.index[-2],"Low"].values
            test.loc[test.index[1]:test.index[-1],'LC'] = (test_data.loc[test.index[1]:,"Low"].values - test_data.loc[:test.index[-2],"Close"].values)/test_data.loc[:test.index[-2],"Close"].values
            test.loc[test.index[1]:test.index[-1],'CO'] = (test_data.loc[test.index[1]:,"Close"].values - test_data.loc[:test.index[-2],"Open"].values)/test_data.loc[:test.index[-2],"Open"].values
            test.loc[test.index[1]:test.index[-1],'CH'] = (test_data.loc[test.index[1]:,"Close"].values - test_data.loc[:test.index[-2],"High"].values)/test_data.loc[:test.index[-2],"High"].values
            test.loc[test.index[1]:test.index[-1],'CL'] = (test_data.loc[test.index[1]:,"Close"].values - test_data.loc[:test.index[-2],"Low"].values)/test_data.loc[:test.index[-2],"Low"].values
            test.loc[test.index[1]:test.index[-1],'CC'] = (test_data.loc[test.index[1]:,"Close"].values - test_data.loc[:test.index[-2],"Close"].values)/test_data.loc[:test.index[-2],"Close"].values
            test[np.abs(test)<ERROR_RATE] = 0
            test[test>ERROR_RATE] = 1
            test[test<-ERROR_RATE] = -1    
            test = test.iloc[1:,:]
            test = test.astype(np.int)
            high.eval()
            low.eval()
            predict_x = torch.autograd.Variable(torch.Tensor(test.iloc[-input_length:,:].values.reshape(1,-1)))
            if compare_method == "2":  
                svmHigh_mark =  svmHigh.predict(predict_x.numpy())[0]
                svmLow_mark =  svmLow.predict(predict_x.numpy())[0]
            
            
            if compare_method == "2":  

                predict_high = F.softmax(high(predict_x))
                #print(predict_high)
                high_mark = torch.max(predict_high.data,1)[1].numpy()
                sendmessage = ""
                if high_mark == 0:
                    sendmessage += "没有破新高。"+stampToTime(test_time[-1]+60*MINUTE+Timeblock)+"到"+stampToTime(test_time[-1]+3600+Timeblock)+"没有超过前"+str(int(last_length*MINUTE/60))+"个小时最高"+str(test_data.loc[test_data.index[-last_length:],"High"].max())+"的"+str(ERROR_RATE)+"\n"
                    #print("没有破新高。"+stampToTime(test_time[-1]+60*MINUTE)+"到"+stampToTime(test_time[-1]+3600)+"没有超过前一个小时最高"+str(test_data.loc[test_data.index[-last_length:],"High"].max())+"的"+str(ERROR_RATE))
                if high_mark == 1:
                    sendmessage += "破新高！"+stampToTime(test_time[-1]+60*MINUTE+Timeblock)+"到"+stampToTime(test_time[-1]+3600+Timeblock)+"超过前"+str(int(last_length*MINUTE/60))+"个小时最高"+str(test_data.loc[test_data.index[-last_length:],"High"].max())+"的"+str(ERROR_RATE)+"\n"
                    #print("破新高！"+stampToTime(test_time[-1]+60*MINUTE)+"到"+stampToTime(test_time[-1]+3600)+"超过前一个小时最高"+str(test_data.loc[test_data.index[-last_length:],"High"].max())+"的"+str(ERROR_RATE))
                sendmessage += "破新高的概率是"+str(predict_high.detach().numpy()[0][1])+"  没有破新高的概率是"+str(predict_high.detach().numpy()[0][0])+"\n"
                if svmHigh_mark == 0:
                    sendmessage +="SVM的预测是不破新高。" +"\n"+"\n"
                if svmHigh_mark == 1:
                    sendmessage +="SVM的预测是破新高。" +"\n"+"\n"
        
                #print("破新高的概率是"+str(predict_high.detach().numpy()[0][1])+"  没有破新高的概率是"+str(predict_high.detach().numpy()[0][0]))
                predict_low = F.softmax(low(predict_x))
                low_mark = torch.max(predict_low.data,1)[1].numpy()
                #print(predict_low)
    
                if low_mark == 0:
                    sendmessage+="没有破新低。"+stampToTime(test_time[-1]+60*MINUTE+Timeblock)+"到"+stampToTime(test_time[-1]+3600+Timeblock)+"没有超过前"+str(int(last_length*MINUTE/60))+"个小时最低"+str(test_data.loc[test_data.index[-last_length:],"Low"].min())+"的"+str(ERROR_RATE)+"\n"
                    #print("没有破新低。"+stampToTime(test_time[-1]+60*MINUTE)+"到"+stampToTime(test_time[-1]+3600)+"没有超过前一个小时最低"+str(test_data.loc[test_data.index[-last_length:],"Low"].min())+"的"+str(ERROR_RATE))
                if low_mark == 1:
                    sendmessage += "破新低！"+stampToTime(test_time[-1]+60*MINUTE+Timeblock)+"到"+stampToTime(test_time[-1]+3600+Timeblock)+"超过前"+str(int(last_length*MINUTE/60))+"个小时最低"+str(test_data.loc[test_data.index[-last_length:],"Low"].min())+"的"+str(ERROR_RATE)+"\n"
                    #print("破新低！"+stampToTime(test_time[-1]+60*MINUTE)+"到"+stampToTime(test_time[-1]+3600)+"超过前一个小时最低"+str(test_data.loc[test_data.index[-last_length:],"Low"].min())+"的"+str(ERROR_RATE))
                sendmessage+="破新低的概率是"+str(predict_low.detach().numpy()[0][1])+"  没有破新低的概率是"+str(predict_low.detach().numpy()[0][0])+"\n"
                if svmLow_mark == 0:
                    sendmessage +="SVM的预测是不破新低。" +"\n"+"\n"
                if svmLow_mark == 1:
                    sendmessage +="SVM的预测是破新高。" +"\n"+"\n"
                if ((low_mark == 1 or high_mark == 1) and ((timeNow.minute+5) % 60 != 0)) or ((timeNow.minute+5) % 60 == 0):
                    #print("破新低的概率是"+str(predict_low.detach().numpy()[0][1])+"  没有破新低的概率是"+str(predict_low.detach().numpy()[0][0]))
                    print(sendmessage)
                    #response = requests.post('http://dev.s20180502.shareted.com/openapi/coin/signal/report',data=sendMessage(sendmessage))
                    # 通过get请求返回的文本值
                    #print(response.text)
                    title = COIN+" - 1H 预测"+str(timeNow.year)+str(timeNow.month)+str(timeNow.day)+" "+ stampToTime(test_time[-1]+60*MINUTE+Timeblock)[-8:-3]+"新高"+'%.2f%%' % ( 100*predict_high.detach().numpy()[0][1])+",新低"+'%.2f%%' % ( 100*predict_low.detach().numpy()[0][1]) 
                    print(title)
                    sendMail(title,sendmessage)
            time.sleep(60)
        if timeNow.hour % 3 == 0 and timeNow.minute == 20:
            highPath = reNew(COIN+r"1hourHigh/FC"+COIN+"High"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre"+str(ERROR_RATE)+"error"+str(input_length)+"input2K.pth",COIN+r"1hourHigh/SVM"+COIN+"High"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre0.005error"+str(input_length)+"input2K.m","High")
            lowPath = reNew(COIN+r"1hourLow/FC"+COIN+"Low"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre"+str(ERROR_RATE)+"error"+str(input_length)+"input2K.pth",COIN+r"1hourLow/SVM"+COIN+"Low"+str(MINUTE)+"min"+str(int(last_length*MINUTE/60))+"hourlast"+str(int(predict_length*MINUTE/60))+"hourpre0.005error"+str(input_length)+"input2K.m","Low",test_acc = 0.75)
            
            low = torch.load(lowPath[0])
            high = torch.load(highPath[0])
            svmHigh = joblib.load(highPath[1])
            svmLow = joblib.load(lowPath[1])

            
            
            time.sleep(60)
