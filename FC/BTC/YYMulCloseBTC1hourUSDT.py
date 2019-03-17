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
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sns.set_style('whitegrid')
true = True
false = False
send_time = 1
input_length = 192  #每一次输入多少个价格向量
predict_length = 16  #判断未来出现多少个K线进行输出判断
LONG_INPUT = True
compare_method = "2" #1是三分类，2是二分类，3是排序,4是0.618
ERROR_RATE = 0
COIN = "BTC"
MINUTE = 15
Timeblock = 8 * 3600
MulEncoding = 13
ERROR_RATE_Vol = 0.2
K = 5
MAn = [K,20,30,60,80,120]

MAMethod = "MA"


class FCNN(torch.nn.Module):
    def __init__(self,input_size):
        super(FCNN, self).__init__()
        self.hidden = 300
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
        x = F.dropout(x, 0.4, self.training)
        return x


def EMA(data,n):
    a = 2/(n+1)
    data = data.copy().dropna(axis = 0)
    #print(data.iloc[1])
    for i in range(1,len(data)):
        data.iloc[i] = a*data.iloc[i] + (1-a)*data.iloc[i-1]
    return data

def MA(data,m,n,index):
    for ma in m:
        data.loc[:,MAMethod+str(ma)+index] = data.loc[:,index].rolling(ma).mean()
        if ma==K and index == 'Close':
            dataIndex = data.index[-100:]
            #data.loc[dataIndex,index].plot(color='g',label="Close")
            #data.loc[dataIndex,MAMethod+str(ma)+index].plot(color ='r',label="SMA")
            a = EMA(data.loc[:,MAMethod+str(ma)+index],n)
           # a.loc[dataIndex].plot(color='y',label = 'EMA1')
            for i in range(2 - 1):
                a =EMA(a,n)
               # a.loc[dataIndex].plot(label = 'EMA'+str(i+1))

            
            data.loc[:,MAMethod+str(ma)+index] =  a      
            #data.loc[dataIndex,MAMethod+str(ma)+index].plot(color ='b',label='EMA3')
            #plt.legend(loc="upper right")
            #plt.show()
    data = data.dropna(axis = 0)
    return data  

def TRIX(data):
    data = data.copy()
    for i in range(1,len(data.index)):
        data.iloc[i] = (data.iloc[i] - data.iloc[i-1]) / data.iloc[i-1] * 100
    data = data.dropna(axis = 0)
    return data

def MATRIX(data,m,n,index):
    for ma in m:
        TR = EMA(EMA(EMA(data.loc[:,index],n),n),n)
        trix = TRIX(TR)
        #print(trix)
        data.loc[:,"MATRIX"+str(ma)+index] = trix.rolling(ma).mean()
    data = data.dropna(axis = 0)
    return data

def getData(start,end,plateform,coin,timespan,aggregate,to="USD_FUTURE_QUARTER"):
    if type(start)== int and type(end)== int:
        startTime = start
        endTime = end
    else:
        startTime = int(time.mktime(time.strptime(start, '%Y-%m-%d %H:%M:%S')))
        endTime = int(time.mktime(time.strptime(end, '%Y-%m-%d %H:%M:%S')))
    #print(startTime)
    #print(endTime)
    params="http://order.shareted.com/api/coin/price?con_date_start="+str(startTime)+"&con_date_end="+str(endTime)+"&con_plateform="+plateform+"&con_coin_from="+coin+"&con_coin_to="+to+"&con_timespan="+timespan+"&con_aggregat="+str(aggregate)+"&con_regular=002"
    
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
            return parseData(eval(data))
        except HTTPError:
            print("Httperror")
            continue
        except Exception:
            print("error")
            continue

def parseData(data):
    #data = eval(data)
    #print(data)
    total = data["total"]
    print(total)
    if total < 0:
        raise Exception
    rows = data["rows"]
    dataFrame = pd.DataFrame(np.zeros((total,5)),columns=["Open","High","Low","Close","Volume"])
    timeStamp = []
    for i in range(len(rows)):
        dataFrame.iloc[i,0] = rows[i]["open"]
        dataFrame.iloc[i,1] = rows[i]["high"]
        dataFrame.iloc[i,2] = rows[i]["low"]
        dataFrame.iloc[i,3] = rows[i]["close"]
        dataFrame.iloc[i,4] = rows[i]["count"]
        timeStamp.append(rows[i]["time"])
    dataFrame.index = timeStamp
    dataFrame = MA(dataFrame,MAn,4,"Close")
    dataFrame = MA(dataFrame,[5],10,'Volume')
    return dataFrame
        
def stampToTime(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
    return otherStyleTime    
    
    
def sendMessage(string,coin = COIN,to = "USD_FUTURE_QUARTER"):
    message = {}
    message["time"] = int(time.time())
    message["type"] = "alarm"
    message["degree"] = 1
    message["coin_from"] = coin
    message["coin_to"] = to
    message["signal"] = "High_Low_Predict"
    message["message"] = string
    return message
    
      
            
def reNew(path,pcapath,compare,train_acc = 0.85,test_acc = 0.8):
    
    def higherDiff(x0,x1,index1,index2,eRROR_RATE = ERROR_RATE):
        x0_max = x0.loc[:,index1].max()
        x1_max = x1.loc[:,index2].max()
        if (x1_max - x0_max)/x0_max > eRROR_RATE:
            return 1
        return 0
        
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
    
    def lowerDiff(x0,x1,index1,index2,eRROR_RATE = ERROR_RATE):
        x0_min = x0.loc[:,index1].min()
        x1_min = x1.loc[:,index2].min()
        if (x0_min - x1_min)/x0_min > eRROR_RATE:
            return 1
        return 0
    
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
        
    
    def YinYang(x0,eRROR_RATE = 0.03):
        eRROR_RATE = 0
        if (x0.loc[:,"High"].max() - x0.loc[:,"Low"].min()) / x0.loc[:,"Low"].min() < eRROR_RATE:
            return 2
        elif (x0.loc[x0.index[0],"Open"] - x0.loc[x0.index[-1],"Close"]) / x0.loc[x0.index[-1],"Close"] < 0:
            return 0
        elif (x0.loc[x0.index[0],"Open"] - x0.loc[x0.index[-1],"Close"]) / x0.loc[x0.index[-1],"Close"] >= 0:
            return 1
            
    #MulEncodingArray = [2,4,6,8,10,12,14,16,18,20]
    train_data = getData(int(time.time())-900000,int(time.time()),"HUOBI",COIN,"MINUTE",MINUTE,to="USDT")

    train = pd.DataFrame(np.zeros((len(train_data.index),16*(MulEncoding-1))))
    columns = []
    for i in range(1,MulEncoding):
        columns.extend(["OO"+str(i),"OH"+str(i),"OL"+str(i),"OC"+str(i),"HO"+str(i),"HH"+str(i),"HL"+str(i),"HC"+str(i),
                     "LO"+str(i),"LH"+str(i),"LL"+str(i),"LC"+str(i),"CO"+str(i),"CH"+str(i),"CL"+str(i),"CC"+str(i)])
    train.columns = columns
    #print(train.columns)
    train.index = train_data.index

    for i in range(1,MulEncoding):
            #lamda = -100
# =============================================================================
#             train.loc[train.index[i]:train.index[-1],'OO'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Open"].values - train_data.loc[:train.index[-1-i],"Open"].values)/train_data.loc[:train.index[-1-i],"Open"].values,2))
#             train.loc[train.index[i]:train.index[-1],'OH'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Open"].values - train_data.loc[:train.index[-1-i],"High"].values)/train_data.loc[:train.index[-1-i],"High"].values,2))
#             train.loc[train.index[i]:train.index[-1],'OL'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Open"].values - train_data.loc[:train.index[-1-i],"Low"].values)/train_data.loc[:train.index[-1-i],"Low"].values,2))
#             train.loc[train.index[i]:train.index[-1],'OC'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Open"].values - train_data.loc[:train.index[-1-i],"Close"].values)/train_data.loc[:train.index[-1-i],"Close"].values,2))
#             train.loc[train.index[i]:train.index[-1],'HO'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"High"].values - train_data.loc[:train.index[-1-i],"Open"].values)/train_data.loc[:train.index[-1-i],"Open"].values,2))
#             train.loc[train.index[i]:train.index[-1],'HH'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"High"].values - train_data.loc[:train.index[-1-i],"High"].values)/train_data.loc[:train.index[-1-i],"High"].values,2))
#             train.loc[train.index[i]:train.index[-1],'HL'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"High"].values - train_data.loc[:train.index[-1-i],"Low"].values)/train_data.loc[:train.index[-1-i],"Low"].values,2))
#             train.loc[train.index[i]:train.index[-1],'HC'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"High"].values - train_data.loc[:train.index[-1-i],"Close"].values)/train_data.loc[:train.index[-1-i],"Close"].values,2))
#             train.loc[train.index[i]:train.index[-1],'LO'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Low"].values - train_data.loc[:train.index[-1-i],"Open"].values)/train_data.loc[:train.index[-1-i],"Open"].values,2))
#             train.loc[train.index[i]:train.index[-1],'LH'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Low"].values - train_data.loc[:train.index[-1-i],"High"].values)/train_data.loc[:train.index[-1-i],"High"].values,2))
#             train.loc[train.index[i]:train.index[-1],'LL'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Low"].values - train_data.loc[:train.index[-1-i],"Low"].values)/train_data.loc[:train.index[-1-i],"Low"].values,2))
#             train.loc[train.index[i]:train.index[-1],'LC'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Low"].values - train_data.loc[:train.index[-1-i],"Close"].values)/train_data.loc[:train.index[-1-i],"Close"].values ,2)) 
#             train.loc[train.index[i]:train.index[-1],'CO'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Close"].values - train_data.loc[:train.index[-1-i],"Open"].values)/train_data.loc[:train.index[-1-i],"Open"].values,2))
#             train.loc[train.index[i]:train.index[-1],'CH'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Close"].values - train_data.loc[:train.index[-1-i],"High"].values)/train_data.loc[:train.index[-1-i],"High"].values,2))
#             train.loc[train.index[i]:train.index[-1],'CL'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Close"].values - train_data.loc[:train.index[-1-i],"Low"].values)/train_data.loc[:train.index[-1-i],"Low"].values,2))
#             train.loc[train.index[i]:train.index[-1],'CC'+str(i)] = np.exp(-lamda*np.power((train_data.loc[train.index[i]:,"Close"].values - train_data.loc[:train.index[-1-i],"Close"].values)/train_data.loc[:train.index[-1-i],"Close"].values,2))
#     
# =============================================================================

        train.loc[train.index[i]:train.index[-1],'OO'+str(i)] = (train_data.loc[train.index[i]:,"Open"].values - train_data.loc[:train.index[-1-i],"Open"].values)/train_data.loc[:train.index[-1-i],"Open"].values
        train.loc[train.index[i]:train.index[-1],'OH'+str(i)] = (train_data.loc[train.index[i]:,"Open"].values - train_data.loc[:train.index[-1-i],"High"].values)/train_data.loc[:train.index[-1-i],"High"].values
        train.loc[train.index[i]:train.index[-1],'OL'+str(i)] = (train_data.loc[train.index[i]:,"Open"].values - train_data.loc[:train.index[-1-i],"Low"].values)/train_data.loc[:train.index[-1-i],"Low"].values
        train.loc[train.index[i]:train.index[-1],'OC'+str(i)] = (train_data.loc[train.index[i]:,"Open"].values - train_data.loc[:train.index[-1-i],"Close"].values)/train_data.loc[:train.index[-1-i],"Close"].values 
        train.loc[train.index[i]:train.index[-1],'HO'+str(i)] = (train_data.loc[train.index[i]:,"High"].values - train_data.loc[:train.index[-1-i],"Open"].values)/train_data.loc[:train.index[-1-i],"Open"].values
        train.loc[train.index[i]:train.index[-1],'HH'+str(i)] = (train_data.loc[train.index[i]:,"High"].values - train_data.loc[:train.index[-1-i],"High"].values)/train_data.loc[:train.index[-1-i],"High"].values
        train.loc[train.index[i]:train.index[-1],'HL'+str(i)] = (train_data.loc[train.index[i]:,"High"].values - train_data.loc[:train.index[-1-i],"Low"].values)/train_data.loc[:train.index[-1-i],"Low"].values
        train.loc[train.index[i]:train.index[-1],'HC'+str(i)] = (train_data.loc[train.index[i]:,"High"].values - train_data.loc[:train.index[-1-i],"Close"].values)/train_data.loc[:train.index[-1-i],"Close"].values
        train.loc[train.index[i]:train.index[-1],'LO'+str(i)] = (train_data.loc[train.index[i]:,"Low"].values - train_data.loc[:train.index[-1-i],"Open"].values)/train_data.loc[:train.index[-1-i],"Open"].values
        train.loc[train.index[i]:train.index[-1],'LH'+str(i)] = (train_data.loc[train.index[i]:,"Low"].values - train_data.loc[:train.index[-1-i],"High"].values)/train_data.loc[:train.index[-1-i],"High"].values
        train.loc[train.index[i]:train.index[-1],'LL'+str(i)] = (train_data.loc[train.index[i]:,"Low"].values - train_data.loc[:train.index[-1-i],"Low"].values)/train_data.loc[:train.index[-1-i],"Low"].values
        train.loc[train.index[i]:train.index[-1],'LC'+str(i)] = (train_data.loc[train.index[i]:,"Low"].values - train_data.loc[:train.index[-1-i],"Close"].values)/train_data.loc[:train.index[-1-i],"Close"].values 
        train.loc[train.index[i]:train.index[-1],'CO'+str(i)] = (train_data.loc[train.index[i]:,"Close"].values - train_data.loc[:train.index[-1-i],"Open"].values)/train_data.loc[:train.index[-1-i],"Open"].values
        train.loc[train.index[i]:train.index[-1],'CH'+str(i)] = (train_data.loc[train.index[i]:,"Close"].values - train_data.loc[:train.index[-1-i],"High"].values)/train_data.loc[:train.index[-1-i],"High"].values
        train.loc[train.index[i]:train.index[-1],'CL'+str(i)] = (train_data.loc[train.index[i]:,"Close"].values - train_data.loc[:train.index[-1-i],"Low"].values)/train_data.loc[:train.index[-1-i],"Low"].values
        train.loc[train.index[i]:train.index[-1],'CC'+str(i)] = (train_data.loc[train.index[i]:,"Close"].values - train_data.loc[:train.index[-1-i],"Close"].values)/train_data.loc[:train.index[-1-i],"Close"].values
    
    for i in range(len(MAn)):
        train.loc[:,"OMA"+str(MAn[i])] = (train_data.loc[:,"Open"] - train_data.loc[:,MAMethod+str(MAn[i])+"Close"]) / train_data.loc[:,MAMethod+str(MAn[i])+"Close"]
        train.loc[:,"CMA"+str(MAn[i])] = (train_data.loc[:,"Close"] - train_data.loc[:,MAMethod+str(MAn[i])+"Close"]) / train_data.loc[:,MAMethod+str(MAn[i])+"Close"]
        train.loc[:,"HMA"+str(MAn[i])] = (train_data.loc[:,"High"] - train_data.loc[:,MAMethod+str(MAn[i])+"Close"]) / train_data.loc[:,MAMethod+str(MAn[i])+"Close"]
        train.loc[:,"LMA"+str(MAn[i])] = (train_data.loc[:,"Low"] - train_data.loc[:,MAMethod+str(MAn[i])+"Close"]) / train_data.loc[:,MAMethod+str(MAn[i])+"Close"]

    train[np.abs(train)<ERROR_RATE] = 0
    train[train>ERROR_RATE] = 1
    train[train<-ERROR_RATE] = -1

    VolMA = train_data.loc[:,'Volume'] / train_data.loc[:,'MA5Volume']
    VolMA[np.abs(VolMA)-1 <ERROR_RATE_Vol] = 0    
    VolMA[VolMA >= 1+ERROR_RATE_Vol] = 1
    VolMA[VolMA <= 1-ERROR_RATE_Vol] = -1
    train.loc[:,"VolMA"] = VolMA
    #train = pd.concat([train, VolMA], axis=1)
    
    
    
    train = train.iloc[MulEncoding-1:,:]

    
    #train = train.astype(np.int)

    
    #train = train.astype(np.int)

    

    train_x,train_y = [],[]
    for i in range(train.shape[0] - input_length - predict_length + 1):
        #print(train.iloc[i:i+input_length,:].values.shape)
        if LONG_INPUT:
            train_x.append(train.iloc[i:i+input_length,:].values.reshape(1,-1))
        else:
            train_x.append(train.iloc[i:i+input_length,:].values)
        if compare =='YY':
            #print(train_data.loc[train.index[i+input_length:i+input_length+predict_length],"MA5Close"].values)
            train_y.append([YinYang(train_data.loc[train.index[i+input_length:i+input_length+predict_length],:],0)])


    train_x = np.array(train_x)
    if LONG_INPUT:
        train_x = train_x.reshape(train_x.shape[0],train_x.shape[2])
    train_y = np.array(train_y)
    train_y = train_y.astype(np.int).reshape(-1)
    #print(train_y)
    print(pd.Series(train_y).value_counts())
    print(train_x.shape)
    pca =LocallyLinearEmbedding(n_components=100)

    train_x = pca.fit_transform(train_x)
# =============================================================================
#     fig = plt.figure()
#     #ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
#     train_x = pd.DataFrame(train_x)
#     train_x.loc[:,'type'] = train_y
#     #plt.scatter(train_x[train_x.type == 1].iloc[:, 0], train_x[train_x.type == 1].iloc[:, 1],train_x[train_x.type == 1].iloc[:, 2], marker='o',color = 'r',label = 'higher')
#     #plt.scatter(train_x[train_x.type == 0].iloc[:, 0], train_x[train_x.type == 0].iloc[:, 1],train_x[train_x.type == 0].iloc[:, 2], marker='o',color = 'blue',label = 'not higher')
#     plt.scatter(train_x[train_x.type == 1].iloc[:, 2], train_x[train_x.type == 1].iloc[:, 1], marker='o',color = 'r',label = 'higher')
#     plt.scatter(train_x[train_x.type == 0].iloc[:, 2], train_x[train_x.type == 0].iloc[:, 1], marker='o',color = 'blue',label = 'not higher')
#     plt.show()
# =============================================================================
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.33)

        
    continue_train = True
    while continue_train:
        net = FCNN(train_x.shape[1])
        cost = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        mse  = []
        epochnum = 100
        for epoch in range(epochnum):
            inputs, labels = torch.autograd.Variable(torch.Tensor(train_x)), torch.autograd.Variable(torch.LongTensor(train_y))
            optimizer.zero_grad()
            outputs = net(inputs)
            #print(outputs)
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
    joblib.dump(pca, pcapath)
    return (path,pcapath)

def backforward(data):
    a = data.loc[:,MAMethod+str(K)+'Close']
    count = (a.iloc[-1] - a.iloc[-2])/abs(a.iloc[-1] - a.iloc[-2])
    for i in range(len(a)-2,0,-1):
        #print(i,a[i],a[i-1])
        c = (a.iloc[i] - a.iloc[i-1])/a.iloc[i-1]
        if abs(c) < ERROR_RATE:
            return 0
        b = (a.iloc[i] - a.iloc[i-1])/abs(a.iloc[i] - a.iloc[i-1])
        #print(b)
        if count*b > 0:
            count += b
        elif abs(count) >= 2:
            return int(count)
        else:
            return 0
    return int(count)

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
            high = torch.load(COIN+r"YY/FC"+str(MulEncoding)+COIN+"yy"+str(MINUTE)+"min"+str(int(predict_length*MINUTE/60))+"hourpre"+str(ERROR_RATE)+"error"+str(input_length)+"input2K.pth")
            highPca = joblib.load(COIN+r"YY/FCPCAMA"+str(predict_length)+".m")

        except FileNotFoundError:
            yyPath = reNew(COIN+r"YY/FC"+str(MulEncoding)+COIN+"yy"+str(MINUTE)+"min"+str(int(predict_length*MINUTE/60))+"hourpre"+str(ERROR_RATE)+"error"+str(input_length)+"input2K.pth",COIN+r"YY/FCPCAMA"+str(predict_length)+".m","YY")
            
            high = torch.load(yyPath[0])
            highPca = joblib.load(yyPath[1])
    

    
    
    while True:
        
        timeNow = datetime.datetime.now()
        timeNow = timeNow.astimezone(pytz.timezone('Asia/Shanghai'))
        if (timeNow.minute + send_time) % 60 == 0:
            
            test_data = getData(int(time.time()) - 400000 ,int(time.time()),"HUOBI",COIN,"MINUTE",MINUTE,to="USDT")

            if timeNow.timestamp() - test_data.index[-1]>15*60:
                sendmessage = stampToTime(timeNow.timestamp()+Timeblock)[6:9]+stampToTime(timeNow.timestamp()+Timeblock)[10:18]+" - "+COIN+"季度"+str(int(predict_length/4))+"H收盘价数据获取失败，最新数据"+stampToTime(test_data.index[-1]+Timeblock).replace("--",'-')
                response = requests.post('http://order.shareted.com/openapi/coin/signal/report',data=sendMessage(sendmessage))
                time.sleep(60)
                continue
            test = pd.DataFrame(np.zeros((len(test_data.index),16*(MulEncoding-1))))
            columns = []
            for i in range(1,MulEncoding):
                columns.extend(["OO"+str(i),"OH"+str(i),"OL"+str(i),"OC"+str(i),"HO"+str(i),"HH"+str(i),"HL"+str(i),"HC"+str(i),
                             "LO"+str(i),"LH"+str(i),"LL"+str(i),"LC"+str(i),"CO"+str(i),"CH"+str(i),"CL"+str(i),"CC"+str(i)])
            test.columns = columns
            test.index = test_data.index
            for i in range(1,MulEncoding):
                test.loc[test.index[i]:test.index[-1],'OO'+str(i)] = (test_data.loc[test.index[i]:,"Open"].values - test_data.loc[:test.index[-1-i],"Open"].values)/test_data.loc[:test.index[-1-i],"Open"].values
                test.loc[test.index[i]:test.index[-1],'OH'+str(i)] = (test_data.loc[test.index[i]:,"Open"].values - test_data.loc[:test.index[-1-i],"High"].values)/test_data.loc[:test.index[-1-i],"High"].values
                test.loc[test.index[i]:test.index[-1],'OL'+str(i)] = (test_data.loc[test.index[i]:,"Open"].values - test_data.loc[:test.index[-1-i],"Low"].values)/test_data.loc[:test.index[-1-i],"Low"].values
                test.loc[test.index[i]:test.index[-1],'OC'+str(i)] = (test_data.loc[test.index[i]:,"Open"].values - test_data.loc[:test.index[-1-i],"Close"].values)/test_data.loc[:test.index[-1-i],"Close"].values 
                test.loc[test.index[i]:test.index[-1],'HO'+str(i)] = (test_data.loc[test.index[i]:,"High"].values - test_data.loc[:test.index[-1-i],"Open"].values)/test_data.loc[:test.index[-1-i],"Open"].values
                test.loc[test.index[i]:test.index[-1],'HH'+str(i)] = (test_data.loc[test.index[i]:,"High"].values - test_data.loc[:test.index[-1-i],"High"].values)/test_data.loc[:test.index[-1-i],"High"].values
                test.loc[test.index[i]:test.index[-1],'HL'+str(i)] = (test_data.loc[test.index[i]:,"High"].values - test_data.loc[:test.index[-1-i],"Low"].values)/test_data.loc[:test.index[-1-i],"Low"].values
                test.loc[test.index[i]:test.index[-1],'HC'+str(i)] = (test_data.loc[test.index[i]:,"High"].values - test_data.loc[:test.index[-1-i],"Close"].values)/test_data.loc[:test.index[-1-i],"Close"].values
                test.loc[test.index[i]:test.index[-1],'LO'+str(i)] = (test_data.loc[test.index[i]:,"Low"].values - test_data.loc[:test.index[-1-i],"Open"].values)/test_data.loc[:test.index[-1-i],"Open"].values
                test.loc[test.index[i]:test.index[-1],'LH'+str(i)] = (test_data.loc[test.index[i]:,"Low"].values - test_data.loc[:test.index[-1-i],"High"].values)/test_data.loc[:test.index[-1-i],"High"].values
                test.loc[test.index[i]:test.index[-1],'LL'+str(i)] = (test_data.loc[test.index[i]:,"Low"].values - test_data.loc[:test.index[-1-i],"Low"].values)/test_data.loc[:test.index[-1-i],"Low"].values
                test.loc[test.index[i]:test.index[-1],'LC'+str(i)] = (test_data.loc[test.index[i]:,"Low"].values - test_data.loc[:test.index[-1-i],"Close"].values)/test_data.loc[:test.index[-1-i],"Close"].values 
                test.loc[test.index[i]:test.index[-1],'CO'+str(i)] = (test_data.loc[test.index[i]:,"Close"].values - test_data.loc[:test.index[-1-i],"Open"].values)/test_data.loc[:test.index[-1-i],"Open"].values
                test.loc[test.index[i]:test.index[-1],'CH'+str(i)] = (test_data.loc[test.index[i]:,"Close"].values - test_data.loc[:test.index[-1-i],"High"].values)/test_data.loc[:test.index[-1-i],"High"].values
                test.loc[test.index[i]:test.index[-1],'CL'+str(i)] = (test_data.loc[test.index[i]:,"Close"].values - test_data.loc[:test.index[-1-i],"Low"].values)/test_data.loc[:test.index[-1-i],"Low"].values
                test.loc[test.index[i]:test.index[-1],'CC'+str(i)] = (test_data.loc[test.index[i]:,"Close"].values - test_data.loc[:test.index[-1-i],"Close"].values)/test_data.loc[:test.index[-1-i],"Close"].values
            
            for i in range(len(MAn)):
                test.loc[:,"OMA"+str(MAn[i])] = (test_data.loc[:,"Open"] - test_data.loc[:,"MA"+str(MAn[i])+"Close"]) / test_data.loc[:,"MA"+str(MAn[i])+"Close"]
                test.loc[:,"CMA"+str(MAn[i])] = (test_data.loc[:,"Close"] - test_data.loc[:,"MA"+str(MAn[i])+"Close"]) / test_data.loc[:,"MA"+str(MAn[i])+"Close"]
                test.loc[:,"HMA"+str(MAn[i])] = (test_data.loc[:,"High"] - test_data.loc[:,"MA"+str(MAn[i])+"Close"]) / test_data.loc[:,"MA"+str(MAn[i])+"Close"]
                test.loc[:,"LMA"+str(MAn[i])] = (test_data.loc[:,"Low"] - test_data.loc[:,"MA"+str(MAn[i])+"Close"]) / test_data.loc[:,"MA"+str(MAn[i])+"Close"]

            
            
            
            
            test[np.abs(test)<ERROR_RATE] = 0
            test[test>ERROR_RATE] = 1
            test[test<-ERROR_RATE] = -1
            VolMA = test_data.loc[:,'Volume'] / test_data.loc[:,'MA5Volume']
            VolMA[np.abs(VolMA)-1 <ERROR_RATE_Vol] = 0    
            VolMA[VolMA >= 1+ERROR_RATE_Vol] = 1
            VolMA[VolMA <= 1-ERROR_RATE_Vol] = -1
            test.loc[:,"VolMA"] = VolMA

    
    
    
            title = ""    
            test = test.iloc[-input_length:,:].values.reshape(1,-1)
            testHigh = highPca.transform(test)


            high.eval()
 
            predict_x_high = torch.autograd.Variable(torch.Tensor(testHigh))
            
            
            
            if compare_method == "2":  
                sendmessage ="【"+COIN+"美"+str(int(predict_length/4))+"H-"
                predict_high = F.softmax(high(predict_x_high))
                high_mark = torch.max(predict_high.data,1)[1].numpy()
    
                    
                    
                if high_mark == 1:
                    title += "阳线!"
                    sendmessage += "阳线！"+stampToTime(timeNow.timestamp()+Timeblock)[13:18]+'〗'+'\n'
                    sendmessage += stampToTime(test_data.index[-1]+60*MINUTE+Timeblock)[6:18]+"到"+stampToTime(test_data.index[-1]+int(60*MINUTE*predict_length)+Timeblock)[6:18]+'\n'
                elif high_mark == 0:
                    title += "阴线!"
                    sendmessage += "阴线！"+stampToTime(timeNow.timestamp()+Timeblock)[13:18]+'〗'+'\n'
                    sendmessage += stampToTime(test_data.index[-1]+60*MINUTE+Timeblock)[6:18]+"到"+stampToTime(test_data.index[-1]+int(60*MINUTE*predict_length)+Timeblock)[6:18]+'\n'
                
                sendmessage += "阳线的概率是"+str('%.2f%%' %(100* predict_high.detach().numpy()[0][1]))+"; "
                sendmessage += "阴线的概率是"+str('%.2f%%' %(100* predict_high.detach().numpy()[0][0]))+" \n"

                if ((high_mark == 1) and ((timeNow.minute+send_time ) % 60 != 0)) or ((timeNow.minute+send_time ) % 60 == 0):
                    #print("破新低的概率是"+str(predict_low.detach().numpy()[0][1])+"  没有破新低的概率是"+str(predict_low.detach().numpy()[0][0]))
                    sendmessage = sendmessage.replace("--","-")
                    print(sendmessage)
                    response = requests.post('http://order.shareted.com/openapi/coin/signal/report',data=sendMessage(sendmessage))
                    # 通过get请求返回的文本值
                    print(response.text)
                    #print(title)
                    #sendMail(title,sendmessage)
            time.sleep(60)
        if timeNow.hour % 5 == 1 and timeNow.minute == 20:
            yyPath = reNew(COIN+r"YY/FC"+str(MulEncoding)+COIN+"yy"+str(MINUTE)+"min"+str(int(predict_length*MINUTE/60))+"hourpre"+str(ERROR_RATE)+"error"+str(input_length)+"input2K.pth",COIN+r"YY/FCPCAMA"+str(predict_length)+".m","YY")
            
            high = torch.load(yyPath[0])
            highPca = joblib.load(yyPath[1])
            
            
            time.sleep(60)
