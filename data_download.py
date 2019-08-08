# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 22:55:49 2019

@author: wang
"""
from urllib import request
from urllib.error import HTTPError
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests
from sklearn.externals import joblib

class Data_download(object):
    def __init__(self,pastSecond,end,plateform,coin,to,timespan,aggregate,start=0):
        self.start = start
        self.past = pastSecond
        self.end = end
        self.plateform = plateform
        self.coin = coin
        self.timespan = timespan
        self.aggregate = aggregate
        self.to = to
    
    
    def downloadData(self):
        true = True
        false = False
        null = ''
        
        if type(self.start)== int and type(self.end)== int:
            startTime = self.start
            endTime = self.end
        else:
            startTime = int(time.mktime(time.strptime(self.start, '%Y-%m-%d %H:%M:%S')))
            endTime = int(time.mktime(time.strptime(self.end, '%Y-%m-%d %H:%M:%S')))
        #print(startTime)
        #print(endTime)
        if self.start == 0:
            params = "http://106.75.90.217:30804/api/coin/kline?con_past_second="+str(self.past)+"&con_date_end="+str(endTime)+"&con_plateform="+self.plateform+"&con_coin_from="+self.coin+"&con_coin_to="+self.to+"&con_timespan="+self.timespan+"&con_aggregat="+str(self.aggregate)+"&con_regular=20190402&con_simple=20190402&count"
        
        else: 
            params = "http://106.75.90.217:30804/api/coin/kline?con_date_start="+str(startTime)+"&con_date_end="+str(endTime)+"&con_plateform="+self.plateform+"&con_coin_from="+self.coin+"&con_coin_to="+self.to+"&con_timespan="+self.timespan+"&con_aggregat="+str(self.aggregate)+"&con_regular=20190402&con_simple=20190402&count"
        
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
                #return data
                return eval(data)
            except HTTPError:
                print("Httperror")
                #time.sleep(60)
                continue
            except Exception as e:
                print("error",e)
                #time.sleep(60)
                continue

    def predict_data_process(self,data,pcapath,MulEncoding = 13,ERROR_RATE = 0, ERROR_RATE_Vol = 0.2, input_length = 96,predict_length = 4,last_length = 16,long_input = False):
        self.input_length = input_length
        self.long_input = long_input
        self.pcapath = pcapath
        saved_columns = ['open','high','low','close','ohlc4','ohlc4_sma_5','ohlc4_sma_5_ema_2','hl2','close_ma_5','close_ma_13','close_ma_30','close_ma_75','close_ma_5_minus_close_ma_30','close_ma_30_minus_close_ma_75','close_ma_5_cross_close_ma_13','close_ma_5_cross_close_ma_30','close_ma_5_cross_close_ma_75','close_ma_13_cross_close_ma_30','close_ma_13_cross_close_ma_75','close_ma_30_cross_close_ma_75']
        
        #print(data.columns)
        
        self.tag = data.loc[:,'TAG']
        
        self.data = data.loc[:,saved_columns]
        
        #return self.data
        train = self.data.copy()
        train = train.dropna(axis = 0)
        
        for i in range(1,MulEncoding):
            train.loc[train.index[i]:train.index[-1],'OO'+str(i)] = (train.loc[train.index[i]:,"open"].values - train.loc[:train.index[-1-i],"open"].values)/train.loc[:train.index[-1-i],"open"].values
            train.loc[train.index[i]:train.index[-1],'OH'+str(i)] = (train.loc[train.index[i]:,"open"].values - train.loc[:train.index[-1-i],"high"].values)/train.loc[:train.index[-1-i],"high"].values
            train.loc[train.index[i]:train.index[-1],'OL'+str(i)] = (train.loc[train.index[i]:,"open"].values - train.loc[:train.index[-1-i],"low"].values)/train.loc[:train.index[-1-i],"low"].values
            train.loc[train.index[i]:train.index[-1],'OC'+str(i)] = (train.loc[train.index[i]:,"open"].values - train.loc[:train.index[-1-i],"close"].values)/train.loc[:train.index[-1-i],"close"].values 
            train.loc[train.index[i]:train.index[-1],'HO'+str(i)] = (train.loc[train.index[i]:,"high"].values - train.loc[:train.index[-1-i],"open"].values)/train.loc[:train.index[-1-i],"open"].values
            train.loc[train.index[i]:train.index[-1],'HH'+str(i)] = (train.loc[train.index[i]:,"high"].values - train.loc[:train.index[-1-i],"high"].values)/train.loc[:train.index[-1-i],"high"].values
            train.loc[train.index[i]:train.index[-1],'HL'+str(i)] = (train.loc[train.index[i]:,"high"].values - train.loc[:train.index[-1-i],"low"].values)/train.loc[:train.index[-1-i],"low"].values
            train.loc[train.index[i]:train.index[-1],'HC'+str(i)] = (train.loc[train.index[i]:,"high"].values - train.loc[:train.index[-1-i],"close"].values)/train.loc[:train.index[-1-i],"close"].values
            train.loc[train.index[i]:train.index[-1],'LO'+str(i)] = (train.loc[train.index[i]:,"low"].values - train.loc[:train.index[-1-i],"open"].values)/train.loc[:train.index[-1-i],"open"].values
            train.loc[train.index[i]:train.index[-1],'LH'+str(i)] = (train.loc[train.index[i]:,"low"].values - train.loc[:train.index[-1-i],"high"].values)/train.loc[:train.index[-1-i],"high"].values
            train.loc[train.index[i]:train.index[-1],'LL'+str(i)] = (train.loc[train.index[i]:,"low"].values - train.loc[:train.index[-1-i],"low"].values)/train.loc[:train.index[-1-i],"low"].values
            train.loc[train.index[i]:train.index[-1],'LC'+str(i)] = (train.loc[train.index[i]:,"low"].values - train.loc[:train.index[-1-i],"close"].values)/train.loc[:train.index[-1-i],"close"].values 
            train.loc[train.index[i]:train.index[-1],'CO'+str(i)] = (train.loc[train.index[i]:,"close"].values - train.loc[:train.index[-1-i],"open"].values)/train.loc[:train.index[-1-i],"open"].values
            train.loc[train.index[i]:train.index[-1],'CH'+str(i)] = (train.loc[train.index[i]:,"close"].values - train.loc[:train.index[-1-i],"high"].values)/train.loc[:train.index[-1-i],"high"].values
            train.loc[train.index[i]:train.index[-1],'CL'+str(i)] = (train.loc[train.index[i]:,"close"].values - train.loc[:train.index[-1-i],"low"].values)/train.loc[:train.index[-1-i],"low"].values
            train.loc[train.index[i]:train.index[-1],'CC'+str(i)] = (train.loc[train.index[i]:,"close"].values - train.loc[:train.index[-1-i],"close"].values)/train.loc[:train.index[-1-i],"close"].values
        
        MAn = [5,13,30,75]
        train = train.iloc[MAn[-1]:,:]
        #train.loc[:,"close_ema_5_minus_close_ema_30"] = (train.loc[:,"close_ema_5"] - train.loc[:,"close_ema_30"]) / train.loc[:,"close_ema_30"]
        #train.loc[:,"close_ema_30_minus_close_ema_75"] = (train.loc[:,"close_ema_30"] - train.loc[:,"close_ema_75"]) / train.loc[:,"close_ema_75"]
       
        train_new_feature = pd.DataFrame()#pd.DataFrame(np.zeros(len(train),5*len(MAn)))        
        N =20
        weights = np.ones(N)/N
        for i in range(len(MAn)):
            
# =============================================================================
#             train.loc[:,"OMA"+str(MAn[i])] = np.abs(train.loc[:,"open"] - train.loc[:,"close"+"_ma_"+str(MAn[i])])# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
#             train.loc[:,"CMA"+str(MAn[i])] = np.abs(train.loc[:,"close"] - train.loc[:,"close"+"_ma_"+str(MAn[i])])# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
#             train.loc[:,"HMA"+str(MAn[i])] = np.abs(train.loc[:,"high"] - train.loc[:,"close"+"_ma_"+str(MAn[i])])# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
#             train.loc[:,"LMA"+str(MAn[i])] = np.abs(train.loc[:,"low"] - train.loc[:,"close"+"_ma_"+str(MAn[i])])# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
#             train.loc[:,"HL2MA"+str(MAn[i])] = np.abs(train.loc[:,"hl2"] - train.loc[:,"close"+"_ma_"+str(MAn[i])])# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
#             train.loc[:,"OSMA"+str(N)+str(MAn[i])] =  np.convolve(weights,train.loc[:,"OMA"+str(MAn[i])].values)[N-1:-N+1]
#             train.loc[:,"CSMA"+str(N)+str(MAn[i])] =  np.convolve(weights,train.loc[:,"CMA"+str(MAn[i])].values)[N-1:-N+1]
#             train.loc[:,"HSMA"+str(N)+str(MAn[i])] =  np.convolve(weights,train.loc[:,"HMA"+str(MAn[i])].values)[N-1:-N+1]
#             train.loc[:,"LSMA"+str(N)+str(MAn[i])] =  np.convolve(weights,train.loc[:,"LMA"+str(MAn[i])].values)[N-1:-N+1]
#             train.loc[:,"HL2SMA"+str(N)+str(MAn[i])] =  np.convolve(weights,train.loc[:,"HL2MA"+str(MAn[i])].values)[N-1:-N+1]
# 
# =============================================================================
            OMA = (train.loc[:,"open"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            CMA = (train.loc[:,"close"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            HMA = (train.loc[:,"high"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            LMA = (train.loc[:,"low"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            HL2MA = (train.loc[:,"hl2"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            #print('osma',np.abs(OMA))
            OSMA =  np.convolve(weights,np.abs(OMA),'same')[N-1:]           
            CSMA =  np.convolve(weights,np.abs(CMA),'same')[N-1:]
            HSMA =  np.convolve(weights,np.abs(HMA),'same')[N-1:]
            LSMA =  np.convolve(weights,np.abs(LMA),'same')[N-1:]
            HL2SMA =  np.convolve(weights,np.abs(HL2MA),'same')[N-1:]
            #print(OMA.shape,HL2SMA.shape)
            O = ((OMA[N-1:] - HL2SMA) * 10 / OSMA.std() ).astype(np.int)
            H = ((HMA[N-1:] - HL2SMA) * 10 / HSMA.std() ).astype(np.int)
            L = ((LMA[N-1:] - HL2SMA) * 10 / LSMA.std() ).astype(np.int)
            C = ((CMA[N-1:] - HL2SMA) * 10 / CSMA.std() ).astype(np.int)
            HL2 = ((HL2MA[N-1:] - HL2SMA) / HL2SMA.std() ).astype(np.int)
            train_new_feature.loc[:,'O'+str(MAn[i])] = O
            train_new_feature.loc[:,'C'+str(MAn[i])] = C
            train_new_feature.loc[:,'H'+str(MAn[i])] = H
            train_new_feature.loc[:,'L'+str(MAn[i])] = L
            train_new_feature.loc[:,'HL2'+str(MAn[i])] = HL2
        
        train_new_feature.index = train.index[N-1:]
        train_new_feature[train_new_feature>31] = 31
        
# =============================================================================
#         for i in ['022_HH','022_LL','022_med','022_fb1','022_fb2','022_fb3','022_fb4']:
#             train.loc[:,"O"+i] = (train.loc[:,"open"] - train.loc[:,i]) / train.loc[:,i]
#             train.loc[:,"C"+i] = (train.loc[:,"close"] - train.loc[:,i]) / train.loc[:,i]
#             train.loc[:,"H"+i] = (train.loc[:,"high"] - train.loc[:,i]) / train.loc[:,i]
#             train.loc[:,"L"+i] = (train.loc[:,"low"] - train.loc[:,i]) / train.loc[:,i]
#             train.loc[:,"HL2"+i] = (train.loc[:,"hl2"] - train.loc[:,i]) / train.loc[:,i]
# 
# =============================================================================
        
        train_data = train.copy()
        deleted_columns = ['high','low','open','close','hl2','close_ma_5','close_ma_13',"close_ma_30",'close_ma_75']#['022_HH','022_LL','022_med','022_fb1','022_fb2','022_fb3','022_fb4','high','low','open','close','hl2','volume','volume_sma_5','close_ema_5','close_ema_13',"close_ema_30",'close_ema_75']
        
        train_data.drop(deleted_columns,axis=1,inplace=True)
        train_data = train_data.dropna(axis = 0)
        train_data = train_data.astype(np.float64)
        train_data[train_data>ERROR_RATE] = 1
        train_data[train_data<-ERROR_RATE] = -1
        train_data[np.abs(train_data)<ERROR_RATE] = 0

        

# =============================================================================
#         VolMA = train.loc[:,'volume'] / train.loc[:,'volume_sma_5']
#         VolMA[np.abs(VolMA)-1 <ERROR_RATE_Vol] = 0    
#         VolMA[VolMA >= 1+ERROR_RATE_Vol] = 1
#         VolMA[VolMA <= 1-ERROR_RATE_Vol] = -1
#         train_data.loc[:,"VolMA"] = VolMA
# =============================================================================
        train_data = pd.concat([train_data, train_new_feature], axis=1, join_axes=[train_data.index])

        self.train = train_data.iloc[MulEncoding-1:,:]
        self.train = train_data.dropna(axis = 0)
        #return self.train
        #print(self.train.shape)
        
        return  self.__process()
        
    def __process(self):
        input_length = self.input_length
        train = self.train
        
        
        train_x = np.array([train.iloc[((-input_length*2-1)):-input_length-1,:].values,train.iloc[-input_length-1:-1,:].values])
        #return train_x
        #train_y = pd.DataFrame(np.zeros((train.shape[0] - input_length - max(predict_length_EMA_all,predict_length_EMA_threhold)*3 + 1,4)))
        #train_y[train_y == 0 ] = np.nan
        #train_y.columns = ['MAClassify','MAClassifyThrehold','DCClassifyHigh','DCClassifyLow']        
        
        if self.long_input:
            train_x = train_x.reshape(train_x.shape[0],-1)
        elif self.long_input == False:
            train_x = train_x.reshape(train_x.shape[0]*train_x.shape[1],train_x.shape[2])

        pca = joblib.load(self.pcapath)
        print(train_x.shape)
        train_x = pca.transform(train_x)
        
        if self.long_input == False:
            train_x = train_x.reshape(-1,input_length,train_x.shape[1])
        #print("train_x",train_x.shape)
        self.X = train_x
        return self.tag.iloc[-1]
    
    def get_predictX(self):
        return self.X
# =============================================================================
#     rows = data["rows"]
#     dataFrame = pd.DataFrame(np.zeros((total,5)),columns=["Open","High","Low","Close","Volume"])
#     timeStamp = []
#     for i in range(len(rows)):
#         dataFrame.iloc[i,0] = rows[i]["open"]
#         dataFrame.iloc[i,1] = rows[i]["high"]
#         dataFrame.iloc[i,2] = rows[i]["low"]
#         dataFrame.iloc[i,3] = rows[i]["close"]
#         dataFrame.iloc[i,4] = rows[i]["count"]
#         timeStamp.append(rows[i]["time"])
#     dataFrame.index = timeStamp
#     print(dataFrame.index)
#     return dataFrame
# 
# =============================================================================
if __name__=='__main__':
    a = Data_download(60000,1549983930,'HUOBI','XRP','USDT','1M',15)
    b = a.downloadData()
