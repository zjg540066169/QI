# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:33:30 2019

@author: wang
"""

import read_data as rd
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
class data_process(object):
    def __init__(self,path,pcapath,MulEncoding = 13,ERROR_RATE = 0, ERROR_RATE_Vol = 0.2, input_length = 96,predict_length = 4,last_length = 16, long_input = False):
        self.data = pd.DataFrame(rd.Read_data(path).decode())
        #self.data.index = self.data.loc[:,'time']
        self.data.drop(['time'],axis=1,inplace=True)
        #print(self.data.shape)
        self.data = self.data#.iloc[int(len(self.data)/20)*1:int(len(self.data)/20)*2,:]
        self.MulEncoding = MulEncoding
        self.ERROR_RATE = ERROR_RATE
        self.ERROR_RATE_Vol = ERROR_RATE_Vol
        self.predict_length = predict_length
        self.input_length = input_length
        self.predict_length = predict_length
        self.last_length = last_length
        self.pcapath = pcapath
        self.long_input = long_input
        if self.long_input:
            self.PCA_dimension = 50
        else:
            self.PCA_dimension = 50
        
    def process(self):
        #return self.data
        MulEncoding = self.MulEncoding
        ERROR_RATE = self.ERROR_RATE
        ERROR_RATE_Vol = self.ERROR_RATE_Vol

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
        train.loc[:,"close_ma_5_minus_close_ma_30"] = (train.loc[:,"close_ma_5"] - train.loc[:,"close_ma_30"]) / train.loc[:,"close_ma_30"]
        train.loc[:,"close_ma_30_minus_close_ma_75"] = (train.loc[:,"close_ma_30"] - train.loc[:,"close_ma_75"]) / train.loc[:,"close_ma_75"]
       
        
        for i in range(len(MAn)):
            train.loc[:,"OMA"+str(MAn[i])] = (train.loc[:,"open"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]) / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            train.loc[:,"CMA"+str(MAn[i])] = (train.loc[:,"close"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]) / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            train.loc[:,"HMA"+str(MAn[i])] = (train.loc[:,"high"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]) / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            train.loc[:,"LMA"+str(MAn[i])] = (train.loc[:,"low"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]) / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            train.loc[:,"HL2MA"+str(MAn[i])] = (train.loc[:,"hl2"] - train.loc[:,"close"+"_ma_"+str(MAn[i])]) / train.loc[:,"close"+"_ma_"+str(MAn[i])]
        
        for i in ['022_HH','022_LL','022_med','022_fb1','022_fb2','022_fb3','022_fb4']:
            train.loc[:,"O"+i] = (train.loc[:,"open"] - train.loc[:,i]) / train.loc[:,i]
            train.loc[:,"C"+i] = (train.loc[:,"close"] - train.loc[:,i]) / train.loc[:,i]
            train.loc[:,"H"+i] = (train.loc[:,"high"] - train.loc[:,i]) / train.loc[:,i]
            train.loc[:,"L"+i] = (train.loc[:,"low"] - train.loc[:,i]) / train.loc[:,i]
            train.loc[:,"HL2"+i] = (train.loc[:,"hl2"] - train.loc[:,i]) / train.loc[:,i]

        
        train_data = train.copy()
        deleted_columns = ['022_HH','022_LL','022_med','022_fb1','022_fb2','022_fb3','022_fb4','high','low','open','close','hl2','volume','volume_sma_5','close_ma_5','close_ma_13',"close_ma_30",'close_ma_75']
        train_data.drop(deleted_columns,axis=1,inplace=True)
        train_data = train_data.dropna(axis = 0)
        train_data = train_data.astype(np.float64)
        train_data[train_data>ERROR_RATE] = 1
        train_data[train_data<-ERROR_RATE] = -1
        train_data[np.abs(train_data)<ERROR_RATE] = 0

        

        VolMA = train.loc[:,'volume'] / train.loc[:,'volume_sma_5']
        VolMA[np.abs(VolMA)-1 <ERROR_RATE_Vol] = 0    
        VolMA[VolMA >= 1+ERROR_RATE_Vol] = 1
        VolMA[VolMA <= 1-ERROR_RATE_Vol] = -1
        train_data.loc[:,"VolMA"] = VolMA
        self.train = train_data.iloc[MulEncoding-1:,:]
        self.train = train_data.dropna(axis = 0)

        a = self.__process()
        return a
    
    def __process(self):
        input_length = self.input_length
        predict_length = self.predict_length
        last_length = self.last_length
        predict_length_EMA_threhold = 24
        predict_length_EMA_all = 8

        train = self.train
        up_index = train[train.loc[:,'021'] == 1].index
        down_index = train[train.loc[:,'021'] == -1].index
        index = set(up_index.append(down_index))
        train_x = []
        train_y = []
        for i in range(train.shape[0] - input_length - max(predict_length_EMA_all,predict_length_EMA_threhold)*3 + 1):
            if train.index[i+input_length] in index: 
                for j in range(-2,3):
                    try:
                        if i+j <0:
                            continue
                        print(i+j,train.shape[0] - input_length - max(predict_length_EMA_all,predict_length_EMA_threhold)*3 + 1)
                        train_y.append([self.__MAClassify(self.data.loc[self.train.index[i+input_length:i+input_length+predict_length_EMA_all],"021_TR"]),self.__MAClassifyThrehold(self.data.loc[train.index[i+input_length:i+input_length+predict_length_EMA_threhold],"021_TR"]),self.__DCClassifyHigh(self.data.index[i+input_length:i+input_length+predict_length],self.train.index[i+input_length-last_length:i+input_length]),self.__DCClassifyLow(self.data.index[i+input_length:i+input_length+predict_length],self.train.index[i+input_length-last_length:i+input_length])])
                        train_x.append(train.iloc[i+j:i+input_length+j,:].values)
                    except IndexError as e:
                        print(e)
                        train_y = np.array(train_y)
                        train_x = np.array(train_x)
                        print(train_x.shape,train_y.shape)

                        exit()
                        break

        train_y = np.array(train_y)
        train_x = np.array(train_x)
        print(train_x)
        print(train_x.shape,train_y.shape)
        print(self.long_input)
        if self.long_input == False:
            train_x = train_x.reshape(train_x.shape[0]*train_x.shape[1],train_x.shape[2])
        else:
            train_x = train_x.reshape(train_x.shape[0],-1)
        try:
            pca = joblib.load(self.pcapath)
            train_x = pca.transform(train_x)
        except FileNotFoundError:
            pca =PCA(n_components=self.PCA_dimension)
            train_x = pca.fit_transform(train_x)
            joblib.dump(pca, self.pcapath)
        if self.long_input == False:
            train_x = train_x.reshape(-1,input_length,train_x.shape[1])
        
        self.X = train_x
        self.Y = train_y
        
    def EMA(self,data,n):
        a = 2/(n+1)
        data = data.copy().dropna(axis = 0)
        #print(data.iloc[1])
        for i in range(1,len(data)):
            data.iloc[i] = a*data.iloc[i] + (1-a)*data.iloc[i-1]
        return data

    
    def __DCClassifyHigh(self,index0,index1):
        eRROR_RATE = self.ERROR_RATE
        x0 = self.data.loc[index0,'high'].copy()
        x1 = self.data.loc[index1,'022_HH'].copy()
        x2 = self.data.loc[index1,'022_fb1'].copy()
        x0_max = x0.max()
        x1_max = x1.max()
        x2_max = x2.max()
        if (x1_max - x0_max)/x0_max > eRROR_RATE:
            return 1
        elif (x2_max - x0_max)/x0_max > eRROR_RATE:
            return 2
        return 0

    def __DCClassifyLow(self,index0,index1):
        eRROR_RATE = self.ERROR_RATE
        x0 = self.data.loc[index0,'low'].copy()
        x1 = self.data.loc[index1,'022_LL'].copy()
        x2 = self.data.loc[index1,'022_fb4'].copy()
        x0_min = x0.min()
        x1_min = x1.min()
        x2_min = x2.min()
        if (x0_min - x1_min)/x0_min > eRROR_RATE:
            return 1
        elif (x0_min - x2_min)/x0_min > eRROR_RATE:
            return 2
        return 0
        
    
    def __MAClassify(self,x0):
        eRROR_RATE = self.ERROR_RATE
        length = len(x0)
        print(length,end = " ")
        C = 0
        print((np.sum(1 == x0[:length]) + np.sum(0 == x0[:length]) ))
        if (np.sum(1 == x0[:length]) + np.sum(0 == x0[:length]) )/ (length-1) >= 1:
            C = 0
        elif (np.sum(-1 == x0[:length])+ np.sum(0 == x0[:length])) / (length-1) >= 1:
            C = 1
        else:
            C = 2
        return C
    
    def __MAClassifyThrehold(self,x0,predict_threhold = 0.8):
        eRROR_RATE = self.ERROR_RATE
        length = len(x0)
        C = 0
        if (np.sum(1 == x0[:length])+ np.sum(0 == x0[:length])) / (length-1) >= predict_threhold:
            C = 0
        elif (np.sum(-1 == x0[:length])++ np.sum(0 == x0[:length])) / (length-1) >= predict_threhold:
            C = 1
        else:
            C = 2
        return C

    
    def get_Y(self):
        return self.Y
    
    def get_X(self):
        return self.X
    
if __name__ == '__main__':
    data_proces = data_process(r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_15m_singal_regular_20190226.json',r'BTC/15PCA.m')
    a= data_proces.process()
    #X,Y = data_proces.get_X(),data_proces.get_Y()
        