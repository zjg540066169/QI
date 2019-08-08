#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:37:39 2019

This class is a specification of Feature_Processing_base, which is used to process
raw data to features.
This version applies 2 method:
The first is to compare 4 type of prices with each other as totally 16 features among multiple K-lines.
The second is to calculate the stardard deviation of the difference between price and MA-lines.
Since the predicting data and training data are different, so we use different function to process. 

@author: jungangzou
"""

from Unified_Model.Based_Class.Feature_Processing_base import Feature_Processing_base
import pandas as pd
import numpy as np


class Feature_Processing_MulKCompare(Feature_Processing_base):
    def __init__(self,parameters):
    #Initialize
        super().__init__(parameters)
        self.parameters = parameters
        


    def feature_processing(self,data,mode):
    #This method is defined to process data with different mode.
        if mode == 'training':
            self.__feature_processing_training(data)
            
        elif mode == 'predicting':
            self.__feature_processing_predicting(data)



    def get_features(self):
    #This abstract method is to return the processed features.
        return self.data
    
    
    def __feature_processing_training(self,data):
        #process the training data.
        train = data.copy()
        train = train.dropna(axis = 0)
        
        #save the columns we need and drop others.
        saved_columns = ['open','high','low','close','ohlc4','ohlc4_sma_5','ohlc4_sma_5_ema_2','hl2','close_ema_5','close_ema_13','close_ema_30','close_ema_75','close_ema_5_minus_close_ema_30','close_ema_30_minus_close_ema_75','close_ema_5_cross_close_ema_13','close_ema_5_cross_close_ema_30','close_ema_5_cross_close_ema_75','close_ema_13_cross_close_ema_30','close_ema_13_cross_close_ema_75','close_ema_30_cross_close_ema_75']
        train = train.loc[:,saved_columns]
        train.loc[:,"close_ema_5_minus_close_ema_30"] = (train.loc[:,"close_ema_5"] - train.loc[:,"close_ema_30"]) / train.loc[:,"close_ema_30"]
        train.loc[:,"close_ema_30_minus_close_ema_75"] = (train.loc[:,"close_ema_30"] - train.loc[:,"close_ema_75"]) / train.loc[:,"close_ema_75"]

        
        
        #compare 4 type of prices with each other as totally 16 features among multiple K-lines
        if 'mul_K_compare' in self.parameters:
            mul_K_compare = self.parameters['mul_K_compare']
        else:
            mul_K_compare = 13
        
        for i in range(1,mul_K_compare):
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
        
        
        
        
        
        #calculate the stardard deviation of the difference between price and MA-lines.
        MAn = [5,13,30,75]
        train = train.iloc[max(MAn):,:]
        
        N =20
        
        weights = np.ones(N)/N
        
        train_new_feature = pd.DataFrame()
        for i in range(len(MAn)):         
            OMA = (train.loc[:,"open"] - train.loc[:,"close"+"_ema_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            CMA = (train.loc[:,"close"] - train.loc[:,"close"+"_ema_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            HMA = (train.loc[:,"high"] - train.loc[:,"close"+"_ema_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            LMA = (train.loc[:,"low"] - train.loc[:,"close"+"_ema_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            HL2MA = (train.loc[:,"hl2"] - train.loc[:,"close"+"_ema_"+str(MAn[i])]).values# / train.loc[:,"close"+"_ma_"+str(MAn[i])]
            #print(OMA)
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
        train_new_feature[train_new_feature>31] = 31# if the std is larger than 31, set it to 31.
        
        
        #concat all features.
        train_data = train.copy()
        deleted_columns = ['high','low','open','close','hl2','close_ema_5','close_ema_13',"close_ema_30",'close_ema_75']#['022_HH','022_LL','022_med','022_fb1','022_fb2','022_fb3','022_fb4','high','low','open','close','hl2','volume','volume_sma_5','close_ema_5','close_ema_13',"close_ema_30",'close_ema_75']
        train_data.drop(deleted_columns,axis=1,inplace=True)
        train_data = train_data.dropna(axis = 0)
        train_data = train_data.astype(np.float64)
        train_data = pd.concat([train_data, train_new_feature], axis=1, join_axes=[train_data.index])
        self.data = train_data.iloc[mul_K_compare-1:,:]
        self.data = self.data.dropna(axis = 0)
        
    def __feature_processing_predicting(self,data):
        #process the predicting data.Since there is only few differences between training and predicting data,
        #the logic are same.
        predict = data.copy()
        predict = predict.dropna(axis = 0)
        
        saved_columns = ['open','high','low','close','ohlc4','ohlc4_sma_5','ohlc4_sma_5_ema_2','hl2','close_ma_5','close_ma_13','close_ma_30','close_ma_75','close_ma_5_minus_close_ma_30','close_ma_30_minus_close_ma_75','close_ma_5_cross_close_ma_13','close_ma_5_cross_close_ma_30','close_ma_5_cross_close_ma_75','close_ma_13_cross_close_ma_30','close_ma_13_cross_close_ma_75','close_ma_30_cross_close_ma_75']

        predict = predict.loc[:,saved_columns]
        #
        if 'mul_K_compare' in self.parameters:
            mul_K_compare = self.parameters['mul_K_compare']
        else:
            mul_K_compare = 13
        
        for i in range(1,mul_K_compare):
            predict.loc[predict.index[i]:predict.index[-1],'OO'+str(i)] = (predict.loc[predict.index[i]:,"open"].values - predict.loc[:predict.index[-1-i],"open"].values)/predict.loc[:predict.index[-1-i],"open"].values
            predict.loc[predict.index[i]:predict.index[-1],'OH'+str(i)] = (predict.loc[predict.index[i]:,"open"].values - predict.loc[:predict.index[-1-i],"high"].values)/predict.loc[:predict.index[-1-i],"high"].values
            predict.loc[predict.index[i]:predict.index[-1],'OL'+str(i)] = (predict.loc[predict.index[i]:,"open"].values - predict.loc[:predict.index[-1-i],"low"].values)/predict.loc[:predict.index[-1-i],"low"].values
            predict.loc[predict.index[i]:predict.index[-1],'OC'+str(i)] = (predict.loc[predict.index[i]:,"open"].values - predict.loc[:predict.index[-1-i],"close"].values)/predict.loc[:predict.index[-1-i],"close"].values 
            predict.loc[predict.index[i]:predict.index[-1],'HO'+str(i)] = (predict.loc[predict.index[i]:,"high"].values - predict.loc[:predict.index[-1-i],"open"].values)/predict.loc[:predict.index[-1-i],"open"].values
            predict.loc[predict.index[i]:predict.index[-1],'HH'+str(i)] = (predict.loc[predict.index[i]:,"high"].values - predict.loc[:predict.index[-1-i],"high"].values)/predict.loc[:predict.index[-1-i],"high"].values
            predict.loc[predict.index[i]:predict.index[-1],'HL'+str(i)] = (predict.loc[predict.index[i]:,"high"].values - predict.loc[:predict.index[-1-i],"low"].values)/predict.loc[:predict.index[-1-i],"low"].values
            predict.loc[predict.index[i]:predict.index[-1],'HC'+str(i)] = (predict.loc[predict.index[i]:,"high"].values - predict.loc[:predict.index[-1-i],"close"].values)/predict.loc[:predict.index[-1-i],"close"].values
            predict.loc[predict.index[i]:predict.index[-1],'LO'+str(i)] = (predict.loc[predict.index[i]:,"low"].values - predict.loc[:predict.index[-1-i],"open"].values)/predict.loc[:predict.index[-1-i],"open"].values
            predict.loc[predict.index[i]:predict.index[-1],'LH'+str(i)] = (predict.loc[predict.index[i]:,"low"].values - predict.loc[:predict.index[-1-i],"high"].values)/predict.loc[:predict.index[-1-i],"high"].values
            predict.loc[predict.index[i]:predict.index[-1],'LL'+str(i)] = (predict.loc[predict.index[i]:,"low"].values - predict.loc[:predict.index[-1-i],"low"].values)/predict.loc[:predict.index[-1-i],"low"].values
            predict.loc[predict.index[i]:predict.index[-1],'LC'+str(i)] = (predict.loc[predict.index[i]:,"low"].values - predict.loc[:predict.index[-1-i],"close"].values)/predict.loc[:predict.index[-1-i],"close"].values 
            predict.loc[predict.index[i]:predict.index[-1],'CO'+str(i)] = (predict.loc[predict.index[i]:,"close"].values - predict.loc[:predict.index[-1-i],"open"].values)/predict.loc[:predict.index[-1-i],"open"].values
            predict.loc[predict.index[i]:predict.index[-1],'CH'+str(i)] = (predict.loc[predict.index[i]:,"close"].values - predict.loc[:predict.index[-1-i],"high"].values)/predict.loc[:predict.index[-1-i],"high"].values
            predict.loc[predict.index[i]:predict.index[-1],'CL'+str(i)] = (predict.loc[predict.index[i]:,"close"].values - predict.loc[:predict.index[-1-i],"low"].values)/predict.loc[:predict.index[-1-i],"low"].values
            predict.loc[predict.index[i]:predict.index[-1],'CC'+str(i)] = (predict.loc[predict.index[i]:,"close"].values - predict.loc[:predict.index[-1-i],"close"].values)/predict.loc[:predict.index[-1-i],"close"].values
        #print('open',predict.loc[:,"open"])
        MAn = [5,13,30,75]
        predict = predict.iloc[max(MAn):,:]
        predict.loc[:,"close_ma_5_minus_close_ma_30"] = (predict.loc[:,"close_ma_5"] - predict.loc[:,"close_ma_30"]) / predict.loc[:,"close_ma_30"]
        predict.loc[:,"close_ma_30_minus_close_ma_75"] = (predict.loc[:,"close_ma_30"] - predict.loc[:,"close_ma_75"]) / predict.loc[:,"close_ma_75"]
        
        #print(predict.loc[:,"open"])
        predict_new_feature = pd.DataFrame()#pd.DataFrame(np.zeros(len(predict),5*len(MAn)))        
        N =20
        
        weights = np.ones(N)/N

        for i in range(len(MAn)):         
            OMA = (predict.loc[:,"open"] - predict.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / predict.loc[:,"close"+"_ma_"+str(MAn[i])]
            #print(predict.loc[:,"open"])
            #print(predict.loc[:,"close"+"_ma_"+str(MAn[i])])
            CMA = (predict.loc[:,"close"] - predict.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / predict.loc[:,"close"+"_ma_"+str(MAn[i])]
            HMA = (predict.loc[:,"high"] - predict.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / predict.loc[:,"close"+"_ma_"+str(MAn[i])]
            LMA = (predict.loc[:,"low"] - predict.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / predict.loc[:,"close"+"_ma_"+str(MAn[i])]
            HL2MA = (predict.loc[:,"hl2"] - predict.loc[:,"close"+"_ma_"+str(MAn[i])]).values# / predict.loc[:,"close"+"_ma_"+str(MAn[i])]
            #print(OMA)
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
            predict_new_feature.loc[:,'O'+str(MAn[i])] = O
            predict_new_feature.loc[:,'C'+str(MAn[i])] = C
            predict_new_feature.loc[:,'H'+str(MAn[i])] = H
            predict_new_feature.loc[:,'L'+str(MAn[i])] = L
            predict_new_feature.loc[:,'HL2'+str(MAn[i])] = HL2
        
        predict_new_feature.index = predict.index[N-1:]
        predict_new_feature[predict_new_feature>31] = 31
        predict_data = predict.copy()
        deleted_columns = ['high','low','open','close','hl2','close_ma_5','close_ma_13',"close_ma_30",'close_ma_75']#['022_HH','022_LL','022_med','022_fb1','022_fb2','022_fb3','022_fb4','high','low','open','close','hl2','volume','volume_sma_5','close_ema_5','close_ema_13',"close_ema_30",'close_ema_75']
        predict_data.drop(deleted_columns,axis=1,inplace=True)
        predict_data = predict_data.dropna(axis = 0)
        predict_data = predict_data.astype(np.float64)
        predict_data = pd.concat([predict_data, predict_new_feature], axis=1, join_axes=[predict_data.index])
        self.data = predict_data.iloc[mul_K_compare-1:,:]
        self.data = self.data.dropna(axis = 0)

