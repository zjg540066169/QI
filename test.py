#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 22:51:22 2019

@author: jungangzou
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def stampToTime(timeStamp):#将时间戳转成字符串
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
    return otherStyleTime    
    
def timeToStamp(strTime):#将字符串转成时间戳
    timeArray = time.strptime(strTime, "%Y--%m--%d %H:%M:%S")
    timestamp = time.mktime(timeArray)
    return timestamp


def secondToStr(second):
    secondStr = str(second)
    for i in range(6-len(secondStr)):
        secondStr = '0'+secondStr
    secondStr = secondStr[0:2]+':'+secondStr[2:4]+':'+secondStr[4:6]
    print(secondStr)
    return secondStr
   
    
    

data = pd.read_csv('data1min.csv',index_col = 'time')
data_60 = data.copy().iloc[0:2,:]
for i in range(0,data.shape[0]-60+1,60):
    print(i,data.shape[0]-60+1)
    data_60.loc[data.index[i],'open'] =  data.loc[data.index[i],'open']
    data_60.loc[data.index[i],'close'] =  data.loc[data.index[i+14],'close']
    data_60.loc[data.index[i],'high'] =  data.loc[data.index[i:i+14],'high'].max()
    data_60.loc[data.index[i],'low'] =  data.loc[data.index[i:i+14],'low'].min()
    data_60.loc[data.index[i],'volume'] =  data.loc[data.index[i:i+14],'volume'].sum()
data_60 = data_60.drop(data_60.index[1],axis = 0)
data_60.to_csv("data_60min.csv")
# =============================================================================
# data = data.drop(['Unnamed: 0','<TICKER>'],axis = 1)
# for i in range(data.shape[0]):
#     a = str(data.loc[i,'<DTYYYYMMDD>'])[0:4] + '--' +str(data.loc[i,'<DTYYYYMMDD>'])[4:6]+'--' +str(data.loc[i,'<DTYYYYMMDD>'])[6:]
#     a = a + ' '+secondToStr(data.loc[i,'<TIME>'])
#     date = int(timeToStamp(a))
#     print(date)
#     data.loc[i,'time'] = date
# =============================================================================
