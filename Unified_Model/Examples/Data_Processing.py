#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:18:11 2019
This method is one of the most important class in this system which processed the data.
In processing training data step, the class need to read data from local file system, 
process data to needed features, return training and validation features. 
In processing prediting data step, the class need to dowload data from remote server, 
process data to needed features, return predicting features. 

@author: jungangzou
"""

from Unified_Model.Based_Class.Data_Processing_base import Data_Processing_base
from Unified_Model.Examples.Feature_Processing.Feature_Processing_MulKCompare import Feature_Processing_MulKCompare
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import time
from urllib import request
from urllib.error import HTTPError

class Data_Processing(Data_Processing_base):
    def __init__(self,parameters):
    #Initialize 
        self.parameters = parameters
        try:# set the class of Feature Processing 
            if self.parameters['Feature_Processing'] == 'Feature_Processing_MulKCompare':
                self.FP_model = Feature_Processing_MulKCompare(self.parameters)
        except KeyError:
            self.FP_model = Feature_Processing_MulKCompare(self.parameters)
        super().__init__(parameters,self.FP_model)
        
        
        
        
        
    #training part
    def training_data_loading(self):
    #this method is used to load training data from local file system and run processing function.
        
        training_data_path = self.parameters['training_data_path']
        if type(training_data_path) is str:    
            training_data_path_list = [training_data_path]
        else:
            training_data_path_list = training_data_path
            
        data = []
        for i in training_data_path_list:#concat many files if the number of file is larger than 1
            with open(i,'r') as f:
                text = f.readlines()[0]
                dic = json.loads(text)
                data.extend(dic)
        self.training_data = pd.DataFrame(data)
        self.training_data_processing()#run processing function

    
    def training_data_processing(self):
    #this method is used to process data.
        
    #define some important parameters
        input_length = self.parameters['input_length']
        pca_path = self.parameters['pca_path']
        try:
            long_input = self.parameters['long_input']
        except KeyError:
            if self.parameters['model'] == 'FC_Ngram': 
                long_input = True
            else:
                long_input = False
                
        try:
            PCA_dimension = self.parameters['pca_dimension']
        except KeyError:
            PCA_dimension = 50
    
            
        self.train_y_data = self.training_data.loc[:,'tag'].copy()#read labels
        
        #since Pytorch model need positive number as label tag, so set the negative label as 2.
        self.train_y_data[self.train_y_data == -1] = 2#
        
        #use FP_model to process data to features and get features.
        self.FP_model.feature_processing(self.training_data.iloc[:,:],mode = 'training')
        self.train_x_data = self.FP_model.get_features()
        
        
        
        #use sampling method to sample.
        index = self.train_x_data.index
        train_x = []
        train_y = []
        for i in range(self.train_x_data.shape[0] - input_length -5):#since we need to oversample for totoally 5 unit, so we need to substract 5 unit to avoid index error
            if self.train_x_data.index[i+input_length] in index: 
                for j in range(-2,3):#oversample
                    try:
                        if i+j <0:
                            continue
                        train_x.append(self.train_x_data.iloc[i+j:i+input_length+j,:].values)

                        train_y.append(self.train_y_data.iloc[i+input_length+j].tolist())

                    except IndexError as e:
                        print(e)
                        break
        
        train_y = np.array(train_y)
        train_x = np.array(train_x)
        
        
        
        #use PCA to reduce the dimension of features.
        if long_input == False:
            train_x = train_x.reshape(train_x.shape[0]*train_x.shape[1],train_x.shape[2])
        else:
            train_x = train_x.reshape(train_x.shape[0],-1)
        try:
            pca = joblib.load(pca_path)
            train_x = pca.transform(train_x)
        except FileNotFoundError:
            pca = PCA(n_components= PCA_dimension)
            train_x = pca.fit_transform(train_x)
            joblib.dump(pca, pca_path)
        if long_input == False:#reshape the 2-dim data to 3-dim data if not long_input
            train_x = train_x.reshape(-1,input_length,train_x.shape[1])
        
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(train_x, train_y, test_size=0.33)


    def get_training_X(self):
    #return training features
        return self.train_x
    
    
    def get_training_Y(self):
    #return training labels
        return self.train_y
    
    
    def get_val_X(self):
    #return validation features
        return self.val_x
    
    
    def get_val_Y(self):
    #return validation labels
        return self.val_y
    
    
    
    
    
    
    
    
    #predicting part
    def __download_data(self,pastSecond,end,plateform,coin,to,timespan,aggregate,con_regular,con_simple,start=0):
    #this method is to download data from remote server and return the data as a dictionary.
    #the parameters 'pastSecond' and 'start' are not compatible.
    
        #these three definitions are necessary for 'eval' function to parse the downloaded data which is a string. 
        #if these definitions are not defined, some errors will occur.
        true = True
        false = False
        null = ''
        
        if type(start)== int and type(end)== int:
            startTime = start
            endTime = end
        else:#if the parameters 'start' and 'end' are not timestamp, convert them to timestamp
            startTime = int(time.mktime(time.strptime(start, '%Y-%m-%d %H:%M:%S')))
            endTime = int(time.mktime(time.strptime(end, '%Y-%m-%d %H:%M:%S')))


        #the parameters 'pastSecond' and 'start' are not compatible.
        if start == 0:
            params = "http://106.75.90.217:30804/api/coin/kline?con_past_second="+str(pastSecond)+"&con_date_end="+str(endTime)+"&con_plateform="+plateform+"&con_coin_from="+coin+"&con_coin_to="+to+"&con_timespan="+timespan+"&con_aggregat="+str(aggregate)+"&con_regular="+str(con_regular)+"&con_simple="+str(con_simple)+"&count"
        
        else:
            params = "http://106.75.90.217:30804/api/coin/kline?con_date_start="+str(startTime)+"&con_date_end="+str(endTime)+"&con_plateform="+plateform+"&con_coin_from="+coin+"&con_coin_to="+to+"&con_timespan="+timespan+"&con_aggregat="+str(aggregate)+"&con_regular="+str(con_regular)+"&con_simple="+str(con_simple)+"&count"
        
        print(params)
        
        #use exception handling mechanism to request data until the data is parsed successfully.
        while True:
            try:
                print("start")
                req = request.Request(params)  # GET method
                print('request')
                reponse = request.urlopen(req) #download data
                print('read')
                data = reponse.read() # parse data as string type
                #print(data)
                reponse.close()
                print("ok")
                #return data
                return eval(data) #convert string to a dictionary
            except HTTPError:
                print("Httperror")
                time.sleep(60)
                continue
            except Exception as e:
                print("error",e)
                time.sleep(60)
                continue
    
    
    def predicting_data_loading(self,pastSecond,end,start=0):
    #this method is used to load predicting data from downloaded data and run processing function.
        plateform = self.parameters['plateform']
        coin = self.parameters['coin']
        to = self.parameters['to']
        timespan = self.parameters['timespan']
        aggregate = self.parameters['aggregate']
        con_regular = self.parameters['con_regular']
        con_simple = self.parameters['con_simple']
        
        #read the downloaded data.
        self.predict_data = pd.DataFrame(self.__download_data(pastSecond,end,plateform,coin,to,timespan,aggregate,con_regular,con_simple,start)['rows'])
        
        #if the last TAG is not 1, we don`t need to run predicting function. 
        #TAG_TYPE is a parameter from downloaded data which illustrates some important information of predicting data.
        self.predict_tag = self.predict_data.loc[:,'TAG'].iloc[-1]
        if self.predict_tag == 1:
            self.tag_type = self.predict_data.loc[self.predict_data.index[-1],'TAG_TYPE']
            self.predict_data = self.predict_data.drop('TAG_TYPE',axis = 1)
        else:
            self.tag_type = ''
        
        
        self.predict_data_backup = self.predict_data.copy()#预防后面发送预测结果的时候需要某些字段的原始信息
        self.predicting_data_processing()#run processing function
        return self.predict_tag#if the predict_tag is not 1, we don`t need to run next predicting function. 

    
    def predicting_data_processing(self):
    #This method is defined to process predicting data to features.
        input_length = self.parameters['input_length']
        pca_path = self.parameters['pca_path']
        try:
            long_input = self.parameters['long_input']
        except KeyError:
            if self.parameters['model'] == 'FC_Ngram': 
                long_input = True
            else:
                long_input = False
        
        #use the predicting mode of FP_model to process data and get features.
        self.FP_model.feature_processing(self.predict_data,mode = 'predicting')
        self.predict_x_data = self.FP_model.get_features()
        
        #for some model like attention mechanism, we need to input 2 batches of data
        self.predict_x_data = np.array([self.predict_x_data.iloc[((-input_length*2-1)):-input_length-1,:].values,self.predict_x_data.iloc[-input_length-1:-1,:].values])
        
        
        #use PCA to reduce the dimension.
        if long_input:
            self.predict_x_data = self.predict_x_data.reshape(self.predict_x_data.shape[0],-1)
        elif long_input == False:
            self.predict_x_data = self.predict_x_data.reshape(self.predict_x_data.shape[0]*self.predict_x_data.shape[1],self.predict_x_data.shape[2])

        pca = joblib.load(pca_path)
        self.predict_x_data = pca.transform(self.predict_x_data)
        
        if long_input == False:
            self.predict_x_data = self.predict_x_data.reshape(-1,input_length,self.predict_x_data.shape[1])
 
    
    def get_predicting_X(self):
    #This method is defined to return prdicting features.
        return self.predict_x_data
    
    def get_predict_data_by_loc(self,row,column):
    #This method is defined to return some important raw information from raw data.    
        index = self.predict_data_backup.index[row]
        return self.predict_data_backup.loc[index,column]
    
    def get_predict_tag_type(self):
    #rerurn TAG_TYPE
        return self.tag_type

if __name__ == '__main__':
    #test
    from Unified_Model.Examples.Parameters_Controll import Parameters_Controll
    pc = Parameters_Controll(path = 'parameters_1.json')
    parameters = pc.get_parameters()
    dp = Data_Processing(parameters)
    #a = dp.training_data_loading()
    past_second = parameters['input_length'] * 5 * parameters['aggregate'] * 60
    dp.predicting_data_loading(past_second,end = int(time.time()))
    a,b = dp.get_predicting_X()