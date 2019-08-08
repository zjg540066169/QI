#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 09:18:44 2019

This class is a subclass of Prediction_Result_Return_base to send the message prediction result .

@author: jungangzou
"""

from Unified_Model.Based_Class.Prediction_Result_Return_base import Prediction_Result_Return_base
import time
import requests


class Prediction_Result_Return(Prediction_Result_Return_base):
    def __init__(self,parameters):
    #Initialize
        super().__init__(parameters)
        self.parameters = parameters

    def make_message(self,data):
    #This method is to make the sent message from specific data
    
    
    #the data from Service
        predict_result = data[0] #this is the number of output with the max softmax probability, is a number
        predict_prob = data[1] #this is the probability of outputs for each number, is a list
        
        current_price = data[2] #this is the current close price of predicting data, is a number
        tag_type = data[3] #this is TAG_TYPE, which is a string
        
    #make sent message
        string = '【'+self.parameters['model']+'】 15M '+self.parameters['coin']+'/'+self.parameters['to']+' 现价 '+str(current_price)+'\n'
        string += '未来8K'
        if predict_result == 1:
            string += '【破DC高轨】概率'+str('%.2f%%' %(100*predict_prob[1]))
        elif predict_result == 2:
            string += '【破DC低轨】概率 '+str('%.2f%%' %(100*predict_prob[2]))
        elif predict_result == 0:
            string += '【不破高低】概率 '+str('%.2f%%' %(100*predict_prob[0]))
        string += '\n'
        string += '信号备注：'+tag_type
        self.string = string.replace("--","-")
        


    def send_message(self):
    #This method is defined to send the message to remote server.
        #response = requests.post('http://106.75.90.217:30804/openapi/coin/signal/report',data=self.__sendMessage(self.string))
                    
        #print(response.text)
        print('send',self.string)
    
    
    def __stampToTime(self,timeStamp):
    #this function is used to tranlate time stamp to time string
        timeArray = time.localtime(timeStamp)
        otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
        return otherStyleTime    

    

    def __sendMessage(self,string):
    #make the required parameters to send message
        message = {}
        message["time"] = int(time.time())
        message["type"] = "alarm"
        message["degree"] = 1
        message["coin_from"] = self.parameters['coin']
        message["coin_to"] = self.parameters['to']
        message["signal"] = "High_Low_Predict"
        message["message"] = string
        return message
