#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:46:42 2019
This class is a subclass of Service_base, which controll 4 parts of the whole models.
In this class, we can train models and predict, also send messages and save parameters.

@author: jungangzou
"""
from Unified_Model.Based_Class.Service_base import Service_base
import datetime,pytz,time


class Service(Service_base):
    def __init__(self,ML_model,DP_model,PR_model,PC_model):
    #Initialize
        super().__init__(ML_model,DP_model,PR_model,PC_model)
        self.ML_model = ML_model
        self.DP_model = DP_model
        self.PR_model = PR_model
        self.PC_model = PC_model
        self.parameters = self.PC_model.get_parameters()
   
    def get_training_data(self):
    #This method returns trainging features/labels, validation features/labels with numpy.array type
        self.DP_model.training_data_loading()
        return self.DP_model.get_training_X(),self.DP_model.get_training_Y(),self.DP_model.get_val_X(),self.DP_model.get_val_Y()
    
    
    def get_testing_data(self):
    #This method is defined to get testing data. In this example we don`t need this function.
        pass    
    

    def get_predicting_data(self):
    #This method returns predicting_tag and predicting features with numpy.array type
        try:
            past_second = self.parameters['past_second']
        except KeyError:
            if self.parameters['timespan'][-1] == 'M':
                past_second = self.parameters['input_length'] * 5 * self.parameters['aggregate'] * 60
            elif self.parameters['timespan'][-1] == 'H':
                past_second = self.parameters['input_length'] * 5 * self.parameters['aggregate'] * 3600
        predicting_tag = self.DP_model.predicting_data_loading(past_second,int(time.time()),start=0)
        return predicting_tag, self.DP_model.get_predicting_X()
    

    def train(self):
    #This method is defined to train model and save it.
        self.train_x, self.train_y, self.val_x, self.val_y = self.get_training_data()
        self.train_f1_score, self.val_f1_score = self.ML_model.train(self.train_x, self.train_y, self.val_x, self.val_y)
        self.parameters['self.train_f1_score'] = self.train_f1_score
        self.parameters['self.val_f1_score'] = self.val_f1_score 
        self.ML_model.save_model(self.parameters['model_path'])

    def test(self):
    #This method is defined to test model from ML_model
        pass    
    

    def predict(self):
    #This method is predict result
        self.tag, self.predicting_x = self.get_predicting_data() 
        if self.tag == 0:#if predict tag is 0, stop. 
            return False
        try:
            self.predict_result, self.predict_prob = self.ML_model.predict(self.predicting_x)
        except ValueError as e:# if model is not trained, train it.
            print(e)
            self.train()
            self.predict_result, self.predict_prob = self.ML_model.predict(self.predicting_x)
        
        #save the predicting_result to parameters.
        try:
            self.parameters['predict_result'].append(self.predict_result.tolist())
            self.parameters['predict_prob'].append(self.predict_prob.tolist())
        except KeyError:
            self.parameters['predict_result'] = [self.predict_result.tolist()]
            self.parameters['predict_prob'] = [self.predict_prob.tolist()]
        self.TAG_TYPE = self.DP_model.get_predict_tag_type()
        return True

    def result_return(self):
    #This method is defined to send message to remote server.
    
        #make the data sent to PR_model.
        return_data = [self.predict_result, self.predict_prob, self.DP_model.get_predict_data_by_loc(-1,'close'), self.TAG_TYPE]
        
        
        #send message.
        self.PR_model.make_message(return_data)
        self.PR_model.send_message()
        
        
        #save the predicting time to parameters.
        timeNow = datetime.datetime.now()
        timeNow = timeNow.astimezone(pytz.timezone('Asia/Shanghai'))
        self.timeNow = timeNow
        time = self.__stampToTime(timeNow.timestamp()+self.parameters['timeblock'])[6:9]+self.__stampToTime(timeNow.timestamp()+self.parameters['timeblock'])[10:18]

        try:
            self.parameters['predict_time'].append(time)
        except KeyError:
            self.parameters['predict_time'] = [time]

        
    def parameter_controll(self,path):
        #save parameters to path.
        self.PC_model.set_parameters(self.parameters)
        self.PC_model.write_parameters_to_file(path)
            
    def __stampToTime(self,timeStamp):
        #convert timestamp to time string.
        timeArray = time.localtime(timeStamp)
        otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
        return otherStyleTime    

if __name__ == '__main__':
    #predict
    from Unified_Model.Examples.Data_Processing import Data_Processing
    from Unified_Model.Examples.Machine_Learning_Model_NN import Machine_Learning_Model_NN
    from Unified_Model.Examples.Parameters_Controll import Parameters_Controll
    from Unified_Model.Examples.Prediction_Result_Return import Prediction_Result_Return
    
    #initialize
    pc = Parameters_Controll(path = 'parameters_1.json')
    parameters = pc.get_parameters()
    ml = Machine_Learning_Model_NN(parameters)
    pr = Prediction_Result_Return(parameters)
    dp = Data_Processing(parameters)
    ser = Service(ml,dp,pr,pc)
    
    #predict
    while True:
        timeNow = datetime.datetime.now()
        timeNow = timeNow.astimezone(pytz.timezone('Asia/Shanghai'))
        if (timeNow.minute + parameters["send_lag"]) % 15 == 0:
            if ser.predict() == True:
                ser.result_return()
                ser.parameter_controll("parameters_2.json")
        time.sleep(300)
