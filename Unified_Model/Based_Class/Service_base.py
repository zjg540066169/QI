#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:57:57 2019

This abstract class is defined for all the stock_predicting model, which contains 
4 parts: machine learning models, data processing, prediction result return, 
parameter controll. In this abstract class, all the methods should be implemented in subclass.

@author: jungangzou
"""
import abc
from Unified_Model.Based_Class.Data_Processing_base import Data_Processing_base
from Unified_Model.Based_Class.Machine_Learning_base import Machine_Learning_base
from Unified_Model.Based_Class.Prediction_Result_Return_base import Prediction_Result_Return_base
from Unified_Model.Based_Class.Parameters_Controll_base import Parameters_Controll_base

class Service_base(metaclass = abc.ABCMeta):
    def __init__(self,ML_model,DP_model,PR_model,PC_model):
    #Initialize
        if not isinstance(ML_model,Machine_Learning_base):
            raise TypeError('Please give a Machine_Learning model instance')
            
        if not isinstance(DP_model,Data_Processing_base):
            raise TypeError('Please give a Data_Processing model instance')
            
        if not isinstance(PR_model,Prediction_Result_Return_base):
            raise TypeError('Please give a Prediction_Result_Return model instance')
            
        if not isinstance(PC_model,Parameters_Controll_base):
            raise TypeError('Please give a Parameters_Controll model instance')

        self.ML_model = ML_model
        self.DP_model = DP_model
        self.PR_model = PR_model
        self.PC_model = PC_model

    @abc.abstractmethod
    def get_training_data(self):
    #This abstract method is defined to get training data from DP_model
        pass
    
    @abc.abstractmethod
    def get_testing_data(self):
    #This abstract method is defined to get testing data from DP_model
        pass    
    
    @abc.abstractmethod
    def get_predicting_data(self):
    #This abstract method is defined to get predicting data from DP_model
        pass    
    
    @abc.abstractmethod
    def train(self):
    #This abstract method is defined to train model from ML_model
        pass    
    
    @abc.abstractmethod
    def test(self):
    #This abstract method is defined to test model from ML_model
        pass    
    
    @abc.abstractmethod
    def predict(self):
    #This abstract method is defined to predict data from ML_model
        pass    
    
    @abc.abstractmethod
    def result_return(self):
    #This abstract method is defined to return prediction result from PR_model
        pass    
    
    @abc.abstractmethod
    def parameter_controll(self):
    #This abstract method is defined to controll parameters from PC_model
        pass    