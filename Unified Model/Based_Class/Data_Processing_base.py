#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:06:29 2019

@author: jungangzou
"""

import abc
from Feature_Processing_base import Feature_Processing_base

class Data_Processing_base(metaclass = abc.ABCMeta):
    def __init__(self,FP_model,training_data_path,predicting_data_path,parameters):
    #Initialize 
        if not isinstance(FP_model,Feature_Processing_base):
            raise TypeError('Please give a Feature_Processing instance')
        self.FP_model = FP_model
        self.training_data_path = training_data_path
        self.predicting_data_path = predicting_data_path
        self.parameters = parameters
                    
    @abc.abstractmethod
    def training_data_loading(self):
    #This abstract method is defined to get training data from DP_model
        pass
    
    @abc.abstractmethod
    def training_data_processing(self):
    #This abstract method is defined to get testing data from DP_model
        pass    
    
    @abc.abstractmethod
    def get_training_X(self):
    #This abstract method is defined to get predicting data from DP_model
        pass
    
    @abc.abstractmethod
    def get_training_Y(self):
    #This abstract method is defined to get predicting data from DP_model
        pass
    
    @abc.abstractmethod
    def get_testing_X(self):
    #This abstract method is defined to get predicting data from DP_model
        pass
    
    @abc.abstractmethod
    def get_testing_Y(self):
    #This abstract method is defined to get predicting data from DP_model
        pass
    
    @abc.abstractmethod
    def predicting_data_loading(self):
    #This abstract method is defined to get training data from DP_model
        pass
    
    @abc.abstractmethod
    def predicting_data_processing(self):
    #This abstract method is defined to get testing data from DP_model
        pass    
    
    @abc.abstractmethod
    def get_predicting_X(self):
    #This abstract method is defined to get predicting data from DP_model
        pass