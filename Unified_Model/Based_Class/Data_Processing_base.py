#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:06:29 2019

This abstract class is defined as a specification to process data and provide training
data, validation data and prediction data. 
It contains an element which is a subclass of Feature_Processing_base.
In this class, we should initialize it with parameters from outside.
In this abstract class, all the methods should be implemented in subclass.


@author: jungangzou
"""

import abc
from Unified_Model.Based_Class.Feature_Processing_base import Feature_Processing_base


class Data_Processing_base(metaclass = abc.ABCMeta):
    def __init__(self,parameters,FP_model):
    #Initialize 
        if not isinstance(FP_model,Feature_Processing_base):
            raise TypeError('Please give a Feature_Processing instance')
        self.FP_model = FP_model        
        self.parameters = parameters                
    
    
    @abc.abstractmethod
    def training_data_loading(self):
    #This abstract method is defined to load training data, and process training data.
        self.training_data_processing()
    
    @abc.abstractmethod
    def training_data_processing(self):
    #This abstract method is defined to process training data to features. 
        pass    
    
    @abc.abstractmethod
    def get_training_X(self):
    #This abstract method is defined to return training features.
        pass
    
    @abc.abstractmethod
    def get_training_Y(self):
    #This abstract method is defined to return training labels.
        pass
    
    @abc.abstractmethod
    def get_val_X(self):
    #This abstract method is defined to return validation features.
        pass
    
    @abc.abstractmethod
    def get_val_Y(self):
    #This abstract method is defined to return validation labels.
        pass
    
    @abc.abstractmethod
    def predicting_data_loading(self):
    #This abstract method is defined to load predicting data, and process predicting data.
        self.predicting_data_processing()    
    
    @abc.abstractmethod
    def predicting_data_processing(self):
    #This abstract method is defined to process predicting data to features. 
        pass    
    
    @abc.abstractmethod
    def get_predicting_X(self):
    #This abstract method is defined to get predicting features.
        pass
    
    
    
    
