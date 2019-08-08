#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:05:18 2019

This abstract class is defined as a specification to use different machine learning  models
which is used to train models and predict. 
In this class, we should initialize it with parameters from outside.
We defined some nessesary methods to operate machine learning models. 
In this abstract class, all the methods should be implemented in subclass.

@author: jungangzou
"""

import abc

class Machine_Learning_base(metaclass = abc.ABCMeta):
    def __init__(self,parameters):
    #Initialize
        self.parameters = parameters


    @abc.abstractmethod
    def train(self,train_x,train_y,val_x,val_y):
    #This abstract method is defined to train machine learning model.
        pass
  
    
    @abc.abstractmethod
    def test(self,test_x,test_y):
    #This abstract method is defined to test model.
        pass

    
    @abc.abstractmethod
    def predict(self,predict_x):
    #This abstract method is defined to predict result.
        pass