#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:33:51 2019

This abstract class is defined as a specification to process features from data. 
It is an element of Data_Processing_base which can be replaced by another subclass
of Feature_Processing_base. In this class, we should initialize it with parameters from outside.
In this abstract class, all the methods should be implemented in subclass.

@author: jungangzou
"""
import abc


class Feature_Processing_base(metaclass = abc.ABCMeta):
    def __init__(self,parameters):
    #Initialize
        self.parameters = parameters

    @abc.abstractmethod
    def feature_processing(self,data):
    #This abstract method is defined to process features.
        pass
    
    @abc.abstractmethod
    def get_features(self):
    #This abstract method is defined to return features to DP_model
        pass