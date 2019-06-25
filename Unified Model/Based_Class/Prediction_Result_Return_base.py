#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:49:05 2019

@author: jungangzou
"""

import abc

class Prediction_Result_Return_base(metaclass = abc.ABCMeta):
    def __init__(self,parameters):
    #Initialize
        self.parameters = parameters

    @abc.abstractmethod
    def make_message(self,data):
    #This abstract method is defined to get training data from DP_model
        pass
    
    @abc.abstractmethod
    def send_message(self):
    #This abstract method is defined to get training data from DP_model
        pass