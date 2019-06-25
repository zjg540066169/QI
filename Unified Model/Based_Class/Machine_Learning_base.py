#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:05:18 2019

@author: jungangzou
"""

import abc

class Machine_Learning_base(metaclass = abc.ABCMeta):
    def __init__(self,ML_model,parameters):
    #Initialize
        self.ML_model = ML_model
        self.parameters = parameters

    @abc.abstractmethod
    def train(self):
    #This abstract method is defined to get training data from DP_model
        pass
    
    @abc.abstractmethod
    def test(self):
    #This abstract method is defined to get testing data from DP_model
        pass