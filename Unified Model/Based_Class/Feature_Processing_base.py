#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:17:00 2019

@author: jungangzou
"""

import abc
import pandas as pd
import numpy as np

class Feature_Processing_base(metaclass = abc.ABCMeta):
    def __init__(self,data,label,parameters):
    #Initialize
        self.data = data
        self.label = label
        self.parameters = parameters

    @abc.abstractmethod
    def feature_processing(self):
    #This abstract method is defined to get training data from DP_model
        pass
    
    @abc.abstractmethod
    def get_features(self):
    #This abstract method is defined to get testing data from DP_model
        pass