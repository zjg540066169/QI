#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:52:46 2019

@author: jungangzou
"""

import abc

class Parameters_Controll_base(metaclass = abc.ABCMeta):
    def __init__(self,parameters):
    #Initialize
        self.parameters = parameters

    @abc.abstractmethod
    def read_parameters_from_file(self,path):
    #This abstract method is defined to get training data from DP_model
        pass
    
    @abc.abstractmethod
    def write_parameters_to_file(self,path):
    #This abstract method is defined to get training data from DP_model
        pass
