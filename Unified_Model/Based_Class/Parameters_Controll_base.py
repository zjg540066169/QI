#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:52:46 2019

This abstract class is defined as a specification to controll all the parameters. 
In this class, we should initialize it with parameters from outside.
We defined some method to operate parameters include read, write, update. 
In this abstract class, all the methods should be implemented in subclass.

@author: jungangzou
"""

import abc

class Parameters_Controll_base(metaclass = abc.ABCMeta):
    def __init__(self,parameters):
    #Initialize
        self.parameters = parameters

    @abc.abstractmethod
    def read_parameters_from_file(self,path):
    #This abstract method is defined to read the parameters from a file.
        pass
    
    @abc.abstractmethod
    def write_parameters_to_file(self,path):
    #This abstract method is defined to write the parameters to file.
        pass

    @abc.abstractmethod
    def get_parameters(self):
    #This abstract method is defined to return parameters to another class.
        pass

    @abc.abstractmethod
    def set_parameters(self,parameters):
    #This abstract method is defined to set parameters.
        pass
