#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:49:05 2019

This abstract class is defined as a specification to return predicting result
by email or special ways. In this class, we should initialize it with parameters
from outside and use the data conveyed to make messages and send. 
In this abstract class, all the methods should be implemented in subclass.

@author: jungangzou
"""

import abc

class Prediction_Result_Return_base(metaclass = abc.ABCMeta):
    def __init__(self,parameters):
    #Initialize
        self.parameters = parameters

    @abc.abstractmethod
    def make_message(self,data):
    #This abstract method is defined to make the sent message.
        pass
    
    @abc.abstractmethod
    def send_message(self):
    #This abstract method is defined to send the message to reciever.
        pass