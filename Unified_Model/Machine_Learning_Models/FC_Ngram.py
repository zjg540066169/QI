#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 09:11:25 2019

@author: jungangzou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy



class FC_Ngram(nn.Module):
    def __init__(self, input_size, hidden_size_list,output_size, activation_list = 'relu', dropout = 0.3):
        super().__init__()
        
        if type(hidden_size_list) is list:
            self.hidden_layers_number = len(hidden_size_list)
            hidden_size = [input_size] + hidden_size_list + [output_size]                
        elif type(hidden_size_list) is int:
            self.hidden_layers_number = len([hidden_size_list])
            hidden_size = [input_size] + [hidden_size_list] + [output_size]
            
            
        
        if type(activation_list) is list:
            
            if len(activation_list) == 1:
                self.activation = self.__activation_function_factory(activation_list[0],self.hidden_layers_number)

            elif len(activation_list) != self.hidden_layers_number:
                raise Exception('The number of activation functions is not equal to the number of hidden layers')
                            
            else:
                self.activation = []
                for act in activation_list:
                    self.activation+= self.__activation_function_factory(act)
                
        else:
            if type(activation_list) is str:
                self.activation = self.__activation_function_factory(activation_list,self.hidden_layers_number)

                    
                    
        self.linear_list = []
        for i in range(len(hidden_size)-1):
            self.linear_list.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        
        self.bn = torch.nn.BatchNorm1d(hidden_size[-1])
        self.out = torch.nn.Linear(hidden_size[-1],output_size)

        
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.dropout = dropout
        
    def forward(self, x):
        for num in range(len(self.linear_list)-1):
            x = self.linear_list[num](x)
            x = self.activation[num](x)
        x = self.bn(x)#BatchNormal
        x = F.dropout(x, self.dropout, self.training)
        return x
    
    def __activation_function_factory(self,activation,repeat_number = 1):
        
        activation_functions = ['relu','tanh','sigmoid']
        
        if activation not in activation_functions:
            raise ValueError('Please select activation functions in ',str(activation_functions))
        
        if activation == 'relu':
            return [F.relu]*repeat_number
        elif activation == 'tanh':
            return [F.tanh]*repeat_number
        elif activation == 'sigmoid':
            return [F.sigmoid]*repeat_number

