#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:22:04 2019

This class is a subclass of Parameters_Controll_base to control parameters.
In this class, we can interact parameters with local system. 
Only when the parameters are saved as a JSON file can we load the parameters. 

@author: jungangzou
"""

from Unified_Model.Based_Class.Parameters_Control_base import Parameters_Control_base
import json

class Parameters_Control(Parameters_Control_base):
    def __init__(self,parameters = None, path = None):
    #Initialize with parameters or load parameters from local file system
        super().__init__(parameters)
        self.parameters = parameters
        if parameters is None and path is not None:
            self.parameters = self.read_parameters_from_file(path)

    def read_parameters_from_file(self,path):
    #read parameters from a JSON file
        with open(path,'r') as f:
            parameters_str = f.read().replace('\n','')
        parameters = json.loads(parameters_str)
        
        return parameters
    
    def write_parameters_to_file(self,path):
    #write the parameters to local file system
        with open(path,'w') as f:
            context = json.dumps(self.parameters)
            #print(context.replace(',',',\n'))
            f.write(context.replace(',',',\n'))
    
    def get_parameters(self):
    #return parameters
        return self.parameters

    def set_parameters(self,parameters):
    #set parameters
        self.parameters = parameters
        
        
if __name__ == '__main__':
    #test
    pc = Parameters_Control(path = 'parameters_1.json')
    pc.write_parameters_to_file('parameters_2.json')