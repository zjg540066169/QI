#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:12:02 2019

This class is a subclass of Machine_Learning_base which contains 3 Neural Network Models.
In this class, we can load the local model by parameter "model_path" or initialize
model by parameter "model" which is the name of NN model.
For parameter "long_input", only can be true when model is "FC_Ngram".



@author: jungangzou
"""
from Unified_Model.Machine_Learning_Models.Attention_RNN import Attention_RNN
from Unified_Model.Machine_Learning_Models.Self_Attention import Self_Attention
from Unified_Model.Machine_Learning_Models.FC_Ngram import FC_Ngram
from Unified_Model.Based_Class.Machine_Learning_base import Machine_Learning_base
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch import optim
from sklearn.metrics import f1_score
import torch.nn.functional as F
import numpy as np

class Machine_Learning_Model_NN(Machine_Learning_base):
    def __init__(self,parameters):
    #Initialize, we can load the model from local file system or initial with parameters
        super().__init__(parameters)
        self.parameters = parameters
        self.trained_flag = False#if model is not trained, this flag is false, then we cannot use the predicting and testing function.
        #initialize model with parameters if we cannot load the model from local system
        try:
            self.model = torch.load(self.parameters['model_path'])
            self.trained_flag = True
        except KeyError and FileNotFoundError:
            self.__model_identify()
        
    def __model_identify(self):
        #this function is used to initialize model with parameters
        
        #set PCA dimension
        try:
            input_size = self.parameters['pca_dimension']
        except KeyError:
            input_size = 50
            
        #set some important parameters
        hidden_size = self.parameters['hidden_size']
        output_size = self.parameters['output_size']
        if 'dropout' in self.parameters:
            dropout = self.parameters['dropout']
        else:
            dropout = 0.5
        
        
        
        
        #for different models, use different parameters.
        if self.parameters['model'] == 'FC_Ngram':
        #if model is 'FC_Ngram', the parameter "long_input" must be True/
            if self.parameters['long_input'] == False:
                raise ValueError('Please set "long_input" as "True" in parameter sets if you want to use FC_Ngram Model')
            if 'activation_function' in self.parameters:#select activation function
                activation_function_list = self.parameters['activation_function']
            else:
                activation_function_list = 'relu'
            self.model = FC_Ngram(input_size,hidden_size,output_size,activation_function_list,dropout)
          
            
        elif self.parameters['model'] == 'Attention_RNN':
            if 'n_layers' in self.parameters:#number of attention layer
                n_layers = self.parameters['n_layers']
            else:
                n_layers = 1
            if 'rnnmodel' in self.parameters:#select a rnnmodel for attention mechanism in 'GRU','LSTM','RNN'
                rnnmodel = self.parameters['rnnmodel']
            else:
                rnnmodel = 'GRU'
            self.model = Attention_RNN(input_size,hidden_size, output_size, n_layers,rnnmodel,dropout)
            
            
        elif self.parameters['model'] == 'Self_Attention':
            
            if 'n_encoder' in self.parameters:#number of self-attention layer
                n_encoder = self.parameters['n_encoder']
            else:
                n_encoder = 1
            if 'heads' in self.parameters:#number of multi-heads mechanism
                heads = self.parameters['heads']
            else:
                heads = 1
            self.model = Self_Attention(input_size,hidden_size, output_size, n_encoder, heads, dropout)


    def train(self,train_x,train_y,val_x,val_y):
    #this function is used to train the NN model, and the data recieved must be Numpy Array.
    #input:training data and validation data (numpy.ndarray)
    #output:list of training f1_score and list of validation f1_score (2 list)
    
        if type(train_x) == type(train_y) == type(val_x) == type(val_y) == np.ndarray:
            pass
        else:
            raise TypeError("Please send the data to ML_Model whose type is numpy.ndarray")
        
          
        #set parameters 'batch_size' and 'epochs'
        if 'batch_size' in self.parameters:
            self.batch_size = self.parameters['batch_size']
        else:
            self.batch_size = 32
            
        if 'epochs' in self.parameters:
            epochs = self.parameters['epochs']
        else:
            epochs = 120
            
        
        #use PyTorch to train the models
        train_dataset = TensorDataset(torch.Tensor(train_x),torch.LongTensor(train_y))
        train_dataloader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True,drop_last=True)
        val_dataset = TensorDataset(torch.Tensor(val_x),torch.LongTensor(val_y))
        val_dataloader = DataLoader(val_dataset,batch_size=self.batch_size,shuffle=True,drop_last=True)

        self.model.train()
        cost = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_f1_score  = []
        self.val_f1_score = []

        for epoch in range(epochs):
            train_accuracy_total = 0
            for iter,(inputs,labels) in enumerate(train_dataloader):
                #print(labels.size())
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = cost(outputs,labels.view(-1))
                train_accuracy_total += f1_score(labels.numpy(), torch.max(outputs.data,1)[1].numpy(), average='micro')
                loss.backward()
                optimizer.step()
            train_accuracy_total /= (iter+1)
            self.train_f1_score.append(train_accuracy_total)
            print("train f1:",train_accuracy_total,end = "    ")
            
            
            self.model.eval()   
            val_accuracy_total = 0
            for iter, (inputs,true_y) in enumerate(val_dataloader):
                val_y = self.model(inputs)
                val_accuracy_total +=  f1_score(true_y.numpy(), torch.max(val_y.data,1)[1].numpy(), average='micro')
            val_accuracy_total /= (iter+1)
            self.val_f1_score.append(val_accuracy_total)
            print("val f1:",val_accuracy_total)    
            self.model.train()
        self.trained_flag = True
        return self.train_f1_score,self.val_f1_score


    def test(self,test_x,test_y):
    #this function is to test model by testing data
    #input:testing data(numpy.ndarray)
    #output:test_f1_score (a number)
        if self.trained_flag == False:
            raise ValueError('Model is not trained, please train it before use')
        if type(test_x) == type(test_y) == np.ndarray:
            pass
        else:
            raise TypeError("Please send the data to ML_Model whose type is numpy.ndarray")

        self.model.eval()
        test_dataset = TensorDataset(torch.Tensor(test_x),torch.LongTensor(test_y))
        test_dataloader = DataLoader(test_dataset,batch_size=self.batch_size,shuffle=True,drop_last=True)

        test_accuracy_total = 0
        for iter, (inputs,true_y) in enumerate(test_dataloader):
            test_y = self.model(inputs)
            test_accuracy_total +=  f1_score(true_y.numpy(), torch.max(test_y.data,1)[1].numpy(), average='micro')
        test_accuracy_total /= (iter+1)
        print("test f1:",test_accuracy_total)    
        return test_accuracy_total

    
    def predict(self,predict_x):
    #this function is to use model to predict data
    #input:predict data(numpy.ndarray)
    #output:predict result(classification) and predict probability for each classification
        if self.trained_flag == False:
            raise ValueError('Model is not trained, please train it before use')

        if type(predict_x) == np.ndarray:
            pass
        else:
            raise TypeError("Please send the data to ML_Model whose type is numpy.ndarray")
        
        self.model.eval()
        inputs= torch.autograd.Variable(torch.Tensor(predict_x))
        if self.parameters['long_input'] == False: 
            outputs = self.model(inputs)[-1] 
        else:
            outputs = self.model(inputs)
        #print(F.softmax(outputs).data)
        #print(torch.max(F.softmax(outputs).data,0)[1].numpy())
        predict_result = torch.max(F.softmax(outputs).data,0)[1].numpy()
        if self.parameters['long_input'] == False: 
            predict_prob = F.softmax(outputs).data.numpy()
        else:
            predict_prob = F.softmax(outputs).data.numpy()
        return predict_result,predict_prob

    
    def get_parameters(self):
    # return parameters
        return self.parameters
    
    
    def set_parameters(self,parameters):
    # set parameters
        self.parameters = parameters
        
    
    def save_model(self,path):
    # save PyTorch model to local system
        if self.trained_flag == False:
            raise Exception('Model is not trained, please train it before save')
        torch.save(self.model, path)
        

    def load_model(self,path):
    #load model from local file system
        self.model = torch.load(path)
        
if __name__ == '__main__':
    from Unified_Model.Examples.Parameters_Controll import Parameters_Controll
    pc = Parameters_Controll(path = 'parameters_1.json')
    parameters = pc.get_parameters()
    ml = Machine_Learning_Model_NN(parameters)
    
