# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 21:28:10 2019

@author: wang
"""

from train_net import LSTMClassifier,MulTaskLoss
import torch
import torch.nn.functional as F
from data_process import data_process
from sklearn.model_selection import train_test_split
from torch import optim
from sklearn.metrics import f1_score
import numpy as np
class Test_Net(object):
    def __init__(self,save_path):
        self.net = torch.load(save_path)
        

    def test(self,path,pca_path,train_acc = 0.85,test_acc = 0.85):
        data_proces = data_process(path,pca_path)
        data_proces.process()
        self.X,self.Y = data_proces.get_X(),data_proces.get_Y().values

        self.net.eval()
        mse = []
        for i in range(10):
            inputs, labels = torch.autograd.Variable(torch.Tensor(self.X[int(len(self.X)/10*i):int(len(self.X)/10*(i+1)),:])), torch.autograd.Variable(torch.LongTensor(self.Y[int(len(self.X)/10*i):int(len(self.X)/10*(i+1)),:]))
            outputs = self.net(inputs)
            train_accuracy = 0
            for i in range(4):
                print(f1_score(labels[:,i].numpy(), torch.max(outputs[i].data,1)[1].numpy(), average='micro'))
                train_accuracy += f1_score(labels[:,i].numpy(), torch.max(outputs[i].data,1)[1].numpy(), average='micro')
            train_accuracy /= 4
            print("f1:",train_accuracy,end = "    ")
            mse.append(train_accuracy)
        return np.array(mse).mean()
    
    def predict(self,predict_X):
        self.net.eval()
        inputs= torch.autograd.Variable(torch.Tensor(predict_X))
        outputs = self.net(inputs)
        predict_result,predict_prob = [],[]
        for i in range(4):
            predict_result.append(torch.max(F.softmax(outputs[i]).data,1)[1].numpy())
            predict_prob.append(F.softmax(outputs[i]).data.numpy())
        return predict_result,predict_prob

    
if __name__=='__main__':
    path = r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_15m_singal_regular_20190226.json'
    pca_path = r'BTC/15PCA.m'
    save_path = r'BTC/15LSTM.pth'
    tn = Test_Net(save_path)
    print(tn.test(path,pca_path))
    
