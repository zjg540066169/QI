#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:55:41 2019

@author: jungangzou
"""
import torch
import torch.nn.functional as F
from data_process import data_process
from sklearn.model_selection import train_test_split
from torch import optim
from sklearn.metrics import f1_score
from transformer_model import Transformer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset



BATCH_SIZE = 32


class Train_Net(object):
    def __init__(self,path,save_path,pca_path):
        data_proces = data_process(path,pca_path,long_input = False)
        data_proces.process()
        self.X,self.Y = data_proces.get_X(),data_proces.get_Y()[:,0]

        self.save_path = save_path
        
    def train(self,train_acc = 0.95,test_acc = 0.95):
        #return self.X,self.Y
        continue_train = True
        while continue_train:
            train_x, val_x, train_y, val_y = train_test_split(self.X, self.Y, test_size=0.33)
            train_dataset = TensorDataset(torch.Tensor(train_x),torch.LongTensor(train_y))
            train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
            test_dataset = TensorDataset(torch.Tensor(val_x),torch.LongTensor(val_y))
            test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
            
            net = Transformer(train_x.shape[2],1,1,0.3) 
            cost = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            mse  = []
            epochnum = 240
            for epoch in range(epochnum):
                train_accuracy_total = 0
                for iter,(inputs,labels) in enumerate(train_dataloader):
                    #print(labels.size())
                    optimizer.zero_grad()
                    outputs = net(inputs,torch.zeros(BATCH_SIZE,1,train_x.shape[2]),None,None).squeeze(1)
                    loss = cost(outputs,labels.view(-1))
                    train_accuracy_total += f1_score(labels.numpy(), torch.max(outputs.data,1)[1].numpy(), average='micro')
                    loss.backward()
                    optimizer.step()
                train_accuracy_total /= (iter+1)
                mse.append(train_accuracy_total)
                print("f1:",train_accuracy_total,end = "    ")
            
                
                net.eval()   
                pre_accuracy_total = 0
                for iter, (inputs,true_y) in enumerate(test_dataloader):
                    predict_y = net(inputs,torch.zeros(BATCH_SIZE,1,train_x.shape[2]),None,None).squeeze(1)
                #print(pd.Series(torch.max(predict_y.data,1)[1].numpy()).value_counts()[1]/pd.Series(val_y).value_counts()[1])
                    pre_accuracy_total +=  f1_score(true_y.numpy(), torch.max(predict_y.data,1)[1].numpy(), average='micro')
                pre_accuracy_total /= (iter+1)
                print("validation f1:",pre_accuracy_total)    
                net.train()
                if train_accuracy_total > train_acc and pre_accuracy_total > test_acc:
                    continue_train = False
                    break
        torch.save(net, self.save_path)
        

if __name__ == '__main__':
    path_list = [r'../data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_15m_singal_regular_20190226.json']#,r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_1h_singal_regular_20190226.json',r'data/bitfinex_2017-01-01_to_2019-01-15_ltc_usdt_1h_singal_regular_20190226.json']#,r'data/cccagg_1498676400_to_1550865600_eos_usd_1h_singal_regular_20190226.json',r'data/cccagg_1521208800_to_1550977200_ont_usd_1h_singal_regular_20190226.json']
    path = r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_15m_singal_regular_20190226.json'
    pca_path = r'BTC60LSTMPCA.m'
    save_path = r'BTC60Transformer.pth'
    tn = Train_Net(path_list,save_path,pca_path)
    
    a,b = tn.train()