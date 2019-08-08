#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:16:47 2019

@author: jungangzou
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 21:47:57 2019

@author: wang
"""
import torch
import torch.nn.functional as F
from data_process import data_process
from sklearn.model_selection import train_test_split
from torch import optim
from sklearn.metrics import f1_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from attentionRNN import BahdanauAttnDecoderRNN,EncoderRNN
from transformer_model import Transformer
import datetime
import pandas as pd

sns.set_style('darkgrid')
BATCH_SIZE = 32
epochnum = 80
class LSTMClassifier(torch.nn.Module):
    def __init__(self,input_size,input_length = 96):
        super(LSTMClassifier, self).__init__()
        self.hidden = 100
        self.input_size = input_size[2]
        self.lstm = torch.nn.LSTM(input_size[2], self.hidden,batch_first=True)

        self.bn1 = torch.nn.BatchNorm1d(input_length)
        self.out1 = torch.nn.Linear(self.hidden,3)

    
    def forward(self,x):
        
        x = self.lstm(x)[0]
        x = F.relu(x)
        #print(x1.size())
        x1 = self.bn1(x)#BatchNormal
        x1 = self.out1(x1)
        x1 = F.dropout(x1, 0.3, self.training)[:,-1,:].view(-1,3)
        
        return x1
    
class FCClassifier(torch.nn.Module):
    def __init__(self,input_size,input_length = 96,activation='relu'):
        super(FCClassifier, self).__init__()
        self.hidden = 300
        self.activation = activation
        self.input_size = input_size[1]
        self.fc = torch.nn.Linear(input_size[1], self.hidden)
        self.bn1 = torch.nn.BatchNorm1d(self.hidden)
        self.out1 = torch.nn.Linear(self.hidden,3)

        
    def forward(self,x): 
        x = self.fc(x)
        if self.activation=='sigmoid':
            x = F.sigmoid(x)
        elif self.activation=='tanh':
            x = F.tanh(x)
        elif self.activation=='relu':
            x = F.relu(x)

        #print(x1.size())
        x1 = self.bn1(x)#BatchNormal
        x1 = self.out1(x1)
        x1 = F.dropout(x1, 0.3, self.training).view(-1,3)
        
        return x1
 
    
class Attention_GRU(torch.nn.Module):
    def __init__(self, input_size,hidden_size, output_size, n_layers=1, dropout_p=0.3):
        super(Attention_GRU, self).__init__()
        # Define parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.encode = EncoderRNN(input_size,hidden_size,n_layers=n_layers)
        self.decode = BahdanauAttnDecoderRNN(hidden_size,output_size,n_layers=n_layers,dropout_p=dropout_p)

    def forward(self, word_input):
        encoder_output,encoder_hidden = self.encode(word_input)
        decoder_output,decoder_hidden = self.decode(torch.zeros(BATCH_SIZE,1,self.output_size),encoder_hidden,encoder_output)
        return decoder_output
    

class MulTaskLoss(torch.nn.Module):
    def __init__(self):
        super(MulTaskLoss, self).__init__()
        
    def forward(self,outputs,labels):
        loss0 = torch.nn.CrossEntropyLoss()(outputs,labels)        

        return loss0
    
class Train_Net(object):
    def __init__(self,path,save_path,pca_path,network,input_length,last_length,predict_length):
        if network == 'FC':
            self.long_input = True
        else:
            self.long_input = False
        self.input_length = input_length
        self.last_length = last_length
        self.predict_length = predict_length
        self.network = network
        data_proces = data_process(path,pca_path,long_input = self.long_input,input_length = self.input_length,predict_length = predict_length,last_length = last_length)
        data_proces.process()
        self.X,self.Y = data_proces.get_X(),data_proces.get_Y()
        self.save_path = save_path
        
    def train(self,train_acc = 0.85,test_acc = 0.78,activation='relu'):
        #return self.X,self.Y
        train_mse = []
        val_mse = []
        for time in range(1):
            train_x, val_x, train_y, val_y = train_test_split(self.X, self.Y, test_size=0.33)
            if self.network=='FC':
                net = FCClassifier(train_x.shape,activation)
            elif self.network=='Self-attention':
                net = Transformer(train_x.shape[2],1,1,0.3) 
            elif self.network=='LSTM':
                net = LSTMClassifier(train_x.shape,self.input_length)
            elif self.network=='LSTM-attention':
                net = Attention_GRU(50,100,3)

            train_dataset = TensorDataset(torch.Tensor(train_x),torch.LongTensor(train_y))
            train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
            test_dataset = TensorDataset(torch.Tensor(val_x),torch.LongTensor(val_y))
            test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

            
            cost = MulTaskLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            mse  = []
            v_mse = []
            
            for epoch in range(epochnum):
                train_accuracy_total = 0
                for iter,(inputs,labels) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    if self.network == 'Self-attention':
                        outputs = net(inputs,torch.zeros(BATCH_SIZE,1,train_x.shape[2]),None,None).squeeze(1)
                        loss = cost(outputs,labels.view(-1))
                    elif self.network == 'LSTM-attention':
                        outputs = net(inputs)
                        loss = cost(outputs,labels.view(-1))
                    else:
                        outputs = net(inputs)
                        loss = cost(outputs,labels)
                    train_accuracy = 0
                    for i in range(1):
                        train_accuracy += f1_score(labels.numpy(), torch.max(outputs.data,1)[1].numpy(), average='micro')
                    train_accuracy /= 1
                    train_accuracy_total += train_accuracy
                    loss.backward()
                    optimizer.step()
                
                train_accuracy_total /= (iter+1)

                print("f1:",train_accuracy_total,end = "    ")
                mse.append(train_accuracy_total)
            
                net.eval()   
                pre_accuracy_total = 0
                for iter, (inputs,true_y) in enumerate(test_dataloader):
                    if self.network == 'Self-attention':
                        predict_y = net(inputs,torch.zeros(BATCH_SIZE,1,train_x.shape[2]),None,None).squeeze(1)
                    else:
                        predict_y = net(inputs)
                    pre_accuracy = 0
                #print(pd.Series(torch.max(predict_y.data,1)[1].numpy()).value_counts()[1]/pd.Series(val_y).value_counts()[1])
                    for i in range(1):
                        pre_accuracy += f1_score(true_y.numpy(), torch.max(predict_y.data,1)[1].numpy(), average='micro')
                    pre_accuracy /= 1
                    pre_accuracy_total += pre_accuracy
                pre_accuracy_total /= (iter+1)
                print("validation f1:",pre_accuracy_total) 
                v_mse.append(pre_accuracy_total)
                net.train()
                #if train_accuracy > train_acc and pre_accuracy > test_acc:
                #    continue_train = False
                #    break
            torch.save(net, self.save_path)
            train_mse.append(mse)
            val_mse.append(v_mse)
        return np.array(train_mse).mean(0),np.array(val_mse).mean(0)
        

if __name__ == '__main__':
    path_list = r'../data/bitfinex_2017-01-01_to_2019-01-14_eth_usdt_15m_singal_regular_20190526.json'#,r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_1h_singal_regular_20190226.json',r'data/bitfinex_2017-01-01_to_2019-01-15_ltc_usdt_1h_singal_regular_20190226.json']#,r'data/cccagg_1498676400_to_1550865600_eos_usd_1h_singal_regular_20190226.json',r'data/cccagg_1521208800_to_1550977200_ont_usd_1h_singal_regular_20190226.json']
    #path_list = r'../data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_1h_singal_regular_20190226.json'#,r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_1h_singal_regular_20190226.json',r'data/bitfinex_2017-01-01_to_2019-01-15_ltc_usdt_1h_singal_regular_20190226.json']#,r'data/cccagg_1498676400_to_1550865600_eos_usd_1h_singal_regular_20190226.json',r'data/cccagg_1521208800_to_1550977200_ont_usd_1h_singal_regular_20190226.json']

    #path = r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_15m_singal_regular_20190226.json'
    train_models = []
    val_models = []
    run_time = []
    task = 0
    for input_length,last_length,predict_length in zip([96],[32],[8]):
        train = []
        val = [] 
        running_time = []
        for i in ['LSTM-attention']:
            if i == 'FC':
                pca_path = str(input_length)+r'BTClong_input15PCA.m'
                save_path = str(input_length)+r'BTClong_input15_test.pth'
            else:
                pca_path = str(input_length)+r'BTCStep15PCA.m'
                save_path = str(input_length)+r'BTCStep15_test.pth'
                
            tn = Train_Net(path_list,save_path,pca_path,i,input_length,last_length,predict_length)
            begin = datetime.datetime.now()
            a,b = tn.train()
            end = datetime.datetime.now()
            running_time.append(end-begin)
            train.append(a)
            val.append(b)
            #plt.plot(np.arage(50),a,label=i)
            #plt.plot(np.arange(50),b,label=i+' validation')
        #plt.plot(np.arange(epochnum),train[0],label='FC')
        #plt.plot(np.arange(epochnum),train[1],label='LSTM')
        plt.plot(np.arange(epochnum),train[0],label='LSTM-attention')
        #plt.plot(np.arange(epochnum),train[3],label='Self-attention')
        plt.legend(loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('f1-score')
        plt.title('Training Accuracy for Different Models in Task 1-'+str(task))
        plt.show()
        
        
        #plt.plot(np.arange(epochnum),val[0],label='FC')
        #plt.plot(np.arange(epochnum),val[1],label='LSTM')
        plt.plot(np.arange(epochnum),val[0],label='LSTM-attention')
        #plt.plot(np.arange(epochnum),val[3],label='Self-attention')
        plt.legend(loc='upper left')
        plt.xlabel('epoch')
        plt.ylabel('f1-score')
        plt.title('Validation Accuracy for Different Models in Task 1-'+str(task))
        plt.show()
        train_models.append(train)
        val_models.append(val)
        task += 1
        run_time.append(running_time)
# =============================================================================
#     z = pd.DataFrame(np.array(run_stamp),columns=['FC','LSTM','LSTM-attention','Self-attention'],index = ['1-4','1-5','1-6'])
#     plt.plot(z.index,z.loc[:,'FC'])
#     plt.plot(z.index,z.loc[:,'LSTM'])
#     plt.plot(z.index,z.loc[:,'LSTM-attention'])
#     plt.plot(z.index,z.loc[:,'Self-attention'])
#     plt.scatter(z.index,z.loc[:,'FC'])
#     plt.scatter(z.index,z.loc[:,'LSTM'])
#     plt.scatter(z.index,z.loc[:,'LSTM-attention'])
#     plt.scatter(z.index,z.loc[:,'Self-attention'])
#      
#     plt.legend(loc='upper left')
#     plt.xlabel('task')
#     plt.ylabel('second')
#     plt.title('Running Time for Different Models for Different Task')
#     plt.show()
#     
# =============================================================================
