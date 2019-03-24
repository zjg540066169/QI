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
BATCH_SIZE = 32
NETWORK = 'FC'#FC LSTM Attention
class LSTMClassifier(torch.nn.Module):
    def __init__(self,input_size,input_length = 96):
        super(LSTMClassifier, self).__init__()
        self.hidden = 300
        self.input_size = input_size[2]
        self.lstm = torch.nn.LSTM(input_size[2], self.hidden,batch_first=True)

        self.ln2 = torch.nn.Linear(self.hidden,self.hidden)
        self.bn1 = torch.nn.BatchNorm1d(input_length)
        self.out1 = torch.nn.Linear(self.hidden,3)

        self.bn2 = torch.nn.BatchNorm1d(input_length)
        self.out2 = torch.nn.Linear(self.hidden,3)

        self.bn3 = torch.nn.BatchNorm1d(input_length)
        self.out3 = torch.nn.Linear(self.hidden,3)
        
        self.bn4 = torch.nn.BatchNorm1d(input_length)
        self.out4 = torch.nn.Linear(self.hidden,3)

    
    def forward(self,x):
        
        x = self.lstm(x)[0]
        x = F.sigmoid(x)
        x = self.ln2(x)
        x = F.relu(x)
        #print(x1.size())
        x1 = self.bn1(x)#BatchNormal
        x1 = self.out1(x1)
        x1 = F.dropout(x1, 0.1, self.training)[:,-1,:].view(-1,3)
        
        x2 = self.bn2(x)#BatchNormal
        x2 = self.out2(x2)
        x2 = F.dropout(x2, 0.1, self.training)[:,-1,:].view(-1,3)

        x3 = self.bn3(x)#BatchNormal
        x3 = self.out3(x3)
        x3 = F.dropout(x3, 0.1, self.training)[:,-1,:].view(-1,3)
        
        x4 = self.bn4(x)#BatchNormal
        x4 = self.out4(x4)
        x4 = F.dropout(x4, 0.1, self.training)[:,-1,:].view(-1,3)
        return [x1,x2,x3,x4]
    
class FCClassifier(torch.nn.Module):
    def __init__(self,input_size,input_length = 96):
        super(FCClassifier, self).__init__()
        self.hidden = 300
        self.input_size = input_size[1]
        self.fc = torch.nn.Linear(input_size[1], self.hidden*2)

        self.ln2 = torch.nn.Linear(self.hidden*2,self.hidden)
        
        self.bn1 = torch.nn.BatchNorm1d(self.hidden)
        self.out1 = torch.nn.Linear(self.hidden,3)

        self.bn2 = torch.nn.BatchNorm1d(self.hidden)
        self.out2 = torch.nn.Linear(self.hidden,3)

        self.bn3 = torch.nn.BatchNorm1d(self.hidden)
        self.out3 = torch.nn.Linear(self.hidden,3)
        
        self.bn4 = torch.nn.BatchNorm1d(self.hidden)
        self.out4 = torch.nn.Linear(self.hidden,3)
        
    def forward(self,x): 
        x = self.fc(x)
        x = F.sigmoid(x)
        x = self.ln2(x)
        x = F.relu(x)
        #print(x1.size())
        x1 = self.bn1(x)#BatchNormal
        x1 = self.out1(x1)
        x1 = F.dropout(x1, 0.1, self.training).view(-1,3)
        
        x2 = self.bn2(x)#BatchNormal
        x2 = self.out2(x2)
        x2 = F.dropout(x2, 0.1, self.training).view(-1,3)

        x3 = self.bn3(x)#BatchNormal
        x3 = self.out3(x3)
        x3 = F.dropout(x3, 0.1, self.training).view(-1,3)
        
        x4 = self.bn4(x)#BatchNormal
        x4 = self.out4(x4)
        x4 = F.dropout(x4, 0.1, self.training).view(-1,3)
        return [x1,x2,x3,x4]
    

class MulTaskLoss(torch.nn.Module):
    def __init__(self):
        super(MulTaskLoss, self).__init__()
        
    def forward(self,outputs,labels):
        loss0 = torch.nn.CrossEntropyLoss()(outputs[0],labels[:,0])        
        loss1 = torch.nn.CrossEntropyLoss()(outputs[1],labels[:,1])        
        loss2 = torch.nn.CrossEntropyLoss()(outputs[2],labels[:,2])        
        loss3 = torch.nn.CrossEntropyLoss()(outputs[3],labels[:,3])        

        return loss0+loss1+loss2+loss3
    
class Train_Net(object):
    def __init__(self,path,save_path,pca_path):
        if NETWORK == 'FC':
            long_input = True
        else:
            long_input = False
        data_proces = data_process(path,pca_path,long_input = long_input)
        data_proces.process()
        self.X,self.Y = data_proces.get_X(),data_proces.get_Y()
        self.save_path = save_path
        
    def train(self,train_acc = 0.85,test_acc = 0.85):
        #return self.X,self.Y
        continue_train = True
        while continue_train:
            train_x, val_x, train_y, val_y = train_test_split(self.X, self.Y, test_size=0.33)
            if NETWORK == 'FC':
                net = FCClassifier(train_x.shape)
            else:
                net = LSTMClassifier(train_x.shape)
            train_dataset = TensorDataset(torch.Tensor(train_x),torch.LongTensor(train_y))
            train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
            test_dataset = TensorDataset(torch.Tensor(val_x),torch.LongTensor(val_y))
            test_dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

            
            cost = MulTaskLoss()
            optimizer = optim.Adam(net.parameters(), lr=0.001)
            mse  = []
            epochnum = 240
            for epoch in range(epochnum):
                train_accuracy = 0
                for iter,(inputs,labels) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = cost(outputs,labels)
                    for i in range(4):
                        train_accuracy += f1_score(labels[:,i].numpy(), torch.max(outputs[i].data,1)[1].numpy(), average='micro')
                    train_accuracy /= 4
                    
                    loss.backward()
                    optimizer.step()
                train_accuracy /= (iter+1)

                print("f1:",train_accuracy,end = "    ")
                mse.append(train_accuracy)
            
                net.eval()   
                pre_accuracy = 0
                for iter, (inputs,true_y) in enumerate(test_dataloader):
                    predict_y = net(inputs)
                #print(pd.Series(torch.max(predict_y.data,1)[1].numpy()).value_counts()[1]/pd.Series(val_y).value_counts()[1])
                    for i in range(4):
                        pre_accuracy += f1_score(true_y[:,i].numpy(), torch.max(predict_y[i].data,1)[1].numpy(), average='micro')
                    pre_accuracy /= 4
                pre_accuracy /= iter
                print("validation f1:",pre_accuracy)    
                net.train()
                if train_accuracy > train_acc and pre_accuracy > test_acc:
                    continue_train = False
                    break
        torch.save(net, self.save_path)
        

if __name__ == '__main__':
    path_list = r'../data/bitfinex_2017-01-01_to_2019-01-15_ltc_usdt_1h_singal_regular_20190226.json'#,r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_1h_singal_regular_20190226.json',r'data/bitfinex_2017-01-01_to_2019-01-15_ltc_usdt_1h_singal_regular_20190226.json']#,r'data/cccagg_1498676400_to_1550865600_eos_usd_1h_singal_regular_20190226.json',r'data/cccagg_1521208800_to_1550977200_ont_usd_1h_singal_regular_20190226.json']
    #path = r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_15m_singal_regular_20190226.json'
    if NETWORK == 'FC':
        pca_path = r'LTCFC15PCA.m'
        save_path = r'LTCFC15.pth'

    tn = Train_Net(path_list,save_path,pca_path)
    
    a = tn.train()