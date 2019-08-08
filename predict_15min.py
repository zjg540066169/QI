# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:47:42 2019

@author: wang
"""

from attentionRNN import BahdanauAttnDecoderRNN,EncoderRNN
from data_download import Data_download
from sendEmail import sendMail
import time,datetime,pytz,requests
import pandas as pd
import torch
import torch.nn.functional as F
NETWORK = 'GRUA'#FC LSTM Attention
coin = 'BTC'
input_length = 96
pcaPath = str(input_length)+r'BTCStep15PCA.m'

MulEncoding = 13
ERROR_RATE = 0 
ERROR_RATE_Vol = 0.2

predict_length = 8
last_length = 32
pastSecond = 15*60*200
plateform = 'HUOBI'
to = "USDT"
timespan = '1M'
aggregate = 15
send_time = -2
long_input = False
Timeblock = 0
nnPath = str(input_length)+r'BTCStep15_test.pth'
if NETWORK == 'FC':
    pcaPath = coin+NETWORK+'15PCA.m'
    nnPath = coin+NETWORK+'15.pth'
    long_input = True

def stampToTime(timeStamp):
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
    return otherStyleTime    

    

def sendMessage(string,coin = coin,to = "USDT"):
    message = {}
    message["time"] = int(time.time())
    message["type"] = "alarm"
    message["degree"] = 1
    message["coin_from"] = coin
    message["coin_to"] = to
    message["signal"] = "High_Low_Predict"
    message["message"] = string
    return message


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
        decoder_output,decoder_hidden = self.decode(torch.zeros(word_input.size()[0],1,self.output_size),encoder_hidden,encoder_output)
        return decoder_output


class Test_Net(object):
    def __init__(self,save_path):
        self.net = torch.load(save_path)
        print(self.net)
        
    def predict(self,predict_X):
        self.net.eval()
        inputs= torch.autograd.Variable(torch.Tensor(predict_X))
        print('inputs:',inputs.size())
        outputs = self.net(inputs)
        return outputs



if __name__=='__main__':
    NN = Test_Net(nnPath)
    while True:
        timeNow = datetime.datetime.now()
        timeNow = timeNow.astimezone(pytz.timezone('Asia/Shanghai'))
        if (timeNow.minute + send_time) % 1 == 0:

            dd = Data_download(pastSecond*2,int(time.time()),plateform,coin,to,timespan,aggregate)
            data = pd.DataFrame(dd.downloadData()['rows'])
            a = dd.predict_data_process(data,pcaPath,MulEncoding,ERROR_RATE, ERROR_RATE_Vol, input_length,predict_length,last_length,long_input = long_input )
# =============================================================================
#             if dd.predict_data_process(data,pcaPath,MulEncoding,ERROR_RATE, ERROR_RATE_Vol, input_length,predict_length,last_length,long_input = long_input ) ==0:
#                 time.sleep(60)
#                 continue
#             
# =============================================================================
            predict_X = dd.get_predictX()
            print('predict_X.size',predict_X.size)
            
            
            predict_prob = NN.predict(predict_X)
            print('predict_prob',predict_prob.shape)
            predict_prob = F.softmax(predict_prob)
                #print(predict_high)
            predict_result = torch.max(predict_prob.data,1)[1].numpy()
            predict_prob = predict_prob
            predict_result = predict_result
            print(predict_prob,predict_result)
            string = '【'+NETWORK+'】 15M '+coin+'/'+to+' 现价 '+str(data.loc[data.index[-1],'close'])+'\n'

            string += '未来8K'
            if predict_result == 1:
                string += '【破DC高轨】概率'+str('%.2f%%' %(100*predict_prob[1]))
            elif predict_result == 2:
                string += '【破DC低轨】概率 '+str('%.2f%%' %(100*predict_prob[2]))
            elif predict_result == 0:
                string += '【不破高低】概率 '+str('%.2f%%' %(100*predict_prob[0]))
            string += '\n'
            string += '信号备注：'+stampToTime(timeNow.timestamp()+Timeblock)[6:9]+stampToTime(timeNow.timestamp()+Timeblock)[10:18]+'\n13均线下拐头'

            
            
            
            
            
            string = string.replace("--","-")
            print(string)
            #response = requests.post('http://106.75.90.217:30804/openapi/coin/signal/report',data=sendMessage(string))
                    # 通过get请求返回的文本值
            #print(response.text)
            time.sleep(300)

            
