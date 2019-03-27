# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:47:42 2019

@author: wang
"""

from data_download import Data_download
from test_net import Test_Net
from train_net import LSTMClassifier,MulTaskLoss
from sendEmail import sendMail
import time,datetime,pytz,requests
import pandas as pd
NETWORK = 'FC'#FC LSTM Attention
coin = 'LTC'
pcaPath = coin+NETWORK+'60PCA.m'
nnPath = coin+NETWORK+'60.pth'
MulEncoding = 13
ERROR_RATE = 0 
ERROR_RATE_Vol = 0.2
input_length = 96
predict_length = 4
last_length = 16
pastSecond = 60*60*200
plateform = 'HUOBI'
to = "USDT"
timespan = '1M'
aggregate = 60
send_time = -2
long_input = False
if NETWORK == 'FC':
    pcaPath = coin+NETWORK+'60PCA.m'
    nnPath = coin+NETWORK+'60.pth'
    long_input = True


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

if __name__=='__main__':
    NN = Test_Net(nnPath)
    while True:
        timeNow = datetime.datetime.now()
        timeNow = timeNow.astimezone(pytz.timezone('Asia/Shanghai'))
        if (timeNow.minute + send_time) % 60 == 0:

            dd = Data_download(pastSecond,int(time.time()),plateform,coin,to,timespan,aggregate)
            data = pd.DataFrame(dd.downloadData()['rows'])
            if dd.predict_data_process(data,pcaPath,MulEncoding,ERROR_RATE, ERROR_RATE_Vol, input_length,predict_length,last_length,long_input = long_input) ==0:
                time.sleep(60)
                continue
            predict_X = dd.get_predictX()
            predict_result,predict_prob = NN.predict(predict_X)
            string = NETWORK+' '+coin+'/'+to+' 现价 '+str(data.loc[data.index[-1],'close'])+'\n60M 五均线出现'
            if data.loc[data.index[-2],'021'] == 1:
                string += '上拐头'
            elif data.loc[data.index[-2],'021'] == -1:
                string += '下拐头\n'
            string += '\n60M 未来24K(80%)'
            if predict_result[1] == 0:
                string += '上涨概率 '+str('%.2f%%' %(100*predict_prob[1][0][0]))
            elif predict_result[1] == 1:
                string += '下跌概率 '+str('%.2f%%' %(100*predict_prob[1][0][1]))
            elif predict_result[1] == 2:
                string += '横盘概率 '+str('%.2f%%' %(100*predict_prob[1][0][2]))
            string += '\n'
            string += '60M 级别未来8K(100%)'
            if predict_result[0] == 0:
                string += '上涨概率 '+str('%.2f%%' %(100*predict_prob[0][0][0]))
            elif predict_result[1] == 1:
                string += '下跌概率 '+str('%.2f%%' %(100*predict_prob[0][0][1]))
            elif predict_result[1] == 2:
                string += '横盘概率 '+str('%.2f%%' %(100*predict_prob[0][0][2]))
            #string += '15M 级别未来8根均线100%占比\n'
            
            
            string += '未来4小时破最高'+str(data.loc[data.index[-2],"022_HH"])+"的概率是"+str('%.2f%%' %(100 * predict_prob[2][0][0]))+'\n'
            string += '未来4小时破次高'+str(data.loc[data.index[-2],"022_fb1"])+"的概率是"+str('%.2f%%' %(100 * predict_prob[2][0][1]))+'\n'
            string += '未来4小时破最低'+str(data.loc[data.index[-2],"022_LL"])+"的概率是"+str('%.2f%%' %(100 * predict_prob[3][0][0]))+'\n'
            string += '未来4小时破次低'+str(data.loc[data.index[-2],"022_fb4"])+"的概率是"+str('%.2f%%' %(100 * predict_prob[3][0][1]))+'\n'
            
            
            print(string)
            string = string.replace("--","-")
            print(string)
            response = requests.post('http://order.shareted.com/openapi/coin/signal/report',data=sendMessage(string))
                    # 通过get请求返回的文本值
            print(response.text)
            time.sleep(300)

            
