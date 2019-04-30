# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:22:57 2019

@author: wang
"""
import json
import pandas as pd

class Read_data(object):
    def __init__(self,path_list):
        if type(path_list) is str:    
            self.path_list = [path_list]
        else:
            self.path_list = path_list
            
        
    def decode(self):
        data = []
        for i in self.path_list:
            with open(i,'r') as f:
                text = f.readlines()[0]
                dic = json.loads(text)
                data.extend(dic)
        return data
        
        
if __name__ == '__main__':
    path_list = [r'../data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_15m_singal_regular_20190226.json']#,r'data/bitfinex_2017-01-01_to_2019-01-15_btc_usdt_1h_singal_regular_20190226.json',r'data/bitfinex_2017-01-01_to_2019-01-15_ltc_usdt_1h_singal_regular_20190226.json',r'data/cccagg_1498676400_to_1550865600_eos_usd_1h_singal_regular_20190226.json',r'data/cccagg_1521208800_to_1550977200_ont_usd_1h_singal_regular_20190226.json']

    data = Read_data(path_list).decode()
    b = pd.DataFrame(data)
    b.index = b.time
    
    #b = b.loc[:,['open','high','low','close','volume']]
    #b.to_csv('btc_15min.csv')
    