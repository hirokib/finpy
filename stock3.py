# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 23:55:28 2015

@author: Hiroki
"""

import pandas.io.data as pd
import numpy as np
import datetime as dt 
import math
import matplotlib.pyplot as plt
import copy as cp
import scipy.stats as sci
plt.ion()
from stock_func import *


ETF = "SPY"

stock_symbols =  [ETF,"AMD",'AAPL','MSFT','TSLA']
start = dt.datetime(2014, 2, 7)
end = dt.datetime(2015, 2, 6)


#s = data(stock_symbols,'yahoo')
s = get_contract(stock_symbols,1,2,2014,1,2,2015)
q = stock_data(s.values,s.items,s.major_axis,s.minor_axis)
q.dcgr()
q.norm_dcgr()

s30 = q.change_range(30)
s60 = q.change_range(60)

        
def monte_carlo(stock_price,alpha,std,days):
    path = []    
    path.append(stock_price)
    for day in range(days-1):
        path.append(path[-1]*math.e**(alpha+std*sci.norm.ppf(random.random())))
    return path



test = monte_carlo(3.03,0.0042,0.0446,30)
s30.minor_xs('AMD')['monte'] = test

test1 = pd.Series(test,index = s30.major_axis)
    

#print s30.sppc_pois("AMD",3.03,3.50,39)

























        