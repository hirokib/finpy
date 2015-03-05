# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 13:18:22 2015

@author: hiroki
"""
import pandas.io.data as pd
import numpy as np
import datetime as dt 
import math
import random
import matplotlib.pyplot as plt
import copy as cp
import scipy.stats as sci
plt.ion

def get_contract(stock_symbols,month1=1,date1=2,year1=2014,month2=1,date2=2,year2=2015):  
        start = dt.datetime(year1,month1,date1)
        end = dt.datetime(year2,month2,date2)
        data = pd.DataReader(stock_symbols,'yahoo',start,end)
        data = data.drop([u'Open', u'High', u'Low', u'Close'])
        return data    



class stock_data(pd.Panel):
    
    def change_range(self,delta):
        n = pd.Panel.copy(self)
        n = n.ix[:,-delta:,:]
        return n
    
    def dcgr(self):
        self['dcgr'] = self['Adj Close'].apply(np.log) - self['Adj Close'].apply(np.log).shift(1)
        
    def norm_dcgr(self):
        self.avg = np.mean(self["dcgr"])
        self.std = np.std(self["dcgr"])
        self['norm_dcgr'] = self['dcgr']
        #normalize = lambda x: (x-self.avg)/self.std
        self['norm_dcgr'] = (self['dcgr']-self.avg)/self.std


    def market_corr(self,x):
       return self.minor_xs(ETF)["norm_dcgr"].corr(self.minor_xs(x)["norm_dcgr"])
    
    
    def norm_dist(self,x,mean,std):
        return 1/(math.sqrt(2*math.pi*math.pow(std,2)))*math.pow(math.e,((-(x-mean)**2)/(2*std**2)))
    
    def cum_dist(self,x,mean,std):
        'Cumulative distribution function for the standard normal distribution'
        return (1.0 + math.erf( (x-mean) / (std*math.sqrt(2.0)))) / 2.0
    
    def sharpe_ratio(self,mean,std,rf = 0.0):
        return (mean-rf)/std
    
    def norm_base(self,call_price,strike_price,mean,std):
        lgr = math.log(call_price/strike_price) 
        return self.norm_dist(lgr,mean,std)
    
    def cum_base(self,call_price,strike_price,mean,std):
        lgr = math.log(call_price/strike_price)
        return self.cum_dist(lgr,mean,std)
        
    def stk_mean(self,x):
        return np.mean(self.minor_xs(x)["dcgr"])
    
    def stk_std(self,x):
        return np.std(self.minor_xs(x)["dcgr"],dtype=np.float64)    
    
    def beta(self,x):
        sd1 = self.stk_std(x)   
        sd2 = self.stk_std(ETF)
        ratio = sd1/sd2
        return self.market_corr(x)*ratio
    
    def min(self,stock,item):
        return np.min(self.minor_xs(stock)[item])
        
    def max(self,stock,item):
        return np.max(self.minor_xs(stock)[item])
        
    def sppc_norm(self,stock,stock_price,strike_price,days_to_exp):
        #log growth rate price to strike price 
        dcgr = self.stk_mean(stock)
        std = self.stk_std(stock)
        lgrps = math.log(strike_price/stock_price)
        tasd = math.sqrt(days_to_exp)*std
        prob = 1-sci.norm.cdf(lgrps,loc=0,scale=tasd)
        return prob
        
    def sppc_norm_a(self,stock,stock_price,strike_price,days_to_exp):
        #log growth rate price to strike price 
        dcgr = self.stk_mean(stock)
        std = self.stk_std(stock)
        lgrps = math.log(strike_price/(stock_price*math.e**(dcgr*days_to_exp)))
        tasd = math.sqrt(days_to_exp)*std
        prob = 1-sci.norm.cdf(lgrps,loc=dcgr,scale=tasd)
        return prob   
        
    def plot_stock(self,stock):
        self.minor_xs(stock)['norm_dcgr'].plot()
        self.minor_xs(stock)['Adj Close'].plot()
        
#    def sppc_pois(self,stock,stock_price,strike_price,days_to_exp):
#        #log growth rate price to strike price 
#        dcgr = self.stk_mean(stock)
#        std = self.stk_std(stock)
#        lgrps = math.log(strike_price/(stock_price*math.e**(dcgr*days_to_exp)))
#        tasd = math.sqrt(days_to_exp)*std
#        prob = sci.poisson.sf(lgrps,mu=dcgr)
#        return prob   
        
 

def BSoption(SP,KP,DV,IRR,DTM,PUT = False):
    _d1num = d1num(SP,KP,DV,IRR,DTM)
    _duvol = duvol(DV,DTM)
    if(PUT):
        _ND1 = sci.norm.cdf(-(_d1num/_duvol))
        _ND2 = sci.norm.cdf(-(_d1num/_duvol-_duvol))
        return -SP*_ND1+KP*math.exp(-IRR*DTM/365)*_ND2
    else:
        _ND1 = sci.norm.cdf(_d1num/_duvol)
        _ND2 = sci.norm.cdf(_d1num/_duvol - _duvol)
        return SP*_ND1-KP*math.exp(-IRR*(DTM/365))*_ND2
    


def d1num(SP,KP,IRR,DV,DTM):
    return math.log(SP/KP)+((IRR/365)+(DV**2)/2)*DTM
    
def duvol(DV,DTM):
    return DV*DTM**0.5
    
def deltad1(num,duv):
    return sci.norm.cdf(num/duv)
    
    
def deltad2(num,duv):
    return sci.norm.cdf((num/duv)-duv)

def ringer(IRR,DTM):
    return math.exp(-IRR*DTM/365)    


#def IDVcalc(SP,KP,IRR,DV,DTM,CIPD):
#    CIPD = CIPD+0.00001
#    num = d1num(SP,KP,IRR,DV,DTM)
#    duv = duvol(CIPD,DTM)
#    DND1 = deltad1(num,duv)
#    ring = ringer(IRR,DTM)
#    DND2 = deltad2(num,duv)    
#    tempcall = SP*DND1-KP*ring*DND2
#    while(tempcall < 215):
#        CIPD = CIPD+0.00001
#        num = d1num(SP,KP,IRR,DV,DTM)
#        duv = duvol(CIPD,DTM)
#        DND1 = deltad1(num,duv)
#        ring = ringer(IRR,DTM)
#        DND2 = deltad2(num,duv)    
#        tempcall = SP*DND1-KP*ring*DND2



def IDVcalc(SP,KP,IRR,DTM,CP,CIPD = 0.0001):
    CIPD += 0.0001    
    DeNom = math.log(SP/KP)+((IRR/365)+(CIPD**2)/2)*DTM
    DuVol = CIPD*DTM**0.5
    DND1 = sci.norm.cdf(DeNom/DuVol)
    DND2 = sci.norm.cdf(DeNom/DuVol - DuVol)
    Ringer = math.exp(-IRR*DTM/365)
    tempcall = SP*DND1-KP*Ringer*DND2
    while( tempcall < CP ):
        if( tempcall >= CP):
            return CIPD
        print CIPD, tempcall
        CIPD += 0.000001
        DeNom = math.log(SP/KP)+((IRR/365)+(CIPD**2)/2)*DTM
        DuVol = CIPD*DTM**0.5
        DND1 = sci.norm.cdf(DeNom/DuVol)
        DND2 = sci.norm.cdf(DeNom/DuVol - DuVol)
        Ringer = math.exp(-IRR*DTM/365)
        tempcall = SP*DND1-KP*Ringer*DND2
        
print 'test'
        
print IDVcalc(131.680,134.00,0.0100,22,0.460)
        

#def monte_step(dt, prices, c0, c1, noises):
#    return prices * np.exp(c0 * dt + c1 * noises)
#
#def monte_carlo_norm(paths, dt, interest, volatility):
#    c0 = interest - 0.5 * volatility ** 2
#    c1 = volatility * np.sqrt(dt)
# 
#    for j in xrange(1, paths.shape[1]): # for all trials
#        prices = paths[:, j - 1]
#        # generate normally distributed random number
#        noises = np.random.normal(0., 1., prices.size)
#        # calculate the next batch of prices for all trials
#        paths[:, j] = monte_step(dt, prices, c0, c1, noises)
    
    
        
        
        
        
        