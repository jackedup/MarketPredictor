#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###
###This file pulls from yahoo finance minute data for last 30 days
###and puts into MarketData folder in daily increments
###
from pandas_datareader import data as pdr
import datetime as dt
import yfinance as yf
import os

yf.pdr_override()
today = dt.date.today()


symbols =['AAPL', 'SPY', 'NVDA', 'SPCE', 'CPRX', 'BABA', 'AMZN',
         'GOOGL', 'MSFT', 'PTON','NKE','PYPL','LULU','TSM']

###makes path for every symbol if it does not exist

for i in symbols:
    if not os.path.exists('MarketData/'+i):
        os.makedirs('MarketData/'+i)
        
###pulls data from yahoo finance from last 30 days that does not exist
for i in symbols:
    day = dt.date.today()
    yesterday = day - dt.timedelta(days=1)
    for j in range(29):
        if not os.path.exists('MarketData/'+i+'/'+yesterday.strftime("%Y-%m-%d")+'.csv'):
            if not yesterday.weekday() > 4 :
                pdr.DataReader(i,yesterday.strftime("%Y-%m-%d"),day.strftime("%Y-%m-%d"), interval="1m").to_csv('MarketData/'+i+'/'+yesterday.strftime("%Y-%m-%d")+'.csv')
        day = yesterday
        yesterday = day - dt.timedelta(days=1)

print ("done")

