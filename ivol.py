url = 'https://www.optionsprofitcalculator.com/ajax/getOptions?stock={}&reqId=1'

import numpy as np
import requests as rq
import json
import time
import datetime
import matplotlib.pyplot as plt
import warnings

# Filter out runtime warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")


class Tree:

    def __init__(self, S, K, r, q, t, steps=6, optype='call'):
        self.S = S
        self.K = K
        self.r = r
        self.q = q
        self.t = t
        self.N = steps
        self.dt = t/steps
        self.optype = optype
        self.row = 4*steps + 2
        self.col = steps + 1
        self.center = int(self.row / 2 - 1)
        self.tree = [[0 for j in range(self.col)] for i in range(self.row)]

    def params(self, v):
        self.up = np.exp(v*np.sqrt(2.0*self.dt))
        self.down = 1.0/self.up
        self.m = 1.0
        
        A = np.exp((self.r - self.q)*self.dt/2.0)
        B = np.exp(-v*np.sqrt(self.dt/2.0))
        C = np.exp(v*np.sqrt(self.dt/2.0))

        self.pu = pow((A - B)/(C - B), 2)
        self.pd = pow((C - A)/(C - B), 2)
        self.pm = 1.0 - (self.pu + self.pd)

    def optionTree(self):
        self.tree[self.center][0] = self.S
        for j in range(self.col):
            for i in range(1, self.col - j):
                self.tree[self.center - 2*i][i + j] = self.tree[self.center - 2*(i-1)][i-1+j]*self.up
                self.tree[self.center + 2*i][i + j] = self.tree[self.center + 2*(i-1)][i-1+j]*self.down
                self.tree[self.center][i +j] = self.tree[self.center][i - 1 + j]*self.m    

        for i in range(self.row):
            if i % 2 != 0:
                if self.optype == 'call':
                    self.tree[i][-1] = np.max([self.tree[i - 1][-1] - self.K, 0.0])
                else:
                    self.tree[i][-1] = np.max([self.K - self.tree[i - 1][-1], 0.0])

        inc = 2
        for j in range(2, self.col+1):
            for i in range(inc, self.row - inc):
                if i % 2 != 0:
                    A = self.tree[i - 2][-j+1]
                    B = self.tree[i][-j+1]
                    C = self.tree[i + 2][-j+1]
                    cash = self.pu*A + self.pm*B + self.pd*C
                    cash = np.exp(-self.r*self.dt)*cash
                    if np.isnan(cash):
                        return 0
                    if self.optype == 'call':
                        self.tree[i][-j] = np.max([self.tree[i - 1][-j] - self.K, cash])
                    else:
                        self.tree[i][-j] = np.max([self.K - self.tree[i - 1][-j], cash])
            inc += 2
        
        return self.tree[self.center + 1][0]

    def optionVega(self, v):
        dV = 0.01
        self.params(v+dV)
        c1 = self.optionTree()
        self.params(v-dV)
        c0 = self.optionTree()
        vega = (c1 - c0)/(2.0*dV)
        return vega

    def impliedVol(self, option_price):
        v0 = 0.11
        v1 = 0.88
        while True:
            self.params(v0)
            v1 = v0 - (self.optionTree() - option_price)/self.optionVega(v0)
            if abs(v1 - v0) < 0.0001:
                break
            v0 = v1
        if v1 < 0:
            return np.nan
        return v1

class Data:

    def __init__(self, ticker='SPY'):
        self.stock_price = 520.84
        self.div_yield = 0.0129
        self.risk_free = 0.0525

        headers = {
            'User-Agent': 'MacOS',
            'Content-Type': 'application/json'
        }
        self.dataset = rq.get(url.format(ticker), headers=headers).json()
    
    def cleanup(self, bounds=60):
        the_date = '2024-05-17'
        calls = self.dataset['options'][the_date]['c']
        puts = self.dataset['options'][the_date]['p']
        call_options = []
        put_options = []
        for i, j in calls.items():
            if float(i) >= self.stock_price - bounds and float(i) <= self.stock_price + bounds:
                call_options.append([float(i), float(j['l'])])
        for i, j in puts.items():
            if float(i) >= self.stock_price - bounds and float(i) <= self.stock_price + bounds:
                put_options.append([float(i), float(j['l'])])
        the_time = time.mktime(datetime.datetime.strptime(the_date, '%Y-%m-%d').timetuple())
        delta = (int(the_time) - int(time.time()))/(60*60*24*365)
        
        # S, K, rf, div, t, op
        return self.stock_price, self.risk_free, self.div_yield, delta, call_options, put_options

data = Data()
S, r, q, t, calls, puts = data.cleanup()

fig = plt.figure()
ax = fig.add_subplot(111)

call_x, call_y = [], []
for K, opPrice in calls:
    try:
        tree = Tree(S, K, r, q, t, optype='call', steps=50)
        iv = tree.impliedVol(opPrice)
        if np.isnan(iv) == False:
            ax.scatter(K, iv, color='limegreen')
            plt.pause(0.001)
    except:
        pass

put_x, put_y = [], []
for K, opPrice in puts:
    try:
        tree = Tree(S, K, r, q, t, optype='put', steps=50)
        iv = tree.impliedVol(opPrice)
        if np.isnan(iv) == False:
            ax.scatter(K, iv, color='red')
            plt.pause(0.001)
    except:
        pass


plt.show()
    



