def token():
    key = ''
    return key

address = ['https://api.marketdata.app/v1/options/expirations/{}/?token={}',
           'https://api.marketdata.app/v1/options/chain/{}/?token={}&expiration={}&side={}',
           'https://api.marketdata.app/v1/stocks/quotes/{}/']
             

import requests
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class NNET(nn.Module):

    def __init__(self, inputs, outputs):
        super(NNET, self).__init__()
        self.layer1 = nn.Linear(inputs, 100)
        self.layer2 = nn.Linear(100, 200)
        self.layer3 = nn.Linear(200, 100)
        self.layer4 = nn.Linear(100, outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x


class Feed:

    def __init__(self, ticker='AAPL'):
        self.ticker = ticker
        self.session = requests.Session()
        self.risk_free_rate = pow(1 + 0.05, 1.0/30.0) - 1

    def stockPrice(self):
        url = address[2].format(self.ticker)
        resp = self.session.get(url).json()
        price = resp['last'][0]
        return price
    
    def optionExpiry(self):
        url = address[0].format(self.ticker, token())
        resp = self.session.get(url).json()
        return resp['expirations'][1]
    
    def optionChain(self, expiration, side='call'):
        price = self.stockPrice()
        url = address[1].format(self.ticker, token(), expiration, side)
        resp = self.session.get(url).json()
        K = resp['strike']
        Bid = resp['bid']
        Ask = resp['ask']
        Mid = resp['mid']
        Iv = resp['iv']
        Delta = resp['delta']
        Gamma = resp['gamma']
        Theta = resp['theta']
        Vega = resp['vega']
        Rho = resp['rho']
        input = []
        output = []
        testBid = []
        testAsk = []
        for I in zip(K, Bid, Ask, Mid, Iv, Delta, Gamma, Theta, Vega, Rho):
            input.append([I[3], price, I[0], self.risk_free_rate])
            testBid.append([I[1], price, I[0], self.risk_free_rate])
            testAsk.append([I[2], price, I[0], self.risk_free_rate])
            output.append(list(I[4:]))
        return input, output, testBid, testAsk, K, Iv, Delta, Gamma, Theta, Vega, Rho
    
fig = plt.figure(figsize=(9, 5))
iv = fig.add_subplot(231)
delta = fig.add_subplot(232)
gamma = fig.add_subplot(233)
theta = fig.add_subplot(234)
vega = fig.add_subplot(235)
rho = fig.add_subplot(236)

feed = Feed()
expiry = feed.optionExpiry()
inputs, outputs, testBids, testAsks, K, IV, DELTA, GAMMA, THETA, VEGA, RHO = feed.optionChain(expiry)
        
IN = [torch.tensor(i, dtype=torch.float32) for i in inputs]
OUT = [torch.tensor(i, dtype=torch.float32) for i in outputs]

TBid = [torch.tensor(i, dtype=torch.float32) for i in testBids]
TAsk = [torch.tensor(i, dtype=torch.float32) for i in testAsks]

IN = torch.stack(IN)
OUT = torch.stack(OUT)

TBid = torch.stack(TBid)
TAsk = torch.stack(TAsk)

lr = 0.0001
epochs = 3000

model = NNET(4, 6)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for i in range(epochs):
    out = model(IN)
    loss = criterion(out, OUT)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Epochs Left: ", epochs - i - 1)

with torch.no_grad():
    test_bids = model(TBid)
    test_asks = model(TAsk)

testBids = test_bids.numpy()
testAsks = test_asks.numpy()

colors = ['red', 'green']
plots = [iv, delta, gamma, theta, vega, rho]

for color, ba in zip(colors, [testBids.T, testAsks.T]):
    for plotter, data, actual in zip(plots, ba, (IV, DELTA, GAMMA, THETA, VEGA, RHO)):
        plotter.plot(K, data, color=color)
        plotter.plot(K, actual, color='black')
        plt.pause(0.01)


plt.show()

