#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:23:47 2018

@author: edwinzhang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import random

def loadTrainData():
    categories = ['alt.atheism', 'talk.religion.misc',
                  'comp.graphics', 'sci.space']
    newsgroup_train = fetch_20newsgroups(subset='train', 
                                         categories=categories)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(newsgroup_train.data)
    return (vectorizer, vectors, newsgroup_train.target)


(featurizer, X, Y) = loadTrainData()

n_samples = int(X.shape[0])
feat_dim = int(X.shape[1])
label_dim = int(len(set(Y)))

class BowClassifier(nn.Module):
    
    def __init__(self, feat_dim, label_dim):
        super(BowClassifier, self).__init__()
        self.featDim = feat_dim
        self.labelDim = label_dim
        self.sampleSize = torch.Size([1, feat_dim])
        self.linear = nn.Linear(feat_dim, label_dim, bias=True)
    
    def forward(self, doc):
        return F.log_softmax(self.linear(doc), dim=1)
    
    def makeTorchTensor(self, x):
        u = x.tocoo()
        i = torch.LongTensor([u.row, u.col])
        v = torch.FloatTensor(u.data)
        return torch.sparse.FloatTensor(i, v, self.sampleSize).to_dense()

    

model = BowClassifier(feat_dim, label_dim)
optimizer = optim.SGD(model.parameters(), lr=1e-2)
nll_loss = nn.NLLLoss()
torch.manual_seed(0)

for batch in range(10):
    idxes = list(range(n_samples))
    random.shuffle(idxes)
    for idx in idxes:
        optimizer.zero_grad()
        label = torch.LongTensor([Y[idx]])
        loss = nll_loss(
                model.forward(model.makeTorchTensor(X[idx])), 
                label)
    
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        count = 0
        for idx in range(n_samples):
            predict = model.forward(model.makeTorchTensor(X[idx]))
            yp = torch.argmax(predict, dim=1)
            if (yp == torch.LongTensor([Y[idx]])):
                count = count + 1
        print(count*1.0/n_samples)