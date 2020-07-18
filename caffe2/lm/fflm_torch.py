#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 21:51:29 2020

@author: lacozhang@gmail.com
"""

import torch
from torch import nn
from torch import optim
from lm_reader import LanguageModelData
import random
import logging

logger = logging.getLogger(__name__)


class FFLM(nn.Module):
    
    def __init__(self, embed_size, vocab_size, hidden_size, hist_size, drop_out):
        super(FFLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fflm = nn.Sequential(
            nn.Linear(hist_size*embed_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size, vocab_size)
        )
        
    def forward(self, words):
        embedding = self.embedding(words)
        feats = embedding.view(embedding.size(0), -1)
        logits = self.fflm(feats)
        
        return logits
    
HIST_SIZE = 2
BATCH_SIZE = 32

lmData = LanguageModelData()
trainData = lmData.getTrainData(loglinear=True, ngram=HIST_SIZE)
validData = lmData.getValidData(loglinear=True, ngram=HIST_SIZE)
evalData = lmData.getEvalData(loglinear=True, ngram=HIST_SIZE)

model = FFLM(32, lmData.VocabSize, 128, HIST_SIZE, 0.2)
loss = nn.NLLLoss()

optimizer = optim.Adagrad(model.parameters(), lr=1e-4)

for iter_cnt in range(2):
    random.shuffle(trainData)
    
    model.train()
    
    for idx in range(0, len(trainData), BATCH_SIZE):
        start_idx = idx
        end_idx = min(start_idx + BATCH_SIZE, len(trainData))
        
        inputs = torch.LongTensor(
            [item[0] for item in trainData[start_idx:end_idx]]
        )
        labels = torch.LongTensor(
            [item[1] for item in trainData[start_idx:end_idx]]
        )
        
        optimizer.zero_grad()
        loss_value = loss(model(inputs), labels)
        
        loss_value.backward()
        
        optimizer.step()
        
        if idx > 0 and idx % 100 == 0:
            print("average loss value: {}".format(
                loss_value
            ))
        
        
        
        
    




        