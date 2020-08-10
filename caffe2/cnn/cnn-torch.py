#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 10:26:53 2020

@author: lacozhang@gmail.com
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
import class_data as data
from torch import optim
import torch.nn.functional as F


class CNNText(nn.Module):

    def __init__(
            self, 
            vocab_size, 
            embed_size, 
            nclasses, 
            num_filters, 
            window_size
    ):
        super(CNNText, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        self.conv = nn.Conv1d(
            embed_size, 
            num_filters, 
            kernel_size=window_size, 
            stride=1, 
            padding=0,
            dilation=1,
            groups=1,
            bias=True
        )
        self.relu = nn.ReLU()
        self.projection = nn.Linear(num_filters, nclasses)
        
        
    def forward(self, words):
        features = self.embedding(words).transpose(1,2)
        return self.projection(self.relu(self.conv(features).max(dim=2)[0]))

EMBED_SIZE = 64
NUM_FILTERS = 32
WIN_SIZE = 3
BATCH_SIZE = 2
model = CNNText(
    vocab_size=data.Metadata.vocab_size, 
    embed_size=EMBED_SIZE, 
    nclasses=data.Metadata.label_size,
    num_filters=NUM_FILTERS, 
    window_size=WIN_SIZE
)
optimizer = optim.Adagrad(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()
train = data.getTrainIterator(BATCH_SIZE)
dev = data.getDevIterator(BATCH_SIZE)
test = data.getTestIterator(BATCH_SIZE)


pad_idx = data.Metadata.word2id["<pad>"]

for epoch_idx in range(10):

    train.init_epoch()
    ex_cnt = 0
    model.train()
    print("epoch: {}".format(
        epoch_idx
    ))
    for item in train:
        ex_cnt += 1
        
        if item.text.size(1) < WIN_SIZE:
            text = F.pad(item.text, (0, WIN_SIZE - item.text.size(1)), mode='constant', value=pad_idx)
        else:
            text = item.text
        loss = criterion(model(text), item.label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ex_cnt % 1000 == 0:
            print("loss value: {}".format(
                loss.item()
            ))
