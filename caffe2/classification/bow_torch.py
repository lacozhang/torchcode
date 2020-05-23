#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 16:19:53 2020

@author: yuzhang
"""


import torch
from torch import nn, optim
import logging
import random
from io_utils import (
    inputFilePath,
    inputDevPath,
    IOUtils
)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CBOW(nn.Module):
    
    def __init__(self, vocabSize, nclass, embed_size):
        super(CBOW, self).__init__()
        self.vocab = vocabSize
        self.nclass = nclass
        self.embed_size = embed_size
        
        self.embedding = nn.Embedding(self.vocab, self.embed_size)
        self.fc = nn.Linear(self.embed_size, self.nclass, bias=True)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.weight, 0.0)
        
    
    def forward(self, words):
        words_embedding = self.embedding(words)
        embed_sum = torch.reshape(
            torch.sum(
                words_embedding, 
                dim=[0]
            ), 
            shape=(-1, self.embed_size)
        )
        return self.fc(embed_sum)
    

def modelTrain(model, iterCnts, trainData, devData=None):
    
    lossfunc = nn.CrossEntropyLoss()
    totalLoss = 0.0
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for iterIdx in range(iterCnts):
        logger.info(f'Iter: {iterIdx}')
        random.shuffle(trainData)
        model.train()
        
        for (label, words) in trainData:
            labelTensor = torch.LongTensor(
                [label] if isinstance(label, int) else label
            )
            wordsTensor = torch.LongTensor(
                words if isinstance(words, list) else None
            )
            
            score = model(wordsTensor)
            loss = lossfunc(score, labelTensor)
            totalLoss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        logger.info(f'average loss: {totalLoss/len(trainData)}')
        if devData is not None:
            model.eval()
            with torch.no_grad():
                totalCnt = 0.0
                correct  = 0.0
                
                for (label, words) in devData:
                    predictedLabel = torch.argmax(
                        model(torch.LongTensor(words))
                    ).item()
                    
                    if predictedLabel == label:
                        correct += 1.0
                    
                    totalCnt += 1.0
                    
            logger.info(f'accuracy: {correct/totalCnt}')


metaData = IOUtils(inputFilePath)
metaData.buildData()

trainData = metaData.createDataset(inputFilePath)
devData = metaData.createDataset(inputDevPath)


model = CBOW(
    len(metaData.Vocab()),
    len(metaData.Labels()),
    32
)

modelTrain(model, 10, trainData, devData)