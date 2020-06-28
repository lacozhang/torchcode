#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:51:07 2020

@author: lacozhang
"""

ptbTrain = "/Users/edwinzhang/src/nn4nlp-code/data/ptb/train.txt"
ptbValid = "/Users/edwinzhang/src/nn4nlp-code/data/ptb/valid.txt"
ptbTest = "/Users/edwinzhang/src/nn4nlp-code/data/ptb/test.txt"

import codecs
import re
import logging

logger =logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LanguageModelData(object):
    UNKToekn = "<unk>"
    UNKID = 0
    StartToken = "<s>"
    StartID = 1
    EndToken = "</s>"
    EndID = 2
    
    WSTokenizer = re.compile('\s+')
    
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.trainData = None
        self.validData = None
        self.evalData = None
        
        
        self._updateSpecialToken()
        self._updateMetadata(ptbTrain)
        return None
        
    def _updateSpecialToken(self):
        self.word2id.update(
            {
                LanguageModelData.UNKToekn : LanguageModelData.UNKID,
                LanguageModelData.StartToken: LanguageModelData.StartID,
                LanguageModelData.EndToken: LanguageModelData.EndID
            }
        )
        
        self.id2word.update(
            {
                LanguageModelData.UNKID : LanguageModelData.UNKToekn,
                LanguageModelData.StartID : LanguageModelData.StartToken,
                LanguageModelData.EndID : LanguageModelData.EndToken
            }
        )
        
    def _updateMetadata(self, inputFile):
        with codecs.open(inputFile, "r", "utf-8") as src:
            for line in src:
                words = re.split(LanguageModelData.WSTokenizer, line.strip())
                for word in words:
                    if word not in self.word2id:
                        idx = len(self.word2id)
                        self.word2id[word] = idx
                        self.id2word[idx] = word
        self.VocabSize = len(self.word2id)
                        
    
    
    def ReadData(self, filePath, appendTag=False):
        fullData = []
        with codecs.open(filePath, 'r', 'utf-8') as src:
            for line in src:
                textLine = line
                if appendTag:
                    textLine = LanguageModelData.StartToken + " " + line + " " + LanguageModelData.EndToken
                
                words = re.split(LanguageModelData.WSTokenizer, textLine)
                wordIds = []
                for word in words:
                    if word in self.word2id:
                        wordIds.append(self.word2id[word])
                    else:
                        wordIds.append(self.UNKID)
                    
                    
                fullData.append(wordIds)
            
            logger.info(f"Read {len(fullData)} sentence")
        return fullData
    
    def _processData(self, data, loglinear=False, batchSize=2, ngram=2):
        trainData = []
        if loglinear:
            cnt = 0
            for line in data:
                cnt += len(line)
                for i in range(ngram, len(line)):
                    trainData.append(
                        (line[(i-ngram):i], line[i])
                    )
                    
            print(f"Total {cnt} Tokens")
        else:
            fullLine = []
            for line in data:
                fullLine.extend(line)
            
            print(f"sequence length: {len(fullLine)}")
            
            
            seqLen = (len(fullLine) - batchSize * 2)//batchSize
            print(f"sequence length: {seqLen}")
            trainData = []
            for i in range(batchSize):
                trainData.append(
                    (
                        fullLine[(i*seqLen) : ((i+1)*seqLen)],
                        fullLine[(i*seqLen+1): ((i+1)*seqLen+1)]
                    )
                )
        
        return trainData
                
        
    def GetData(self, filePath, loglinear=False, appendTag=False, batchSize=2, ngram=2):
        data = self.ReadData(filePath, appendTag=appendTag)
        res = self._processData(
            data, 
            loglinear=loglinear, 
            batchSize=batchSize, 
            ngram=ngram
        )
        
        return res

    def getTrainData(self, loglinear=False, appendTag=False, batchSize=2, ngram=2):
        if self.trainData is None:        
            self.trainData = self.GetData(
                ptbTrain,     
                loglinear=loglinear, 
                appendTag=appendTag, 
                batchSize=batchSize, 
                ngram=ngram)
        
        return self.trainData

    def getValidData(self, loglinear=False, appendTag=False, batchSize=2, ngram=2):
        if self.validData is None:
            self.validData = self.GetData(
                ptbValid,
                loglinear=loglinear,
                appendTag=appendTag,
                batchSize=batchSize,
                ngram=ngram)
            
        
        return self.validData
    
    def getEvalData(self, loglinear=False, appendTag=False, batchSize=2, ngram=2):
        if self.evalData is None:
            self.evalData = self.GetData(
                ptbTest,
                loglinear=loglinear,
                appendTag=appendTag,
                batchSize=batchSize,
                ngram=ngram)
            
        return self.evalData
            