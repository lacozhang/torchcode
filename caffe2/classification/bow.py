#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:23:25 2020

@author: yu zhang
"""

import codecs
import random
from caffe2.python import workspace, core, model_helper, brew, utils
from caffe2.python.modeling import initializers
from caffe2.proto import caffe2_pb2
from caffe2.python import net_drawer
from caffe2.python.optimizer import build_sgd
import logging
import numpy as np
from IPython import display
import copy
from io_utils import (
    inputFilePath,
    inputDevPath,
    IOUtils
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


            
    
class Utils(object):
    
    base_path = "/Users/edwinzhang/src/torchcode/caffe2/classification/nn"
    
    @staticmethod
    def print_value_shape(blob):
        print("Blob {} - shape: {}/value: {}".format(
            blob,
            workspace.FetchBlob(blob).shape,
            workspace.FetchBlob(blob),            
        ))
        
    @staticmethod
    def plot_network(net, path):
        graph = net_drawer.GetPydotGraph(net, rankdir="LR")
        with open(path, 'wb') as f:
            f.write(graph.create_svg())
        


class BOWModel(object):
    
    def __init__(self, vocab, embed_size, nclass):
        self.vocab = sorted(vocab)
        self.word2idx = {
            word:index for (index, word) in enumerate(vocab)
        }
        self.idx2word = {
            index:word for (index, word) in enumerate(vocab)
        }
        self.embed_size = embed_size
        self.nclass = nclass
        self.create_predict = False
        
    def CreateModel(self):
        self.model = model_helper.ModelHelper("BOW_Model")

        self.words = self.model.net.AddExternalInput(
            "words"
        )
        
        self.embedding = self.model.param_init_net.XavierFill(
            [], 
            "embedding", 
            shape=[len(self.vocab), self.embed_size],
        )
        self.words_embedding = self.model.net.Gather(
            [self.embedding, self.words],
            ["word_embedding"]
        )
        self.words_embedding_reshape, _ = self.model.net.Reshape(
            [self.words_embedding],
            ["word_embed_reshaped", "old_shape"],
            shape=[-1, self.embed_size]
        )
        self.embedding_sum = self.model.net.ReduceSum(
            [self.words_embedding_reshape],
            ["embedding_sum"],
            axes=(0,),
        )
        self.embedding_sum_reshape, _ = self.model.net.Reshape(
            [self.embedding_sum],
            ["embedding_sum_reshape", "sum_old_reshape"],
            shape=[-1, self.embed_size]
        )
        self.logits = brew.fc(
            self.model,
            self.embedding_sum_reshape, 
            "logits", 
            self.embed_size,
            self.nclass,
        )
        
        self.predict_net = core.Net(self.model.net.Proto())
        
        # Argmax
        self.predict_label = self.predict_net.ArgMax(
            [self.logits],
            ["predict_label"],
            axis=1,
        )
        
        self.label = self.model.net.AddExternalInput("label")        
        self.softmax, self.avgloss = self.model.net.SoftmaxWithLoss(
            [self.logits, self.label],
            ["softmax", "avgloss"],
        )
        
        self.model.AddGradientOperators([self.avgloss])
        
        build_sgd(
            self.model,
            base_learning_rate=0.01,
            policy="step",
            stepsize=1,
            gamma=0.9999,
        )
        
    def _prepareInput(self, label, words):
        wordsArray = np.array(
            [[ self.word2idx[w]] for w in words if w in self.word2idx],
            dtype=np.int32,
        )
        if label is not None:
            labelArray = np.array(
                [label], 
                dtype=np.int32
            )
        else:
            labelArray=None
        return wordsArray, labelArray
        

    def TrainModel(self, trainData, epochs, devData=None):
        workspace.RunNetOnce(self.model.param_init_net)
        logger.info("init Net created")
        netCreated=False
        iterCnt = 0
        for epochCnt in range(epochs):
            random.shuffle(trainData)
            logger.info("Epoch: {}".format(epochCnt))
            for (label, words) in trainData:
                iterCnt += 1
                
                wordsArray, labelArray = self._prepareInput(label, words)
                workspace.FeedBlob(self.words, wordsArray)
                workspace.FeedBlob(self.label, labelArray)
                
                if not netCreated:
                    netCreated=True
                    logger.info("net creation done!")
                    workspace.CreateNet(self.model.net)
                    logger.info("train Net created")
                
                workspace.RunNet(self.model.net.Name())
                
                if iterCnt % 1000 == 0:
                    if devData is not None:
                        self.EvaluateAccuracy(devData)

    def Predict(self, sentence):
        if not self.create_predict:
            self.create_predict = True
            workspace.CreateNet(self.predict_net)
        
        words, _ = self._prepareInput(None, sentence)
        workspace.FeedBlob(self.words, words)
        workspace.RunNet(self.predict_net.Name())
        
        predicted_label = workspace.FetchBlob(self.predict_label)
        return predicted_label
        
        
    def EvaluateAccuracy(self, dataset):
        total_cnt = 0.0
        correct_cnt = 0.0
        for (label, sentence) in dataset:
            predicted_label = self.Predict(sentence)
            if predicted_label == label:
                correct_cnt += 1
            total_cnt += 1

        print("Accuracy: {}".format(
            correct_cnt / total_cnt
        ))
        
        
        
        

trainData = IOUtils(inputFilePath)
trainData.buildData()
logger.info("Train data preparation finished")

devData = trainData.createDataset(inputDevPath)

bowModel = BOWModel(trainData.Vocab(), 32, len(trainData.Labels()))
bowModel.CreateModel()
logger.info("Model Creation Finished")

bowModel.TrainModel(
    trainData.Data(), 
    10,
    devData
)

Utils.plot_network(
    bowModel.model.net, 
    Utils.base_path + "_train.svg"
)

Utils.plot_network(
    bowModel.predict_net, 
    Utils.base_path + "_predict.svg"
)

#bowModel.TrainModel(trainData.Data(), 3)