#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:49:28 2020

@author: lacozhang
"""

from caffe2.python import ( 
    core, 
    workspace, 
    optimizer, 
    model_helper, 
    brew
 )
from caffe2.proto import caffe2_pb2
from caffe2.python import net_drawer
from lm_reader import LanguageModelData
import logging
import random
import numpy as np


logger = logging.getLogger(__name__)


class Utils(object):
    
    base_path = "/Users/edwinzhang/src/torchcode/caffe2/lm/nn"
    
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


class LogLinearModel(object):
    
    def __init__(self, vocabSize, ngram=2, batchSize=32):
        self.ngram = ngram
        self.vocabSize = vocabSize
        self.batchSize = batchSize
        
        
    def createModel(self):
        
        self.model = model_helper.ModelHelper("loglinear_model")
        self.weights = []
        self.inputs = []
        self.bias = self.model.param_init_net.XavierFill(
            [],
            "bias",
            shape=[1, 1, 1, self.vocabSize]
        )
        
        self.mut_s = self.model.param_init_net.ConstantFill(
            [],
            "mut_s",
            value=self.ngram * 1.0,
            dtype=1,
            shape=[1, 1, 1, self.vocabSize]
        )
        
        self.mut_no_grad = self.model.net.StopGradient(
            self.mut_s,
            self.mut_s
        )
        
        for i in range(self.ngram):
            self.inputs.append(
                self.model.net.AddExternalInput(f"pos_{i}")
            )
            self.weights.append(
                self.model.param_init_net.XavierFill(
                [],
                f"weights_{i}",
                shape=[self.vocabSize, self.vocabSize]
                )
            )
            
        
        self.loglinear_results = []
        for i in range(self.ngram):
            self.loglinear_results.append(
                self.model.net.Gather(
                    [self.weights[i], self.inputs[i]],
                    [f"score_{i}"]
                )
            )
        
        self.last_combine, _ = self.model.net.Concat(
            self.loglinear_results,
            ["combine", "split_info"],
            axis=2
        )
        
        
        self.score_avg = brew.average_pool(
            self.model,
            self.last_combine,
            "score_avg",
            kernel_h=2, kernel_w=1,
            stride_h=2, stride_w=1
        )
        
        self.score_sum =self.model.net.Mul(
            [self.score_avg, self.mut_s],
            ["score_sum"],
            axis=0,
        )
        
        self.logits = self.model.net.Add(
            [self.score_sum, self.bias],
            ["logits"],
            axis=0,
        )
        
        self.predict_net = core.Net(self.model.net.Proto())
        self.label = self.model.net.AddExternalInput("label")
        
        self.predict_word = self.predict_net.ArgMax(
            [self.logits],
            ["predict_word"],
            axis=2
        )
                
        self.softmax, self.avgloss = self.model.net.SoftmaxWithLoss(
            [self.logits, self.label],
            ["softmax", "avgloss"],
        )
        
        self.model.AddGradientOperators([self.avgloss])
        
        optimizer.build_adagrad(
            self.model,             
            base_learning_rate=1e-2,
            epsilon=1e-8
        )
        
    def trainModel(self, trainData, batchSize, epochs, ngram, validData=None, evalData=None):
        
        workspace.RunNetOnce(self.model.param_init_net)
        logger.info("init_net creation")
        already_init = False
        
        for epochIter in range(epochs):
            logger.info(f"iteration {epochIter}")
            
            random.shuffle(trainData)
            cnt = 0
            for idx in range(0, len(trainData), batchSize):
                cnt += 1
                endIdx = min(idx+batchSize, len(trainData))
                
                miniBatchData = trainData[idx:endIdx]
                
                inputData = []
                inputLabel = [ [ [pair[1]] ] for pair in miniBatchData ]
                for i in range(ngram):
                    inputData.append(
                        [
                           [[w[0][i]]] for w in miniBatchData
                        ]
                    )

                for i in range(ngram):
                    workspace.FeedBlob(
                        f"pos_{i}",
                        np.array(inputData[i], dtype=np.int32)
                    )
                    
                workspace.FeedBlob(
                    self.label,
                    np.array(inputLabel, dtype=np.int32)
                )                

                if not already_init:
                    workspace.CreateNet(self.model.net)
                    already_init = True
                    print("init done")
                
                workspace.RunNet(self.model.net.Name())
                
                if cnt % 1000 == 0:
                    logger.info(f"Train {cnt} mini-batches")

                # print(workspace.FetchBlob("avgloss").shape)
                # print(workspace.FetchBlob("avgloss"))
                # print(workspace.FetchBlob("average_loglinear").shape)
                # print(inputData[0])
                # exit()

NGRAM = 2
BATCH_SIZE=32
lmData = LanguageModelData()
trainData = lmData.getTrainData(loglinear=True, ngram=NGRAM)
validData = lmData.getValidData(loglinear=True, ngram=NGRAM)
evalData = lmData.getEvalData(loglinear=True, ngram=NGRAM)


model = LogLinearModel(
    lmData.VocabSize, 
    NGRAM, 
    BATCH_SIZE
)

model.createModel()


Utils.plot_network(
    model.model.net, 
    Utils.base_path + "_train.svg"
)

Utils.plot_network(
    model.predict_net, 
    Utils.base_path + "_predict.svg"
)


model.trainModel(trainData, 32, 1, NGRAM)

            
        