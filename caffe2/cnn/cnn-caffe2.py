#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:57:25 2020

@author: lacozhang
"""
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/Users/edwinzhang/src/torchcode/caffe2/layers")

from caffe2.python import (
    core, workspace, brew, model_helper, optimizer, layer_model_helper,
    schema, layer_model_instantiator, optimizer
)
from caffe2.python.modeling import initializers
from caffe2.python.layers import layers
from caffe2.python import net_drawer
import logging
import class_data as data
import numpy as np
from embedding import Embedding
import torch.nn.functional as F


logger = logging.getLogger(__name__)

class Utils(object):
    
    base_path = "/Users/edwinzhang/src/torchcode/caffe2/cnn/nn"
    
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
            
class TextCNNModel(object):
    def __init__(self, vocab_size, label_size, embed_size, query_length, kernel_size, num_filters):
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.embed_size = embed_size
        self.query_length = query_length
        self.kernel_size = kernel_size
        self.num_filters = num_filters
    
    def buildModel(self):
        self.input_feature_schema = schema.Struct(
            ('words', schema.Scalar((np.int32, (self.query_length,))))
        )
        self.trainer_extra_schema = schema.Struct(
            ('label', schema.Scalar((np.int32, (1,))))
        )

        self.model = layer_model_helper.LayerModelHelper(
            "cnn_text", 
            self.input_feature_schema, 
            self.trainer_extra_schema
        )

        self.input_records = schema.NewRecord(
            self.model.net, 
            self.input_feature_schema + self.trainer_extra_schema
        )


        self.model.default_optimizer = optimizer.AdagradOptimizer(
            alpha=0.1, 
            epsilon=1e-3
        )

        with core.NameScope("embedding"):            
            embeddings = self.model.Embedding(
                input_record=self.input_records.words,
                vocab_size=self.vocab_size,
                embed_size=self.embed_size,
            )
    
            embedding_reshaped = self.model.Reshape(
                embeddings,
                ["embedding_reshaped", "embedding_shape_info"],
                shape=[-1, 1, self.query_length, self.embed_size],
                output_dtypes=(np.float32, (1, self.query_length, self.embed_size))
            ).embedding_reshaped
    
    
        with core.NameScope("conv"):
            conv_output = self.model.Conv(
                embedding_reshaped,
                output_dim=self.num_filters,
                kernel_h=1, 
                kernel_w=self.kernel_size,
                stride_h=1, 
                stride_w=1,
                pad_t=0,
                pad_b=0,
                pad_l=0,
                pad_r=0
            )
    
        conv_max_reduce = self.model.ReduceMax(
            conv_output,
            ["conv_max_reduce"],
            axes=[1,2],
            keepdims=False,
            output_dtypes=(np.float32, (self.num_filters,))
        ).conv_max_reduce
    
        conv_relu = self.model.Relu(
            conv_max_reduce,
            ["conv_relu"]
        ).conv_relu

        # Transform into 1-D tensor and apply Sigmoid
        current_record = self.model.FC(conv_relu, self.label_size)

        softmax, avgloss = self.model.SoftmaxWithLoss(
            [current_record, self.input_records.label], 
            ["softmax", "avgloss"],
            average_by_batch_size=True,
        )

        final_prediction = self.model.ArgMax(
            current_record,
            ["predict_class"],
            axis=1,
            keepdims=True,    
        )

        self.model.output_schema = schema.Struct(('prediction', final_prediction))
        self.model.loss = avgloss

        self.train_init_net, self.train_net = layer_model_instantiator.generate_training_nets(
            self.model
        )
        self.predict_net = layer_model_instantiator.generate_predict_net(
            self.model
        )
        self.eval_net = layer_model_instantiator.generate_eval_net(
            self.model
        )

BATCH_SIZE = 4
QUERY_LENGTH = 64
EMBEDDING = 16
KERNEL_SIZE = 8
NUM_FILTERS = 64

model = TextCNNModel(
    data.Metadata.vocab_size, 
    data.Metadata.label_size, 
    EMBEDDING, 
    QUERY_LENGTH,
    KERNEL_SIZE,
    NUM_FILTERS
)
model.buildModel()

workspace.RunNetOnce(model.train_init_net)


train = data.getTrainIterator(BATCH_SIZE)
dev = data.getDevIterator(BATCH_SIZE)
test = data.getTestIterator(BATCH_SIZE)


pad_idx = data.Metadata.word2id["<pad>"]

logger.info("pad index: {}".format(pad_idx))


net_created = False
predict_net_created = False


def fix_data(text):
    if text.size(1) < QUERY_LENGTH:
        text = F.pad(
            item.text, 
            (0, QUERY_LENGTH - item.text.size(1)), 
            mode='constant', 
            value=pad_idx
        )
    else:
        text = item.text
    
    return text



for epoch_idx in range(10):

    train.init_epoch()
    ex_cnt = 0
    print("epoch: {}".format(epoch_idx))
    for item in train:
        ex_cnt += 1
        
        text = fix_data(item.text)
        label = item.label
                
        schema.FeedRecord(
            model.input_records,
            [ text.numpy().astype(np.int32), label.numpy().astype(np.int32) ]
        )
        
        if not net_created:
            net_created = True
            workspace.CreateNet(model.train_net)
        
        try:
            workspace.RunNet(model.train_net.Name())
        except:
            logger.error(text.shape)
        
        if ex_cnt % 100 == 0:
            print(
                "Loss: {}".format(
                    workspace.FetchBlob("SoftmaxWithLoss/avgloss")))
            
            dev.init_epoch()
            total_cnt = 0.0
            right_cnt = 0.0
            for dev_item in dev:                                
                t = fix_data(dev_item.text)
                l = dev_item.label                
                
                if l.size(0) != BATCH_SIZE:
                    continue
                
                total_cnt += t.size(0)
                
                schema.FeedRecord(
                    model.input_records,
                    [ t.numpy().astype(np.int32), l.numpy().astype(np.int32) ]
                )
                
                if not predict_net_created:
                    predict_net_created = True
                    workspace.CreateNet(model.eval_net)
                
                workspace.RunNet(model.eval_net.Name())
                p = workspace.FetchBlob("ArgMax/predict_class")
                right_cnt += sum(p == l.numpy().astype(np.int32).reshape(BATCH_SIZE,1))
            
            print("eval accuray: {}".format(right_cnt / total_cnt))
            
    print("epoch finished")
        
        
# print(workspace.FetchBlob("embedding/embed_layer/output").shape)
# print(workspace.FetchBlob("embedding/Reshape/embedding_reshaped").shape)
# print(workspace.FetchBlob("conv/conv/output").shape)
# print(workspace.FetchBlob("fc/w").shape)
# print(workspace.FetchBlob("fc/output").shape)
# print(workspace.FetchBlob("ArgMax/predict_class"))
# print(workspace.FetchBlob("conv/conv/conv_kernel").shape)

Utils.plot_network(model.train_net, "train.svg")
Utils.plot_network(model.train_init_net, "param_init.svg")
Utils.plot_network(model.predict_net, "predict_net.svg")
Utils.plot_network(model.eval_net, "eval_net.svg")