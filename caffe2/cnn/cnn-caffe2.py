#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:57:25 2020

@author: lacozhang
"""
import sys
sys.path.append("/Users/edwinzhang/src/torchcode/caffe2/layers")
from caffe2.python import (
    core, workspace, brew, model_helper, optimizer, layer_model_helper,
    schema, layer_model_instantiator, optimizer
)
from caffe2.python.modeling import initializers
from caffe2.python.layers import layers
from caffe2.python import net_drawer
import logging
import numpy as np
from embedding import Embedding


logger = logging.getLogger(__name__)

try:
    layers.register_layer(Embedding.__name__, Embedding)
except:
    logger.info("Error")

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

input_feature_schema = schema.Struct(
    ('words', schema.Scalar((np.int32, (4, 1, 20))))
)
trainer_extra_schema = schema.Struct(
    ('label', schema.Scalar((np.int32, (4,))))
)

model = layer_model_helper.LayerModelHelper(
    "cnn_text", 
    input_feature_schema, 
    trainer_extra_schema
)

input_records = schema.NewRecord(
    model.net, 
    input_feature_schema + trainer_extra_schema)


model.default_optimizer = optimizer.SgdOptimizer(
    base_learning_rate=0.05, 
    momentum=0.01
)

with core.NameScope("embedding"):        
    
    embeddings = model.Embedding(
        input_record=input_records.words,
        vocab_size=30000,
        embed_size=32,
    )
    

    
with core.NameScope("conv"):
    conv_output = model.Conv(
        embeddings,
        output_dim=64,
        kernel_h=1, 
        kernel_w=9,
        stride_h=1, 
        stride_w=1,
        pad_t=0,
        pad_b=0,
        pad_l=0,
        pad_r=0
    )
    conv_reshape = model.Reshape(
        conv_output,
        ["conv_reshape", "conv_shape_info"],
        shape=[-1, 1, 768],
        output_dtypes=(np.float32, (1, 768))
    ).conv_reshape

# Transform into 1-D tensor and apply Sigmoid
current_record = model.FC(conv_reshape, 1)
final_prediction = model.Sigmoid(current_record, 1)

model.output_schema = schema.Struct(('prediction', final_prediction))
model.loss = model.BatchLRLoss(schema.Struct(
        ('label', input_records.label),
        ('logit', current_record)
    ), average_loss=True)

train_init_net, train_net = layer_model_instantiator.generate_training_nets(
    model)
predict_net = layer_model_instantiator.generate_predict_net(
    model)


workspace.RunNetOnce(train_init_net)

words = np.random.randint(1, 20000, size=(4, 1, 20))
labels = np.random.randint(0, 1, size=(4,))

schema.FeedRecord(
    input_records,
    [ words, labels ]
)

workspace.RunNetOnce(train_net)
# print(workspace.FetchBlob("embedding/embed_layer/output").shape)
# print(workspace.FetchBlob("conv/conv/output").shape)
# print(workspace.FetchBlob("fc/w").shape)
# print(workspace.FetchBlob("fc/output").shape)
# print(workspace.FetchBlob("Sigmoid/field_0"))
# print(workspace.FetchBlob("conv/conv/conv_kernel").shape)

Utils.plot_network(train_net, "train.svg")
Utils.plot_network(train_init_net, "param_init.svg")
Utils.plot_network(predict_net, "predict_net.svg")