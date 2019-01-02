#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 17:43:41 2018

@author: edwinzhang
"""

from __future__ import (unicode_literals, print_function, division)
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import unicodedata
import string
import codecs
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import namedtuple
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def findFiles(path):
    return glob.glob(path)

parser = argparse.ArgumentParser()
parser.add_argument("--input", dest='src')
arguments = parser.parse_args()

logger.info("file path {}".format(arguments.src))
filepattern = os.path.join(arguments.src, "*.txt")

all_leters = string.ascii_letters + " .,;'"
n_letters = len(all_leters)

# - include SOS-0/EOS-1
n_letters_unk = n_letters + 2
logger.info("Vocabulary of character", n_letters)

def unicodeToAscii(line):
    return ''.join(c for c in unicodedata.normalize('NFD', line)
                   if unicodedata.category(c) != 'Mn' and c in all_leters)

print(unicodeToAscii('Ślusàrski'))

TrainData = namedtuple("TrainData", ['labels', 'reverse_labels', 'data'])

def read_line(filepath):
    lines = codecs.open(filepath, 'r', 'utf-8').read().strip().split('\n')
    
    return [unicodeToAscii(line) for line in lines]

def read_dataset(path_glob):
    label_mapping = {}
    reverse_label_mapping = {}
    label_data = []
    for filepath in findFiles(path_glob):
        category = os.path.splitext(os.path.basename(filepath))[0]
        assert len(category) > 0
        if category not in label_mapping:
            label_index = len(label_mapping)
            label_mapping[category] = label_index
            reverse_label_mapping[label_index] = category
        label = label_mapping[category]
        label_data.extend([ (line, label) for line in read_line(filepath)])
    return TrainData(
            labels=label_mapping, 
            reverse_labels=reverse_label_mapping, 
            data=label_data
            )

train_data = read_dataset(filepattern)
n_categories = len(train_data.labels)
print("Total categories: {}".format(n_categories))
print("Total data size : {}".format(train_data.data[:20]))

# --- define models
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(
                input_size + hidden_size + n_categories, hidden_size)
        self.i2o = nn.Linear(
                input_size + hidden_size + n_categories, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.soft_max = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden, category):
        merged_input = torch.cat((input, hidden, category), 1)
        hidden = self.i2h(merged_input)
        output = self.i2o(merged_input)
        mid_output = torch.cat((hidden, output), 1)
        output = self.o2o(mid_output)
        output = self.dropout(output)
        output = self.soft_max(output)
        
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNN(n_letters_unk, 128, n_letters_unk)

# --- update the training set
def letterToIndex(c):
    idx = all_leters.find(c)
    if idx != -1:
        return idx + 2
    else:
        return idx

def letterToTensor(c):
    dat = torch.zeros(1, n_letters_unk)
    dat[0][letterToIndex(c)] = 1
    return dat

def lineToTensor(name):
    indices = []
    indices.append(0)
    for c in name:
        indices.append(letterToIndex(c))
    dat = torch.zeros(len(indices), 1, n_letters_unk)
    for pos, idx in enumerate(indices):
        dat[pos][0][idx] = 1
    return dat

def categoryToTensor(label):
    dat = torch.zeros(1, n_categories)
    dat[0][label] = 1
    return dat

def lineToTarget(name):
    target_indices = []
    for c in name:
        target_indices.append(letterToIndex(c))
    target_indices.append(1)
    return torch.LongTensor(target_indices)

# -- setup optimizaer & loss function
n_epochs = 10
iter_cnt = 0
print_every = 10000
plot_every = 1000

current_loss = 0.0
all_losses = []
start_time = time.time()

sgd = optim.SGD(rnn.parameters(), lr=0.0005, weight_decay=1e-5)
criterion = nn.NLLLoss()

def timeSince(oldtime):
    time_diff = time.time() - oldtime
    return '{}m {}s'.format(math.floor(time_diff/60), (int)(time_diff % 60))

def train_one_iter(name, category):

    sgd.zero_grad()
    rnn.zero_grad()
    
    x = lineToTensor(name)
    y = lineToTarget(name)
    y.unsqueeze_(-1)
    aux = categoryToTensor(category)
    hidden = rnn.initHidden()
    
    n_steps = x.size(0)
    loss = 0
    for i in range(n_steps):
        output, hidden = rnn(x[i], hidden, aux)
        l = criterion(output, y[i])
        loss += l
    
    loss.backward()
    sgd.step()
    
    return output, loss.item()/n_steps

def train_one_epoch(samples):
    global iter_cnt
    global current_loss
    
    random.shuffle(samples)
    
    for sample in samples:
        name, category = sample[0], sample[1]
        iter_cnt += 1
        
        sample_output, sample_loss = train_one_iter(name, category)
        current_loss += sample_loss
        
        if iter_cnt % print_every == 0:
            print("iter: {}/time: {}".format(iter_cnt, timeSince(start_time)))
        if iter_cnt % plot_every == 0:
            all_losses.append(current_loss/plot_every)
            current_loss = 0.0
    return None

for i in range(n_epochs):
    train_one_epoch(train_data.data)

plt.figure()
plt.plot(all_losses)

max_length = 20

def sample(category):
    category_tensor = categoryToTensor(category)
    
    x = torch.zeros(1, n_letters_unk)
    x[0][0] = 1
    
    hidden = rnn.initHidden()
    gen_name = ""
    
    for i in range(max_length):
        output, hidden = rnn(x, hidden, category_tensor)
        top_v, top_i = output.topk(1)
        if top_i.item() == 1:
            break
        
        if top_i.item() == 0:
            break
        
        gen_name += all_leters[top_i.item()-2]
    
    return gen_name

for i in range(n_categories):
    print("Generate name for {}".format(train_data.reverse_labels[i]))
    gen_name = sample(i)
    print("Generated name: {}".format(gen_name))