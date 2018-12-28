#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 17:31:13 2018

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
n_letters_unk = n_letters + 1
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

def letterToIndex(c):
    return all_leters.find(c) + 1

def letterToTensor(c):
    dat = torch.zeros(1, n_letters_unk)
    dat[0][letterToIndex(c)] = 1
    return dat

def lineToTensor(name):
    dat = torch.zeros(len(name), 1, n_letters_unk)
    for pos, c in enumerate(name):
        dat[pos][0][letterToIndex(c)] = 1
    return dat

def labelToTensor(label):
    return torch.tensor([label], dtype=torch.long)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i

print(lineToTensor("tech").size())

class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.soft_max = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.soft_max(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters_unk, n_hidden, n_categories)

sgd = optim.SGD(rnn.parameters(), lr=0.005)
criterion = nn.NLLLoss()
n_epochs = 10
print_every = 5000
plot_every = 1000

def train_one_iter(line, label):
    sgd.zero_grad()
    hidden = rnn.initHidden()
    for i in range(line.size()[0]):
        output, hidden = rnn(line[i], hidden)
    
    loss = criterion(output, label)
    loss.backward()
    
    sgd.step()
    return output, loss.item()


def train_one_epoch(samples):
    global current_loss
    global all_losses
    global iter_cnt
    indices = list(range(len(samples)))
    random.shuffle(indices)
    for index in indices:
        iter_cnt += 1
        line = lineToTensor(samples[index][0])
        label = labelToTensor(samples[index][1])
        output, loss = train_one_iter(line, label)
        current_loss += loss
        if iter_cnt % print_every == 0:
            print("iter: {}, time: {}".format(iter_cnt, timeSince(start)))
        
        if iter_cnt % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0.0
    return None

current_loss = 0.0
all_losses = []
iter_cnt = 0

def timeSince(oldtime):
    time_diff = time.time() - oldtime
    return '{}m {}s'.format(math.floor(time_diff/60), (int)(time_diff % 60))

start = time.time()
for epoch in range(n_epochs):
    train_one_epoch(train_data.data)

plt.figure()
plt.plot(all_losses)

confusion_matrix = torch.zeros(n_categories, n_categories)

def evaluate_one_iter(x):
    with torch.no_grad():
        hidden = rnn.initHidden()
        for idx in range(x.size()[0]):
            output, hidden = rnn(x[idx], hidden)
        return output

for (line, label) in train_data.data:
    x = lineToTensor(line)
    y = labelToTensor(label)
    predict = categoryFromOutput(evaluate_one_iter(x))
    confusion_matrix[label][predict] += 1

for i in range(n_categories):
    confusion_matrix[i] = confusion_matrix[i] / confusion_matrix[i].sum()

all_categories = []
for i in range(n_categories):
    all_categories.append(train_data.reverse_labels[i])

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion_matrix.numpy())
fig.colorbar(cax)
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()