#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 17:57:07 2018

@author: edwinzhang
"""

from __future__ import (unicode_literals, print_function, division)
import codecs
import unicodedata
import string
import re
import random
import argparse

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--src", dest='src')
args = parser.parse_args()

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_data(src_file):
    lines = []
    with codecs.open(src_file, 'r', 'utf-8') as src:
        for line in src:
            clean_line = line.strip().split('\t')
            assert len(clean_line) == 2, "line {}".format(line)
            lines.append(
                    (normalizeString(clean_line[0]), 
                     normalizeString(clean_line[1])))
    return lines

SOS_Token = 'SOS'
EOS_Token = 'EOS'
UNK_Token = 'UNK'

class LanguageData(object):
    
    def __init__(self, lines, topk):
        self.lines = lines
        self.word2index = {SOS_Token: 0, EOS_Token: 1, UNK_Token: 2}
        self.index2word = {0: SOS_Token, 1: EOS_Token, 2: UNK_Token}
        self.counts = {}
        self.tokenizer = re.compile('\s+')
        self.topk = topk
        
        self.addCount()
        self.keep()
        self.addWord()
        
    def addCount(self):
        for line in self.lines:
            for word in self.tokenizer.split(line.strip()):
                if word not in self.counts:
                    self.counts[word] = 1
                else:
                    self.counts[word] += 1
    
    def keep(self):
        sorted_words = sorted(
                self.counts.items(), 
                key = lambda item: (item[1], item[0]), 
                reverse=True)
        print(sorted_words[:10])
        self.kept_words = sorted_words[:self.topk]
    
    def addWord(self):
        for word, cnt in self.kept_words:
            if word not in self.word2index:
                token_index = len(self.word2index)
                self.word2index[word] = token_index
                self.index2word[token_index] = word

data = read_data(args.src)

eng_lang = LanguageData(
        [eng for (eng, fra) in data], 
        3000)

fra_lang = LanguageData(
        [fra for (eng, fra) in data],
        3000)

# -- input & output manipulation

MAX_LENGTH = 10

def indicesFromSentence(lang, sentence):
    indices = []
    for word in lang.tokenizer.split(sentence):
        if word not in lang.word2index:
            indices.append(lang.word2index[UNK_Token])
        else:
            indices.append(lang.word2index[word])
    return indices

def tensorFromSentence(lang, sentence):
    indices = indicesFromSentence(lang, sentence)
    indices.append(lang.word2index[EOS_Token])
    return torch.tensor(indices, dtype=torch.long).view(-1, 1)

def tensorFromPair(pair):
    input_tensor = tensorFromSentence(eng_lang, pair[0])
    target_tensor = tensorFromSentence(fra_lang, pair[1])
    return (input_tensor, target_tensor)

# -- define models

class EncoderRNN(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        embed = self.embedding(input).view(1, 1, -1)
        output = embed
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class DecoderRNN(nn.Module):
    
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, hidden):
        print("Encoder starts")
        print('input size : {}'.format(input.size()))
        output = self.embedding(input).view(1, 1, -1)
        print('output size : {}'.format(output.size()))
        output = F.relu(output)
        print('output size : {}'.format(output.size()))
        output, hidden = self.gru(output, hidden)
        print('output size : {}'.format(output.size()))
        print('hidden size : {}'.format(hidden.size()))
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), 
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)        
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

data_tensors = [tensorFromPair(p) for p in data]

# -- training script
teacher_forcing_ratio = 0.5

def train(
        input_tensor, 
        target_tensor, 
        encoder, decoder, 
        encoder_optimizer, 
        decoder_optimizer, 
        criteion, 
        max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    
    loss = 0.0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[0]])
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            loss += criteion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            
            loss += criteion(decoder_output, target_tensor[di])
            if decoder_input.item() == 1:
                break
    
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / target_length

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset ffevery plot_every
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    random.shuffle(data_tensors)
    
    for idx in range(n_iters):
        pair = data_tensors[idx]
        idx += 1

        input_tensor = pair[0]
        target_tensor = pair[1]
        
        input_length = input_tensor.size(0)
        if input_length > MAX_LENGTH:
            continue
        
        loss = train(input_tensor, target_tensor, encoder, decoder, 
                       encoder_optimizer, decoder_optimizer, criterion)
        
        print_loss_total += loss
        plot_loss_total += loss

        if idx % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, idx / n_iters),
                                         idx, idx / n_iters * 100, print_loss_avg))

        if idx % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    
    with torch.no_grad():
        input_tensor = tensorFromSentence(eng_lang, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
        
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                    input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        
        decoder_input = torch.tensor([[0]])
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, 
                    decoder_hidden, 
                    encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.topk(1)
            if topi.item() == 1:
                decoded_words.append('<EOS>')
                break
            else:
                decoder_input = topi.squeeze().detach()
                
        return decoded_words, decoder_attentions[:di + 1]

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

hidden_size = 256
encoder1 = EncoderRNN(len(eng_lang.word2index), hidden_size)
attn_decoder1 = AttnDecoderRNN(
        hidden_size, len(fra_lang.word2index), dropout_p=0.1)

trainIters(encoder1, attn_decoder1, 20, print_every=5000)