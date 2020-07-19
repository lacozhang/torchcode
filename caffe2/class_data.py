#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:04:48 2020

@author: lacozhang@gmail.com
"""

import torchtext
from torchtext import data, vocab
import codecs


train_data = "/Users/edwinzhang/src/nn4nlp-code/data/classes/train.txt"
dev_data = "/Users/edwinzhang/src/nn4nlp-code/data/classes/dev.txt"
test_data = "/Users/edwinzhang/src/nn4nlp-code/data/classes/test.txt"


TEXT = torchtext.data.Field(
    use_vocab=True,
    batch_first=True,
    lower=True,    
)

LABEL = torchtext.data.Field(
    sequential=False,
    use_vocab=True,
    batch_first=True,
    unk_token=None,
    pad_token=None
)


class ClassificationData(torchtext.data.Dataset):
    def __init__(self, file_path, text_field, label_field, **kwargs):
        
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        
        for line in codecs.open(file_path, 'r', 'utf-8'):
            segments = line.strip().split('|||')
            if len(segments) != 2:
                continue
            examples.append(
                data.Example.fromlist(
                    [segments[1], int(segments[0])], 
                    fields
                )
            )
            
        super(ClassificationData, self).__init__(examples, fields, **kwargs)
        
    
    @staticmethod
    def sort_key(ex):
        return len(ex.text)


train = ClassificationData(train_data, TEXT, LABEL)
dev = ClassificationData(dev_data, TEXT, LABEL)
test = ClassificationData(test_data, TEXT, LABEL)

TEXT.build_vocab(train, dev)
LABEL.build_vocab(train, dev)

class Metadata(object):
    vocab_size = len(TEXT.vocab.itos)
    word2id = TEXT.vocab.stoi
    id2word = TEXT.vocab.itos    
    label_size = len(LABEL.vocab.itos)


def getTrainIterator(batch_size, epochs=1):
    return data.Iterator(
        train, 
        batch_size=batch_size, 
        train=True, 
        repeat=False,
        shuffle=True,
        sort=True
    )

def getDevIterator(batch_size):
    return data.Iterator(
        dev,
        batch_size=batch_size,
        train=True,
        repeat=False,
        shuffle=True,
        sort=True
    )

def getTestIterator(batch_size):
    return data.Iterator(
        test,
        batch_size=batch_size,
        train=False,
        repeat=False,
        shuffle=False,
        sort=False
    )