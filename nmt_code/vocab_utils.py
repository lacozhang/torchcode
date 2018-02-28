#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 11:27:05 2018

@author: zhangyu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

from tensorflow.python.ops import lookup_ops
from . import misc_utils as utils

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0


def load_vocab(vocab_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab, vocab_size

def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None,
                eos=None, unk=None):
    """Check if vocab file doesn't exist, create from corpus file"""
    if tf.gfile.Exists(vocab_file):
        utils.print_out("# Vocab file %s exist"%(vocab_file))
        vocab, vocab_size = load_vocab(vocab_file)
        
        if check_special_token:
            if not unk: unk = UNK
            if not sos: sos = SOS
            if not eos: eos = EOS
            assert len(vocab) >= 3
            if vocab[0] != UNK or vocab[1] != sos or vocab[2] != eos:
                utils.print_out("The first 3 vocab words [%s, %s, %s]"
                                " are not [%s, %s, %s]"%(
                                        vocab[0], vocab[1], vocab[2],
                                        unk, sos, eos))
                vocab = [unk, sos, eos] + vocab
                vocab_size += 3
                new_vocab_file = os.path.join(out_dir, 
                                              os.path.basename(vocab_file))
                with codecs.getwriter("utf-8")(
                        tf.gfile.GFile(new_vocab_file, "wb")) as f:
                    for word in vocab:
                        f.write("%s\n"%(word))
                vocab_file = new_vocab_file
    else:
        raise ValueError("vocab file %s does not exist"%(vocab_file))
    
    vocab_size = len(vocab)
    return vocab_size, vocab


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
    """Creates vocab tables for src_vocab_file and tgt_vocab_file"""
    src_vocab_table = lookup_ops.index_table_from_file(
            src_vocab_file, default_value=UNK_ID)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = lookup_ops.index_table_from_file(
                tgt_vocab_file, default_value=UNK_ID)
    return src_vocab_table, tgt_vocab_table

def load_embed_text(embed_file):
    """Load embed text into python dictionary
    
    embed_file in Glove format as follow:
    word -1. 20 
    """
    
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, "rb")) as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), "All embeddings should be same size"
            else:
                emb_size = len(vec)
    return emb_dict, emb_size