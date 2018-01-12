#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:54:44 2018

@author: zhangyu
"""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

def to_scalar(var):
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def log_sum_exp(vec):
    max_val = vec[0, argmax(vec)]
    max_val_broadcast = max_val.view(1, -1).expand(1, vec.size()[1])
    return max_val + torch.log(torch.sum(
            torch.exp(vec - max_val_broadcast)))
    

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4


class BiLSTMCRF(nn.Module):
    """ CRF on top of bi-directional LSTM as potential function
    
    """
    
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        """
            For the transition parameter, the i-th row is the transition 
            parameter to tag id $i$
        """
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(
                torch.randn(self.tagset_size, self.tagset_size))
        
        self.transition.data[self.tag_to_ix[START_TAG], :] = -10000.0
        self.transition.data[:, self.tag_to_ix[STOP_TAG]] = -10000.0
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))
    
    def _calculate_alpha(self, feats):
        """Calculate the alpha in the forward-backward process"""
        
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = autograd.Variable(init_alphas)

        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                trans_score = self.transition[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transition[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    
    def _get_lstm_feature(self, sentence):
        self.hidden = self.init_hidden()
        embedding = self.embedding(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embedding, self.hidden)
        norm_lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        potential_func = self.hidden2tag(norm_lstm_out)
        return potential_func
    
    def _score_sentence(self, feats, tags):
        
        score = autograd.Variable(torch.Tensor([0.0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]),
                                           tags])
        for i, feat in enumerate(feats):
            score = score + feat[tags[i+1]] + self.transition[tags[i+1], tags[i]]
        
        score = score + self.transition[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        
        backpointers = []

        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transition[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transition[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_feature(sentence)
        alpha = self._calculate_alpha(feats)
        gold = self._score_sentence(feats, tags)
        return alpha - gold
    
    def forward(self, sentence):
        feats = self._get_lstm_feature(sentence)
        score, tags = self._viterbi_decode(feats)
        return score, tags


training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTMCRF(len(word_to_ix), tag_to_ix, 
                  embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
print(model(precheck_sent))

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.LongTensor([tag_to_ix[t] for t in tags])
        # Step 3. Run our forward pass.
        neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        neg_log_likelihood.backward()
        optimizer.step()

# Check predictions after training
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
print(model(precheck_sent))
# We got it!