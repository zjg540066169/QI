#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 15:14:09 2019

@author: jungangzou
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class Self_Attention(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, N = 1, heads = 1, dropout = 0.3):
        super().__init__()
        self.encoder = Encoder( input_size,hidden_size, N, heads, dropout)
        #self.decoder = Decoder( input_size, N, heads, dropout)
        self.out = nn.Linear(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
    def forward(self, src, src_mask = None):
        e_outputs = self.encoder(src, src_mask)
        e_outputs = F.relu(e_outputs)[:,-1,:]
        output = self.out(e_outputs)
        return output.view(-1,self.output_size)


class Encoder(nn.Module):
    def __init__(self, input_size,hidden_size, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(input_size, heads, hidden_size, dropout), N)
        self.norm = Norm(input_size)
    
    def forward(self, src, mask):
        x = src
        #x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Norm(nn.Module):
    def __init__(self, input_size, eps = 1e-6):
        super().__init__()
    
        self.size = input_size
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, input_size, dropout = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.d_k = input_size // heads
        self.h = heads
        
        self.q_linear = nn.Linear(input_size, input_size)
        self.v_linear = nn.Linear(input_size, input_size)
        self.k_linear = nn.Linear(input_size, input_size)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(input_size, input_size)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * input_size
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.input_size)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0.1):
        super().__init__()   
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, input_size, heads, hidden_size=2048,dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(input_size)
        self.norm_2 = Norm(input_size)
        self.attn = MultiHeadAttention(heads, input_size, dropout=dropout)
        self.ff = FeedForward(input_size, hidden_size,dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = self.dropout_1(self.attn(x2,x2,x2,mask))
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
