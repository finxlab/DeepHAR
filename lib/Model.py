#!/usr/bin/env python
# coding: utf-8

# In[110]:

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import json
import shutil

logger = logging.getLogger('Model') 
import torch
import torch.nn as nn

class UnifiedModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.model_type = configs.model_type
        if self.model_type == "LSTM":
            self.model = LSTM(configs)
        elif self.model_type == "BILSTM":
            self.model = BILSTM(configs)
        elif self.model_type == "GRU":
            self.model = GRU(configs)
        elif self.model_type == "DeepHAR":
            self.model = DeepHAR(configs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.model(x)) + self.configs.clamp_min
    
    @torch.no_grad()
    def init_before_training(self, train_loader, clamp_min=0.01, tries=10):
        configs = self.configs
        device = configs.device
        inputs, _ = next(iter(train_loader))
        inputs = inputs.to(device)

        for i in range(tries):
            if i > 0: 
                # Reset Parameters
                logger.info("--Parameter reset--")
                self.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

            outputs = self(inputs)
            if not torch.allclose(outputs, torch.full_like(outputs, configs.clamp_min), atol=1e-4):
                return torch.optim.Adam(self.parameters(), lr=configs.learning_rate)

        raise RuntimeError(f"Initialization failed after {tries} tries over.")

    def forward_attention(self, x):
        return self.model.forward_attention(x)



class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout = dropout)
        self.v = nn.Parameter(torch.randn(d_model, 1))

    def forward(self, x):
        # 1. Self-Attention: Query, Key, Value 
        # x: (Batch, Seq_Len, d_model)
        attn_out, _ = self.mha(x, x, x)
        
        # Attention Pooling 
        score = torch.matmul(attn_out, self.v)  # (Batch, Seq_Len, 1)
        weights = torch.softmax(score, dim=1)
        context = torch.sum(weights * attn_out, dim=1) # (Batch, d_model)
        
        return context

    def forward_attention(self, x):
        # 1. Self-Attention: Query, Key, Value 
        # x: (Batch, Seq_Len, d_model)
        attn_out, _ = self.mha(x, x, x)
        
        # Attention Pooling 
        score = torch.matmul(attn_out, self.v)  # (Batch, Seq_Len, 1)
        weights = torch.softmax(score, dim=1)
        context = torch.sum(weights * attn_out, dim=1) # (Batch, d_model)
        
        return weights, self.v

class DeepHAR(nn.Module):
    """
    Implementation of the proposed DeepHAR model for heterogeneous autoregressive 
    volatility forecasting with temporal self-attention.
    """
    def __init__(self, configs):
        super(DeepHAR, self).__init__()
        self.configs = configs

        self.day_proj = nn.Linear(configs.input_dim, configs.lstm_hidden_dim * 2)
        self.week_lstm = nn.LSTM(configs.input_dim, configs.lstm_hidden_dim, bidirectional=True, batch_first=True)
        self.month_lstm = nn.LSTM(configs.input_dim, configs.lstm_hidden_dim, bidirectional=True, batch_first=True)
        
        self.week_temp_attn = TemporalSelfAttention(configs.lstm_hidden_dim * 2, configs.n_heads, dropout=configs.dropout)
        self.month_temp_attn = TemporalSelfAttention(configs.lstm_hidden_dim * 2, configs.n_heads, dropout=configs.dropout)
        self.channel_mha = nn.MultiheadAttention(configs.lstm_hidden_dim * 2, configs.n_heads, batch_first=True, dropout=configs.dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(configs.lstm_hidden_dim * 4, 32), #32 
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        d, w, m = x[:, -1:], x[:, -5:], x[:, -22:]

        d_f = self.day_proj(d) # (B, 1, d_model)
        w_o, _ = self.week_lstm(w)
        m_o, _ = self.month_lstm(m)

        w_f = self.week_temp_attn(w_o)
        m_f  = self.month_temp_attn(m_o)
        kv_past = torch.cat([w_f.unsqueeze(1), m_f.unsqueeze(1)], dim=1) # (B, 2, d_model)

        
        c_o, _ = self.channel_mha(query=d_f, key=kv_past, value=kv_past)

        # 3. Skip Connection 
        deviation = d_f - c_o
        base_feat = d_f + c_o
        final_input = torch.cat([base_feat, deviation], dim=-1).squeeze(1)
        return self.fc(final_input)


    def forward_attention(self, x):
        d, w, m = x[:, -1:], x[:, -5:], x[:, -22:]

        d_f = self.day_proj(d) # (B, 1, d_model)
        w_o, _ = self.week_lstm(w)
        m_o, _ = self.month_lstm(m)

        w_f = self.week_temp_attn(w_o)
        m_f  = self.month_temp_attn(m_o)


        wegihts_w, v_w = self.week_temp_attn.forward_attention(w_o)
        wegihts_m, v_m  = self.month_temp_attn.forward_attention(m_o)

        kv_past = torch.cat([ w_f.unsqueeze(1), m_f.unsqueeze(1)], dim=1) # (B, 2, d_model)

        c_o, c_a = self.channel_mha(query=d_f, key=kv_past, value=kv_past)

        # 3. Skip Connection 
        deviation = d_f - c_o
        base_feat = d_f + c_o
        final_input = torch.cat([base_feat, deviation], dim=-1).squeeze(1)
        
        return wegihts_w, v_w, wegihts_m, v_m, c_a




class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.configs = configs
        self.lstm = nn.LSTM(input_size=configs.input_dim,
                            hidden_size=configs.lstm_hidden_dim,
                            num_layers=configs.lstm_layers,
                            bias=True,
                            batch_first=True,
                            dropout=configs.dropout)

        self.distribution_output = nn.Linear(configs.lstm_hidden_dim, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        output = self.distribution_output(output)
        return output 



        
class BILSTM(nn.Module):
    def __init__(self, configs):
        super(BILSTM, self).__init__()
        self.configs = configs
        self.lstm = nn.LSTM(input_size=configs.input_dim,
                            hidden_size=configs.lstm_hidden_dim,
                            num_layers=configs.lstm_layers,
                            bias=True,
                            batch_first=True,
                            bidirectional=True,
                            dropout=configs.dropout)

        self.distribution_output = nn.Linear(configs.lstm_hidden_dim * 2, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = torch.cat([hidden[-2], hidden[-1]], dim=1)
        output = self.distribution_output(output)
        return output 



class GRU(nn.Module):
    def __init__(self, configs):
        super(GRU, self).__init__()
        self.configs = configs
        self.gru = nn.GRU(input_size=configs.input_dim,
                          hidden_size=configs.lstm_hidden_dim,
                          num_layers=configs.lstm_layers,
                          bias=True,
                          batch_first=True,
                          bidirectional=False,
                          dropout=configs.dropout)

        self.distribution_output = nn.Linear(configs.lstm_hidden_dim, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        output = output[:, -1, :]
        output = self.distribution_output(output)

        return output 




def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    
    
    if is_best:
        torch.save(state, os.path.join(checkpoint, f'ins_{ins_name}_best.pth.tar'))
        logger.info('Best checkpoint copied to best.pth.tar')

def load_checkpoint(checkpoint, model, optimizer = None):
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
        
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
        
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
        optimizer.param_groups[0]['capturable'] = True

    return checkpoint

    
def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        json.dump(d, f, indent=4)


def load_json(path) :
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data


