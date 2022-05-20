import torch
import torch.nn as nn
import torch.nn.functional as F

import os

class Config():
    def __init__(self):
        self.vocab_size = 
        self.hidden_size = 
        self.pad_token_id = 
        self.initializer_range = 
        self.max_position_embedding = 
        self.type_vocab_size = 
        self.dropout = 
        self.hidden_dropout_prob = 