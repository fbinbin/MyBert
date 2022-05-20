import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id=0, initializer_range=0.02):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self._reset_parameters(initializer_range)
    
    def forward(self, input_ids):
        """
            param:  input_ids.shape [input_ids_len, batch_size]
            return  [input_ids_len, batch_size, hidden_size]

        """
        return self.token_embedding(input_ids) # [input_ids_len, batch_size, hidden_size]

    def _reset_parameters(self, initializer_range):
        for p in self.token_embedding.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, mean=0.0, std=initializer_range)



class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_position_embedding=512):
        """
            BERT的位置编码和Transformer不同，Transformer使用根据公式生成固定的位置编码，
            BERT使用的可学习参数的位置编码，从结构上看和token embedding相似，区别是该embeddding编码位置信息

        """
        super(PositionalEmbedding, self).__init__()
        self.positional_embedding = nn.Embedding(num_embeddings=max_position_embedding, embedding_dim=hidden_size)

    def forward(self, position_ids):
        """
            params position_ids [1, position_ids_len]
            return [position_ids_len, 1, hidden_size]

        """
        return self.positional_embedding(position_ids).tanspose(0, 1)
        

class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range):
        super(SegmentEmbedding, self).__init__()
        self.segment_embedding = nn.Embedding(num_embeddings=type_vocab_size, embedding_dim=hidden_size)

    def forward(self, token_type_ids):
        """
            params token_type_ids [token_type_ids_len, batch_size]
            return [token_type_ids_len, batch_size, hidden_size]

        """
        return self.segment_embedding(token_type_ids)

class BertEmbedding(nn.Module):
    def __init__(self, config):
        super(BertEmbedding, self).__init__()
        """将上述三种embedding加起来得到input embedding"""
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            pad_token_id=config.pad_token_id,
            initializer_range=config.initializer_range
        )   # shape [src_len, batch_size, hidden_size]

        self.positional_embedding = PositionalEmbedding(
            hidden_size=config.hidden_size,
            max_position_embedding=config.max_position_embedding
        )   # shape [src_len, 1, hidden_size]

        self.segment_embedding = SegmentEmbedding(
            type_vocab_size=config.type_vocab_size, 
            hidden_size=config.hidden_size, 
            initializer_range=config.initializer_range
        )   # [src_len, batch_size, hidden_size]

        self.norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # position_ids [1, max_position_embedding]
        self.register_buffer("position_ids", torch.arange(config.max_position_embedding).expand(1, -1))


    def forward(self, input_ids=None, position_ids=None, token_type_ids=None):
        
        src_len = input_ids.size(0)
        token_embedding = self.token_embedding(input_ids)

        if position_ids is None:
            position_ids = self.position_ids[:, :src_len]
        positional_embedding = self.positional_embedding(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        segment_embedding = self.segment_embedding(token_type_ids)

        embeddings = token_embedding + positional_embedding + segment_embedding
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings