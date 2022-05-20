import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    整个BertEncoder由多个BertLayer堆叠形成；
    而BertLayer又是由BertOutput、BertIntermediate和BertAttention这3个部分组成；
    同时BertAttention是由BertSelfAttention和BertSelfOutput所构成。

"""
'''Bert Attention实现'''
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.multi_head_attention = MyMultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout
        )

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
            params 
                query: [tgt_len, batch_size, hidden_size]
                key: [src_len, batch_size, hidden_size]
                value: [src_len, batch_size, hidden_size]
                key_padding_mask: [batch_size, src_len]

            return
                attn_output: [tgt_len, batch_size, hidden_size]
                attn_output_weights: [batch_size, tgt_len, src_len]
        
        """
        return self.multi_head_attention(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self_attn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, key_padding_mask=None):
        attn_output = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            attn_mask=None,
            key_padding_mask=key_padding_mask
        )
        attn_output = self.output(attn_output[0], hidden_states)
        return attn_output


'''Bert Layer实现'''
class BertIntermediate(nn.Module):
    def __init__(self,):
        super(BertIntermediate, self).__init__()

    def forward(self, ):
        pass

class BertOutput(nn.Module):
    def __init__(self,):
        super(BertOutput, self).__init__()
    def forward(self, ):
        pass

class BertLayer(nn.Module):
    def __init__(self, ):
        super(BertLayer, self).__init__()

    def forward(self, ):
        pass

'''堆叠多个layer组成Bert Encoder'''
class BertEncoder(nn.Module):
    def __init__(self, ):
        super(BertEncoder, self).__init__()

    def forward(self, ):
        pass
    
'''输出到下游任务需要预处理'''
class BertPooler(nn.Module):
    def __init__(self, ):
        super(BertPooler, self).__init__()
    
    def forward(self, ):
        pass

'''Bert模型主体结构'''
class BertModel(nn.Module):
    def __init__(self, ):
        super(BertModel, self).__init__()

    def forward(self, ):
        pass

