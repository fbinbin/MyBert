import torch
import torch.nn as nn
import torch.nn.functional as F


class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        '''
            本类在已知Q，K，V的情况下，实现多头注意力的计算。
            可以用于encoder、decoder、以及encoder-decoder交互三个部分的多头注意力计算
            注意力机制计算完成后，多头注意力机制输出为[src_len, embed_dim]

            embed_dim:词嵌入维度,等于d_model,论文默认512 
            num_heads:多头数量，论文默认8
            bias:对最后多头注意力输出进行线性变换的时候是否使用偏置

        '''

        self.embed_dim = embed_dim  # embed_dim = d_model
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qdim = self.head_dim
        self.kdim = self.head_dim
        self.vdim = self.head_dim

        self.dropout = dropout
        
        # 需要检查q,k,v的维度是否等于d_model/num_head
        assert self.head_dim * self.num_heads == self.embed_dim

        # 注意，第二个维度是因为embed_dim = qdim * num_heads = kdim * num_heads = vdim * num_heads
        # 通过下面的方式同时初始化num_heads个变换矩阵
        self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))   # 同时初始化num_heads个W_q
        self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))   # 同时初始化num_heads个W_k
        self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))   # 同时初始化num_heads个W_v

        self.out_porj = nn.Linear(embed_dim, embed_dim, bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):

        '''
            编码时的query、key、value都是同一个输入,encoder输入的token经过线性变换后得到
            解码时，输入的部分也都是同一个输入，decoder输入的token经过线性变换后得到
            解码和编码交互时key、value来自于最后一层encoder的memory，query来自于decoder块内masked multi-head attention输出的带解码tgt

            tgt_len:target sequence length
            src_len:sourece sequence length

            param:
                注意q、k、v的第二维度是batch_size，因为decoder在prediction时仍然是串行的，计算需要按顺序，所以把batch顺序放在第二维，以便可以以整个batch输入
                query:      [tgt_len, batch_size, embed_dim]
                key:        [src_len, batch_size, embed_dim]
                value:      [src_len, batch_size, embed_dim]
                attn_mask:  [tgt_len, src_len] or[num_heads*batch_size, tgt_len, src_len]
                key_padding_mask: [batch_size, src_len]

            output:
                attn_output: [tgt_len, batch_size, embed_dim(vdim*num_heads)]
                attn_output_weights: [batch_size, tgt_len, src_len]


        '''
        return multi_head_attention_forward(
            query, key, value, 
            self.num_heads,
            self.dropout,
            self.out_porj.weight,
            self.out_porj.bias,
            q_proj_weight = self.q_proj_weight,
            k_proj_weight = self.k_proj_weight,
            v_proj_weight = self.v_proj_weight,
            attn_mask = attn_mask, key_padding_mask = key_padding_mask,
            training=self.training
        )
    
def multi_head_attention_forward(
    query,      # [tgt_len, batch_size, embed_dim]
    key,        # [src_len, batch_size, embed_dim]
    value,      # [src_len, batch_size, embed_dim]
    num_heads,  
    dropout,
    out_porj_weight,        #[embed_dim=vdim*num_heads, embed_dim]
    out_porj_bias,          #[embed_dim, 1] 
    q_proj_weight=None,     #[embed_dim, qdim*num_heads]
    k_proj_weight=None,     #[embed_dim, kdim*num_heads]
    v_proj_weight=None,     #[embed_dim, vdim*num_heads]
    attn_mask=None,         #[tgt_len, src_len]
    key_padding_mask=None,  #[batch_size, src_len/tgt_len]
    training=True            
):
    '''
        整个multi-head attention计算包括如下过程：
        1计算出Q、K、V三个
        2缩放以及attn_mask维度的判断
        3计算得到注意力权重矩阵
        4进行掩码相关操作
        5拼接z得到多头注意力层输出
    
    '''

    ####################阶段一：计算Q、K、V矩阵##########################################################################
    
    q = F.linear(query, q_proj_weight)  # [tgt_len, batch_size, embed_dim]X[embed_dim, qdim*num_heads] = [tgt_len, batch_size, qdim*num_heads]
    k = F.linear(key, k_proj_weight)    # [src_len, batch_size, embed_dim]X[embed_dim, kdim*num_heads] = [src_len, batch_size, kdim*num_heads]
    v = F.linear(value, v_proj_weight)  # [src_len, batch_size, embed_dim]X[embed_dim, vdim*num_heads] = [src_len, batch_size, vdim*num_heads]


    ##################阶段二：缩放,attn_mask维度判断################################
    #如果attn_mask是二维转化成三维度，再检查维度，三维直接检查
    tgt_len, batch_size,  embed_dim = query.size()
    src_len = key.size(0)

    dk = embed_dim // num_heads
    head_dim = dk
    q = q * (float(dk))**(-0.5)

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, tgt_len, src_len]:
                raise RuntimeError("attention mask维度错误1")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [batch_size*num_heads, tgt_len, src_len]:
                raise RuntimeError("attention mask维度错误2")

    #################阶段三：计算得到注意力权重矩阵################################################################################

    #将q、k、v拆分多头，再分别计算
    q = q.contiguous().view(tgt_len, batch_size * num_heads, head_dim).transpose(0, 1)   # [batch_size*num_heads, tgt_len, kdim/qdim]
    k = k.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)   # [batch_size*num_heads, src_len, kdim]      
    v = v.contiguous().view(-1, batch_size * num_heads, head_dim).transpose(0, 1)   # [batch_szie*num_heads, src_len, vdim]

    attn_weights_matrix = torch.bmm(q, k.transpose(1,2))   # [batch_size*num_heads, tgt_len, src_len]
    #print(attn_weights_matrix.size())
    ####################阶段四：进行掩码相关操作########################################################################
    if attn_mask is not None:
        attn_weights_matrix += attn_mask
    if key_padding_mask is not None:
        # 先将attn_weight拉伸成[batch_size, num_heads, tgt_len, src_len]，再做mask
        attn_weights_matrix = attn_weights_matrix.view(batch_size, num_heads, tgt_len, src_len)
        attn_weights_matrix.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2), #[batch_size, 1, 1 src_len]
            float('-inf')
        )
        attn_weights_matrix = attn_weights_matrix.view(batch_size*num_heads, tgt_len, src_len)
        #print(attn_weights_matrix.size())

    ####################阶段五：拼接z得到多头注意力输出###########################################################################
    attn_weights_matrix = F.softmax(attn_weights_matrix, dim=-1)
    attn_weights_matrix = F.dropout(attn_weights_matrix, p=dropout, training=training)

    #print(attn_weights_matrix.size())
    # [batch_size*num_heads, tgt_len, src_len] X [batch_size*num_heads, src_len, vdim]
    # = [batch_size*num_heads, tgt_len, vdim]
    attn_output = torch.bmm(attn_weights_matrix, v) 
    attn_output = attn_output.transpose(0,1).contiguous().view(tgt_len, batch_size, embed_dim)
    attn_output = F.linear(attn_output, out_porj_weight, out_porj_bias)
    
    attn_output_weights = attn_weights_matrix.view(num_heads, batch_size, tgt_len, src_len)
    return attn_output, attn_output_weights.sum(dim=0)/num_heads




if __name__ == '__main__':
    '''
        单个模块测试，测试multi-head attention的计算是否正确

    '''
    src_len = 5
    batch_size = 2
    d_model = 32
    num_head = 1
    src = torch.rand((src_len, batch_size, d_model))

    src_key_padding_mask = torch.tensor(
        [[True, True, True, False, False],
        [True, True, True, True, False]]
    )
    my_mh = MyMultiheadAttention(embed_dim=d_model, num_heads=num_head)
    r = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)