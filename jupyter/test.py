import torch
from torch import nn
import math

class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, max_relative_position=50):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position  # 最大相对距离k
        
        vocab_size = 2 * max_relative_position + 1  # 总共2k+1个相对位置
        # 创建两个嵌入层，分别用于K和V
        self.relative_position_k = nn.Embedding(vocab_size, d_model)
        self.relative_position_v = nn.Embedding(vocab_size, d_model)
    
    def get_relative_positions(self, seq_len):
        # 创建位置索引向量[0,1,2,...,seq_len-1]
        range_vec = torch.arange(seq_len)
        # 扩展为矩阵
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        # 计算相对位置j-i
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # 裁剪到最大相对位置[-k, k]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # 转换为正数索引[0, 2k]，用于Embedding查找
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat

    def forward(self, seq_len):
        relative_positions = self.get_relative_positions(seq_len) # shape: [seq_len, seq_len]
        
        # 通过嵌入层获取相对位置编码
        relative_position_k_emb = self.relative_position_k(relative_positions) # shape: [seq_len, seq_len, d_model]
        relative_position_v_emb = self.relative_position_v(relative_positions) # shape: [seq_len, seq_len, d_model]
        
        return relative_position_k_emb, relative_position_v_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, max_seq_len=16, dropout_ratio=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.rpe = RelativePositionEncoding(self.d_k)
         
    def forward(self, x, causal_mask):
        batch_size, seq_len , _ = x.shape
        
        Q = self.linear_q(x)    # shape: [batch_size, seq_len, d_model]
        K = self.linear_k(x)
        V = self.linear_v(x)
        
        # reshape并交换seq_len和n_heads两个维度
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # shape: [batch_size, n_heads, seq_len, d_k]
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 获取相对位置编码
        R_k, R_v = self.rpe(seq_len)
        
        # 扩展维度以匹配batch和heads
        R_k = R_k.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len, d_k]
        R_v = R_v.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len, d_k]
        
        # 计算attention scores (包含相对位置信息)
        content_scores = torch.matmul(Q, K.transpose(-1, -2))    # shape: [batch_size, n_heads, seq_len, seq_len]
        position_scores = torch.matmul(Q.unsqueeze(-2), R_k.transpose(-1, -2)).squeeze(-2)
        
        scores = (content_scores + position_scores) / math.sqrt(self.d_k)
        
        if causal_mask is not None:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(causal_mask, -1e9)    # shape: [batch_size, n_heads, seq_len, seq_len]
        
        probs = torch.softmax(scores, dim = -1)
        probs = self.dropout(probs)    # shape: [batch_size, n_heads, seq_len, seq_len]
        
        # 计算attention输出 (包含相对位置信息)
        content_attention = torch.matmul(probs, V)  # shape: [batch_size, n_heads, seq_len, d_k]
        
        position_attention = torch.matmul(probs.unsqueeze(-2), R_v).squeeze(-2)
        attention = content_attention + position_attention

        attention = attention.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_k)  # shape: [batch_size, seq_len, d_model]
        attention = self.linear_o(attention)
        return attention     

n_heads = 4
batch_size = 8
seq_len = 16
d_model = 128

# 产生一个 shape=(seq_len, seq_len) 的上三角矩阵
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

input = torch.randn(batch_size, seq_len, d_model)

MHA = MultiHeadAttention(n_heads, d_model)
output = MHA(input, causal_mask) 