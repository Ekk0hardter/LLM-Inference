import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()    # 调用父类 nn.Module 的构造函数，让 父类的初始化逻辑生效
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads    # 每个头的维度
        
        # Q, K, V 的线性变换
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # 输出线性层
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [batch_size, seq_len, embed_dim]
        mask: [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 线性映射
        # nn.Linear 会 保留 batch_size 和 seq_len 的维度，只对最后一维做操作
        Q = self.q_linear(query)  # [B, L, D]
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # 分头
        # [Batch, Seq_len, Embed_dim] -> [Batch, Seq_len, Num_heads, Head_dim] -> [Batch, Num_heads, Seq_len, Head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)  # [Batch, Head, seq_len, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [Batch, Head, seq_len, seq_len]
        
        # 找到mask中等于0的位置；把对应位置的scores值替换成-inf
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) 
        
        attn = F.softmax(scores, dim=-1)  # [Batch, Head, seq_len, seq_len]
        attn = self.dropout(attn)         
        
        out = torch.matmul(attn, V)  # [Batch, Head, seq_len, head_dim]
        
        # 合并多头
        # [Batch, Head, seq_len, head_dim] -> [Batch, seq_len, Head, head_dim] -> [Batch, seq_len, Embed_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # [Batch, seq_len, Embed_dim]
        out = self.out_linear(out)   # [Batch, seq_len, Embed_dim]
        
        return out, attn

# 测试
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_heads = 4
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    attn_layer = MultiHeadAttention(embed_dim, num_heads)
    
    out, attn = attn_layer(x, x, x)
    print("Output shape:", out.shape)  # [2, 5, 16]
    print("Attention shape:", attn.shape)  # [2, 4, 5, 5]
