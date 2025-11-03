import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_query_heads, num_kv_heads):
        super(GroupedQueryAttention, self).__init__()
        assert embed_dim % num_query_heads == 0, "embed_dim must be divisible by num_query_heads"
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_query_heads
        self.group_size = num_query_heads // num_kv_heads

        # Q, K, V 的线性层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim * num_kv_heads // num_query_heads)
        self.v_proj = nn.Linear(embed_dim, embed_dim * num_kv_heads // num_query_heads)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [batch_size, seq_len, embed_dim]
        mask: [batch_size, seq_len] 或 [batch_size, 1, 1, seq_len]
        """
        Batch, Seq_len, _ = query.size()
        device = query.device

        # 线性投影
        Q = self.q_proj(query).view(Batch, Seq_len, self.num_query_heads, self.head_dim)
        K = self.k_proj(key).view(  Batch, Seq_len, self.num_kv_heads,    self.head_dim)
        V = self.v_proj(value).view(Batch, Seq_len, self.num_kv_heads,    self.head_dim)

        # 形状转为 [Batch, num_heads, L, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # GQA: 复制 K/V 让每组 query 共享同一个 KV
        if self.group_size > 1:
            # K shape: [Batch, num_kv_heads, Seq_len, head_dim] -> [Batch, num_query_heads, Seq_len, head_dim] 
            # repeat_interleave 按照 dim=1 复制 group_size 次
            K = K.repeat_interleave(self.group_size, dim=1)
            V = V.repeat_interleave(self.group_size, dim=1)

        # 注意力分数计算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # 拼接并线性输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        out = self.out_proj(attn_output)

        return out, attn_weights


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = 512
    num_query_heads = 16
    num_kv_heads = 4  # 每4个query头共享1个K/V

    gqa = GroupedQueryAttention(embed_dim, num_query_heads, num_kv_heads).to(device)

    x = torch.randn(2, 10, embed_dim).to(device)
    out, attn = gqa(x, x, x)

    print("Output shape:", out.shape)
    print("Attention shape:", attn.shape)
