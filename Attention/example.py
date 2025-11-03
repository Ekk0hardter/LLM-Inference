import torch
import math



class MHA(nn.Module):
    def __init__(self, d_model, head_num, head_dim):
        self.d_model = d_model
        self.head_num = head_num
        self.head_dim = head_dim

        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.softmax = F.softmax(dim = -1)

        self.output_layer  = nn.Linear(d_model, d_model)

    
    def forward(self, q, k, v):
        # q [B, seq_len, d_model]
        B, seq_len, d_model = q.shape
        Q = self.q_proj(q)
        K = self.k_proj(k)
        V=  self.v_proj(v)

        # [B, head_num, seq_len, head_dim]
        Q = Q.view(B, -1, self.head_num, self.head_dim).transpose(1,2)
        K = K.view(B, -1, self.head_num, self.head_dim).transpose(1,2)
        V = V.view(B, -1, self.head_num, self.head_dim).transpose(1,2)





        result1 = matmul(Q, K.transpose(-2,-1)) / sqrt(head_dim)  # [B, head_num, seq_len, seq_len]
        attn = self.softmax(result1)                        # [B, head_num, seq_len, seq_len]
        result3 = matmul(attn, V)                           # [B, head_num, seq_len, head_dim]

        O = result3.transpose(1,2).view(B, seq_len, -1)

        output = self.output_layer(O)

        return attn, output





