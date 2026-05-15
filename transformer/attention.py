import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        weights = dropout(weights)

    return torch.matmul(weights, value), weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention from Section 3.2.2 of the paper.
    Splits d_model into h heads, runs attention in parallel, concatenates.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # linear projections for Q, K, V and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # project and reshape: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # run attention on all heads in parallel
        # masks are already shaped for broadcasting: (batch, 1, 1, seq) or (1, 1, seq, seq)
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # concat heads: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(attn_output)
