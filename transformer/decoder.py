import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .embeddings import TransformerEmbedding


class DecoderLayer(nn.Module):
    """
    Single decoder layer:
      x -> Masked Self-Attention -> Add & Norm
        -> Cross-Attention (with encoder output) -> Add & Norm
        -> FeedForward -> Add & Norm
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # masked self-attention (decoder attends to itself, causally)
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # cross-attention (decoder attends to encoder output)
        attn_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout2(attn_out))

        # feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x


class Decoder(nn.Module):
    """
    Full decoder: embedding + N stacked decoder layers.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

    def forward(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x
