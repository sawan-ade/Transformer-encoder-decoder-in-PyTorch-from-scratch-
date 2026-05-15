import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    """
    Full Transformer model (Vaswani et al., 2017).

    Encoder processes the source sequence.
    Decoder generates the target sequence one token at a time.
    A final linear layer projects decoder output to vocabulary logits.
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff, n_layers, dropout, max_len)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        logits = self.output_projection(dec_out)
        return logits
