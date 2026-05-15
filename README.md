# Transformer From Scratch

A clean PyTorch implementation of the Transformer architecture from the paper  
**"Attention Is All You Need"** (Vaswani et al., 2017).

Every component — multi-head attention, positional encoding, encoder, decoder — is written from scratch. No `nn.Transformer` shortcuts.

---

## Architecture

```
                        ┌─────────────────────┐
                        │   Output Probabilities│
                        │     (Linear + Softmax)│
                        └──────────┬────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │          DECODER             │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │    Decoder Layer × N    │  │
                    │  │                        │  │
                    │  │  ┌──────────────────┐  │  │
                    │  │  │  Masked Self-Attn │  │  │
                    │  │  │  + Add & Norm     │  │  │
                    │  │  └──────────────────┘  │  │
                    │  │  ┌──────────────────┐  │  │
                    │  │  │  Cross-Attention  │◄├──├──── Encoder Output
                    │  │  │  + Add & Norm     │  │  │
                    │  │  └──────────────────┘  │  │
                    │  │  ┌──────────────────┐  │  │
                    │  │  │  Feed-Forward     │  │  │
                    │  │  │  + Add & Norm     │  │  │
                    │  │  └──────────────────┘  │  │
                    │  └────────────────────────┘  │
                    │                              │
                    │  Token Emb + Positional Enc  │
                    └──────────────┬───────────────┘
                                   │
                              Target Tokens


                    ┌──────────────┴──────────────┐
                    │          ENCODER             │
                    │                              │
                    │  ┌────────────────────────┐  │
                    │  │    Encoder Layer × N    │  │
                    │  │                        │  │
                    │  │  ┌──────────────────┐  │  │
                    │  │  │  Self-Attention   │  │  │
                    │  │  │  + Add & Norm     │  │  │
                    │  │  └──────────────────┘  │  │
                    │  │  ┌──────────────────┐  │  │
                    │  │  │  Feed-Forward     │  │  │
                    │  │  │  + Add & Norm     │  │  │
                    │  │  └──────────────────┘  │  │
                    │  └────────────────────────┘  │
                    │                              │
                    │  Token Emb + Positional Enc  │
                    └──────────────┬───────────────┘
                                   │
                             Source Tokens
```

---

## Project Structure

```
transformer/
├── __init__.py          # package exports
├── attention.py         # scaled dot-product + multi-head attention
├── embeddings.py        # token embedding + sinusoidal positional encoding
├── feed_forward.py      # position-wise FFN
├── masks.py             # padding and causal masks
├── encoder.py           # encoder layer + encoder stack
├── decoder.py           # decoder layer + decoder stack
└── transformer.py       # full model

demo.py                  # toy training demo
requirements.txt         # dependencies
report/                  # LaTeX report + PDF
```

---

## Key Equations

**Scaled Dot-Product Attention:**

```
Attention(Q, K, V) = softmax(Q K^T / √d_k) V
```

**Multi-Head Attention:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

**Positional Encoding:**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Feed-Forward Network:**

```
FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
```

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the demo

```bash
python demo.py
```

This trains a small Transformer on a toy copy task and prints loss, predictions, and tensor shapes.

### Use in your own code

```python
from transformer import Transformer, create_padding_mask, create_causal_mask

model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    n_heads=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1
)

# forward pass
logits = model(src_tokens, tgt_tokens, src_mask, tgt_mask)
```

---

## Default Hyperparameters (from the paper)

| Parameter | Value |
|-----------|-------|
| `d_model` | 512 |
| `n_heads` | 8 |
| `n_layers` | 6 |
| `d_ff` | 2048 |
| `dropout` | 0.1 |

---

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). **Attention Is All You Need.** *Advances in Neural Information Processing Systems (NeurIPS).*  
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## License

This is an educational implementation. Use it however you like.
