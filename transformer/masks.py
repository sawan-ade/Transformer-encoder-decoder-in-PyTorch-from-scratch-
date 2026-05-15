import torch


def create_padding_mask(seq, pad_idx=0):
    """
    Returns a mask that is 1 where the token is NOT padding, 0 where it is.
    Shape: (batch, 1, 1, seq_len) — broadcasts over heads and query positions.
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_causal_mask(size):
    """
    Returns a lower-triangular mask of shape (1, 1, size, size).
    Prevents the decoder from attending to future tokens.
    """
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask
