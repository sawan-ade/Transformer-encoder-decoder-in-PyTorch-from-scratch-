cimport torch
import torch.nn as nn
from transformer import Transformer, create_padding_mask, create_causal_mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # --- hyperparameters (scaled down for demo) ---
    src_vocab_size = 50
    tgt_vocab_size = 50
    d_model = 128
    n_heads = 4
    n_layers = 3
    d_ff = 512
    dropout = 0.1
    pad_idx = 0
    max_len = 100
    batch_size = 4
    src_seq_len = 12
    tgt_seq_len = 10
    num_epochs = 30
    lr = 3e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- build model ---
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len
    ).to(device)

    total_params = count_parameters(model)
    print(f"Transformer built successfully!")
    print(f"  d_model={d_model}, heads={n_heads}, layers={n_layers}, d_ff={d_ff}")
    print(f"  Total trainable parameters: {total_params:,}\n")

    # --- toy data: learn to copy a sequence ---
    # source:  [5, 8, 3, 12, 7, 9, 2, 6, 0, 0, 0, 0]  (padded)
    # target:  [1, 5, 8, 3, 12, 7, 9, 2, 6, 2]          (shifted, 1=BOS, 2=EOS)
    print("=" * 60)
    print("Task: Learn to copy source tokens into target")
    print("=" * 60)

    torch.manual_seed(42)
    src = torch.randint(3, src_vocab_size, (batch_size, src_seq_len)).to(device)
    src[:, -3:] = pad_idx  # add some padding

    # target = BOS + source content + EOS (trimmed to tgt_seq_len)
    bos = torch.ones(batch_size, 1, dtype=torch.long, device=device)  # BOS=1
    eos = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)  # EOS=2
    content = src[:, :tgt_seq_len - 2]
    tgt_full = torch.cat([bos, content, eos], dim=1)

    tgt_input = tgt_full[:, :-1]   # decoder input (remove last token)
    tgt_output = tgt_full[:, 1:]   # expected output (remove BOS)

    print(f"\nSample source:  {src[0].tolist()}")
    print(f"Sample target:  {tgt_full[0].tolist()}")
    print(f"Decoder input:  {tgt_input[0].tolist()}")
    print(f"Expected out:   {tgt_output[0].tolist()}\n")

    # --- masks ---
    src_mask = create_padding_mask(src, pad_idx).to(device)
    tgt_mask = create_causal_mask(tgt_input.size(1)).to(device)

    # --- training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print("Training...")
    print("-" * 40)
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        logits = model(src, tgt_input, src_mask, tgt_mask)

        # reshape for cross entropy: (batch * seq, vocab) vs (batch * seq)
        loss = criterion(
            logits.reshape(-1, tgt_vocab_size),
            tgt_output.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{num_epochs}  |  Loss: {loss.item():.4f}")

    # --- inference ---
    print("\n" + "=" * 60)
    print("Inference")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt_input, src_mask, tgt_mask)
        preds = logits.argmax(dim=-1)

    print(f"\nSource:     {src[0].tolist()}")
    print(f"Expected:   {tgt_output[0].tolist()}")
    print(f"Predicted:  {preds[0].tolist()}")

    match = (preds[0] == tgt_output[0]).sum().item()
    total = tgt_output.size(1)
    print(f"\nAccuracy: {match}/{total} ({100 * match / total:.1f}%)")

    # --- shape walkthrough ---
    print("\n" + "=" * 60)
    print("Tensor Shape Walkthrough")
    print("=" * 60)
    print(f"  src shape:          {src.shape}")
    print(f"  tgt_input shape:    {tgt_input.shape}")
    print(f"  src_mask shape:     {src_mask.shape}")
    print(f"  tgt_mask shape:     {tgt_mask.shape}")
    print(f"  logits shape:       {logits.shape}")
    print(f"  predictions shape:  {preds.shape}")
    print()


if __name__ == "__main__":
    main()
