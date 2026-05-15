import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (Section 3.3).
    FFN(x) = ReLU(x W1 + b1) W2 + b2
    """

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
