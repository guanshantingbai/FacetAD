"""Image-conditioned bounded abnormal prototype used by all three R1 members."""

import torch
from torch import nn
from torch.nn import functional as F


class ReaderGate(nn.Module):
    def __init__(self, width=768, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.empty(1, 1, width))
        self.ln = nn.LayerNorm(width)
        self.mha = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ln_h = nn.LayerNorm(width)
        self.P = nn.Linear(width, width)

    def forward(self, patches, abnormal_text):
        query = self.q.expand(len(patches), -1, -1)
        attended, _ = self.mha(
            self.ln(query),
            self.ln(patches),
            self.ln(patches),
            need_weights=True,
            average_attn_weights=True,
        )
        delta = self.P(self.ln_h(query + attended)).squeeze(1)
        bounded = 0.025 * delta / (delta.norm(dim=-1, keepdim=True) + 1e-6)
        return F.normalize(abnormal_text.float() + bounded, dim=-1)
