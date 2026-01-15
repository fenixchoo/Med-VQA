import torch
import torch.nn as nn

class QFormer(nn.Module):
    def __init__(self, vision_dim=768, hidden_dim=768, num_queries=32, layers=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        self.proj = nn.Linear(vision_dim, hidden_dim)

        self.attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layers)
        ])

    def forward(self, vision_tokens):
        x = self.proj(vision_tokens)
        q = self.queries.expand(x.size(0), -1, -1)
        for attn, norm in zip(self.attn, self.norm):
            out, _ = attn(q, x, x)
            q = norm(q + out)
        return q
