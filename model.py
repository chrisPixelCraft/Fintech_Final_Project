import torch
import torch.nn as nn
from typing import List

class GRN(nn.Module):
    """Gated Residual Network"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.linear1(x))
        out = self.linear2(out)
        gate = torch.sigmoid(self.gate(out))
        return gate * out + (1 - gate) * x

class VSN(nn.Module):
    """Variable Selection Network"""
    def __init__(self, input_sizes: List[int], hidden_size: int):
        super().__init__()
        self.transformers = nn.ModuleList([
            nn.Linear(size, hidden_size) for size in input_sizes
        ])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        transformed = [t(x) for t, x in zip(self.transformers, inputs)]
        weights = self.softmax(torch.stack(transformed, dim=-1))
        return torch.sum(weights * torch.stack(inputs, dim=-1), dim=-1)

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_features: int, hidden_size: int = 64, num_heads: int = 4):
        super().__init__()
        self.vsn = VSN([num_features]*4, hidden_size)
        self.grn = GRN(hidden_size, hidden_size*2, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.quantile_proj = nn.Linear(hidden_size, 3)  # For 0.1, 0.5, 0.9 quantiles

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch_size, num_features)
        selected = self.vsn([x[:,:,i] for i in range(x.size(2))])
        processed = self.grn(selected)
        attn_out, _ = self.attention(processed, processed, processed)
        return self.quantile_proj(attn_out)

