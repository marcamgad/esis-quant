"""
Gated Residual Network (GRN) and Gated Linear Unit (GLU) -- the core
building block of the Temporal Fusion Transformer (Lim et al. 2021, eq. 2-5),
from scratch.

GLU(gamma) = (W4 gamma + b4) . sigmoid(W5 gamma + b5)   -- elementwise gate
    lets the network suppress (gate toward 0) any sub-component that isn't
    useful for a given input.

GRN(a, c) with optional context c:
    eta2 = ELU(W2 a + W3 c + b2)
    eta1 = W1 eta2 + b1
    GRN  = LayerNorm(a_proj + GLU(eta1))
where a_proj is a (skip) projection of the input to output_size when the two
differ. The residual skip + gated update means a GRN can learn to apply as
little or as much nonlinear processing as the data needs -- important when
the same architecture must handle inputs of very different signal content.
"""

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """GLU(x) = (W4 x + b4) * sigmoid(W5 x + b5)."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    """GRN(a, c) = LayerNorm(skip(a) + GLU(W1 ELU(W2 a + W3 c + b2) + b1)).

    input_size : dim of the primary input a.
    hidden_size: inner ELU width.
    output_size: output dim (defaults to input_size). A linear skip
                 projection is added only when input_size != output_size.
    context_size: dim of the optional static context c (None = no context).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int = None,
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        output_size = output_size or input_size
        self.output_size = output_size

        self.fc2 = nn.Linear(input_size, hidden_size)          # W2 a
        self.context = nn.Linear(context_size, hidden_size, bias=False) if context_size else None  # W3 c
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(hidden_size, hidden_size)          # W1
        self.glu = GatedLinearUnit(hidden_size, output_size, dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, a: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        residual = a if self.skip is None else self.skip(a)
        eta2 = self.fc2(a)
        if self.context is not None and context is not None:
            # context is static (no time dim); broadcast over any middle dims of a
            while context.dim() < eta2.dim():
                context = context.unsqueeze(1)
            eta2 = eta2 + self.context(context)
        eta2 = self.elu(eta2)
        eta1 = self.fc1(eta2)
        return self.layer_norm(residual + self.glu(eta1))
