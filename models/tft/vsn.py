"""
Variable Selection Network (VSN), from scratch (Lim et al. 2021, eq. 6-8).

Given `n_vars` input variables (each first embedded to hidden_size), the VSN
learns a softmax weight per variable per sample/time-step -- so the model can
attend to the informative inputs and down-weight noise, and those weights are
directly interpretable as variable-importance scores.

  * each variable v is transformed by its OWN GRN -> xi_v (hidden_size)
  * the flattened stack of all variable embeddings (+ optional static
    context) is passed through one GRN then softmax -> weights (n_vars,)
  * output = sum_v weights_v * xi_v          (hidden_size)

Returns (selected, weights) so the caller can log/plot the selection weights.
"""

import torch
import torch.nn as nn

from models.tft.grn import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """Select + combine `n_vars` scalar-per-variable inputs into one
    hidden_size vector, with interpretable per-variable softmax weights.

    Input to forward: x of shape (..., n_vars) where each of the last-dim
    entries is one variable's value at that (batch[, time]) position. Each
    variable is embedded independently to hidden_size before selection.
    """

    def __init__(self, n_vars: int, hidden_size: int, dropout: float = 0.1,
                 context_size: int = None):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_size = hidden_size

        # per-variable embedding of a scalar -> hidden_size, then a GRN
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_size, hidden_size, dropout) for _ in range(n_vars)
        ])
        # weight network over the flattened raw inputs (+ optional static context)
        self.flattened_grn = GatedResidualNetwork(
            n_vars, hidden_size, n_vars, dropout, context_size=context_size
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        # x: (..., n_vars). Build selection weights from the raw flattened vars.
        weights = self.softmax(self.flattened_grn(x, context))  # (..., n_vars)

        # embed each variable independently
        var_outputs = []
        for v in range(self.n_vars):
            xi_v = self.var_grns[v](x[..., v:v + 1])  # (..., hidden_size)
            var_outputs.append(xi_v)
        stacked = torch.stack(var_outputs, dim=-2)  # (..., n_vars, hidden_size)

        # weighted sum over variables
        selected = (weights.unsqueeze(-1) * stacked).sum(dim=-2)  # (..., hidden_size)
        return selected, weights
