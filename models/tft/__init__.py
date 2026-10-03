"""
Layer 6: Temporal Fusion Transformer (Lim et al. 2021), implemented from
scratch in PyTorch (no pytorch-forecasting / darts). Multi-horizon
probabilistic (quantile) forecasting of forward returns.

Components:
  grn.py       Gated Residual Network + Gated Linear Unit
  vsn.py       Variable Selection Network
  attention.py Interpretable multi-head attention
  model.py     Full TFT (VSN -> LSTM enc/dec -> static enrichment ->
               interpretable attention -> quantile heads)
  trainer.py   Pinball (quantile) loss + walk-forward training
  features.py  Assemble the real (ticker, date) feature panel from the DB
"""
