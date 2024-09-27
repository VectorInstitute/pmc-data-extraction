import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        residual = q

        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(nhead, d_model, d_k=d_model//nhead, d_v=d_model//nhead, dropout=dropout)    # 内含 norm + atten + dropout + residual

        # Implementation of Feedforward model
        self.ffn = nn.Sequential(
          nn.Linear(d_model, dim_feedforward),
          _get_activation_md(activation),
          nn.Dropout(dropout),
          nn.Linear(dim_feedforward, d_model),
          nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):

        q = k = src
        src = self.self_attn(q, k, src)[0]

        src2 = self.norm(src)
        src2 = self.ffn(src2)
        src = src + src2

        return src

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.self_attn = MultiHeadAttention(nhead, d_model, d_k=d_model//nhead, d_v=d_model//nhead, dropout=dropout)

        self.cross_attn = MultiHeadAttention(nhead, d_model, d_k=d_model//nhead, d_v=d_model//nhead, dropout=dropout)

        self.ffn = nn.Sequential(
          nn.Linear(d_model, dim_feedforward),
          _get_activation_md(activation),
          nn.Dropout(dropout),
          nn.Linear(dim_feedforward, d_model),
          nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):

        tgt = self.cross_attn(tgt, memory, memory)[0]

        tgt = self.self_attn(tgt, tgt, tgt)[0]

        tgt2 = self.norm(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + tgt2

        return tgt

class TransformerEncoder(nn.Module):
  def __init__(self, encoder_layer, num_layers):
    super().__init__()
    self.layers = _get_clones(encoder_layer, num_layers)
    self.num_layers = num_layers

  def forward(self, src):
    output = src

    for layer in self.layers:
      output = layer(output)    # (bs, patch_num, feature_dim)

    return output

class TransformerDecoder(nn.Module):
  def __init__(self, decoder_layer, num_layers):
    super().__init__()
    self.layers = _get_clones(decoder_layer, num_layers)
    self.num_layers = num_layers

  def forward(self, encoder_memory, query, return_intermedia=False):
    query_output = query
    intermedia = []

    for i in range(len(self.layers)):
      query_output = self.layers[i](query_output, encoder_memory)   # (bs, query_num, feature_dim)
      if return_intermedia:
        intermedia.append(query_output)

    return query_output, intermedia

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
  """Return an activation function given a string"""
  if activation == "relu":
    return F.relu
  if activation == "gelu":
    return F.gelu
  if activation == "glu":
    return F.glu
  raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_activation_md(activation):
  """Return an activation function given a string"""
  if activation == "relu":
    return nn.ReLU()
  if activation == "gelu":
    return nn.GELU()
  if activation == "glu":
    return nn.GLU()
  raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
