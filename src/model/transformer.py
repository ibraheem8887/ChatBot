"""
Transformer encoder-decoder from scratch for Empathetic Chatbot
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------- Positional Encoding ---------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------- Scaled Dot-Product Attention -----------------
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    output = torch.matmul(attn, v)
    return output, attn


# --------------------- Multi-Head Attention ---------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        b, seq, d = x.size()
        x = x.view(b, seq, self.n_heads, self.d_k).transpose(1, 2)
        return x  # (b, heads, seq, d_k)

    def combine_heads(self, x):
        b, h, seq, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(b, seq, h * d_k)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.q_linear(q))
        k = self.split_heads(self.k_linear(k))
        v = self.split_heads(self.v_linear(v))
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (b,1,q,k)
        out, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = self.combine_heads(out)
        out = self.out_linear(out)
        return out


# --------------------- Feed-Forward Network ---------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------- Encoder Layer ---------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(src2))
        src2 = self.ff(src)
        src = self.norm2(src + self.dropout(src2))
        return src


# --------------------- Decoder Layer ---------------------
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt


# ---------------- Encoder / Decoder stacks ----------------
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt


# --------------------- Full Transformer ---------------------
class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=4,
        d_ff=512,
        enc_layers=2,
        dec_layers=2,
        dropout=0.1,
        pad_idx=3,
        max_len=512,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = Encoder(d_model, n_heads, d_ff, enc_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, dec_layers, dropout)
        self.generator = nn.Linear(d_model, vocab_size, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_pad_mask(self, seq):
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        bsz, tgt_len = tgt.size()
        pad_mask = self.make_pad_mask(tgt)
        nopeak = torch.tril(torch.ones((1, 1, tgt_len, tgt_len), device=tgt.device))
        return pad_mask & nopeak.bool()

    def forward(self, src, tgt):
        src_mask = self.make_pad_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        memory = self.encoder(src_emb, src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask, src_mask)
        logits = self.generator(output)
        return logits


# --------------------- Sanity Test ---------------------
if __name__ == "__main__":
    model = Transformer(vocab_size=1000)
    src = torch.randint(0, 1000, (4, 12))
    tgt = torch.randint(0, 1000, (4, 10))
    out = model(src, tgt)
    print("Output shape:", out.shape)  # (batch, tgt_len, vocab_size)
