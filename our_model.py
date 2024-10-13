###### Only the `GrnTransformer` class is new and the rest of the code is copied from `gene_transformer.py` from GEARS ######
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain_param = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
        ms = x.square().mean(dim=-1, keepdim=True)
        norm_constant = torch.rsqrt(ms + self.eps)
        return self.gain_param * (x * norm_constant)


# Rotary Positional Embedding (RoPE)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        position = torch.arange(seq_len, device=device).unsqueeze(1)  # (seq_len, 1)
        freqs = position * self.inv_freq.unsqueeze(0)  # (seq_len, dim//2)
        cos = freqs.cos()  # (seq_len, dim//2)
        sin = freqs.sin()
        return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim//2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot


# Multi-Head Attention with Grouped Query Attention and RoPE
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, group_size, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.head_dim = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.size()
        H, D = self.num_heads, self.head_dim

        # Compute queries, keys, values
        q = self.wq(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = self.wk(x).view(B, N, H, D).transpose(1, 2)
        v = self.wv(x).view(B, N, H, D).transpose(1, 2)

        # Apply RoPE to q and k
        cos, sin = self.rotary_emb(N, x.device)  # (N, D//2)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # Grouped Query Attention
        if self.group_size < N:
            num_groups = N // self.group_size
            if N % self.group_size != 0:
                # Pad to make divisible by group_size
                pad_len = self.group_size - (N % self.group_size)
                q = F.pad(q, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
                N_padded = N + pad_len
                num_groups += 1
            else:
                N_padded = N

            q = q.view(B, H, num_groups, self.group_size, D)
            k = k.view(B, H, num_groups, self.group_size, D)
            v = v.view(B, H, num_groups, self.group_size, D)

            attn_scores = torch.einsum("bhngd,bhngd->bhng", q, k) / (D**0.5)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            attn_output = torch.einsum("bhng,bhngd->bhngd", attn_probs, v)
            attn_output = attn_output.reshape(B, H, N_padded, D)
            attn_output = attn_output[:, :, :N, :]  # Remove padding if any
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)

        # Combine heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.embed_dim)
        output = self.wo(attn_output)
        return output


# Feed-Forward Network as per Llama3 (w1, w3, w2)
class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.w1(x)
        x3 = self.w3(x)
        x = x1 * F.silu(x3)  # SwiGLU activation
        x = self.w2(x)
        x = self.dropout(x)
        return x


# Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, group_size, dropout=0.0):
        super().__init__()
        self.attention_norm = RMSNorm(embed_dim)
        self.ffn_norm = RMSNorm(embed_dim)
        self.attention = MultiheadAttention(embed_dim, num_heads, group_size, dropout)
        self.feed_forward = FeedForwardNetwork(embed_dim, hidden_dim, dropout)

    def forward(self, x):
        # Self-attention block
        x = x + self.attention(self.attention_norm(x))
        # Feed-forward block
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


######### ALL OF THE ABOVE ARE COPIED FROM `gene_transformer.py` from GEARS #########
class GrnTransformer(nn.Module):
    """Input is a tensor of shape (batch_size, num_genes, embed_dim) and output is a tensor of shape (batch_size, num_genes)."""

    def __init__(
        self,
        num_genes,  # Different from GeneExpressionTransformer from GEARS
        embed_dim=64,
        num_heads=8,
        hidden_dim=256,
        num_layers=1,
        group_size=512,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(embed_dim, num_heads, hidden_dim, group_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(embed_dim)

        # Components different from GeneExpressionTransformer from GEARS
        # embed input vectors of size 758 to size 768. TODO: 758 is hardcoded here
        self.proj = nn.Linear(758, embed_dim)
        self.grn_pred_head = nn.Linear(hidden_dim, num_genes)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.grn_pred_head(x)
