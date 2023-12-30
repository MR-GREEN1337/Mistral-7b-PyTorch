import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: int = Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for the KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # Multiply by sqrt(d_model) to increase variance which tends to decrease


'''
We use RMSNorm instead of Layer Normalization for computational efficiency

'''
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, p: float = -1., eps: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.p = p
        self.eps = eps

        assert self.p != 0, "What do you want me to calculate! p!=0"

    def forward(self, x):
        d_choice = [self.d_model, int(self.d_model * self.p)]
        d_x = d_choice[0 if self.p**2 <= 1 and self.p > 0 else 1]

        if self.p**2 <= 1 and self.p > 0:
            norm_x, _ = torch.split(x, [d_x, self.d_model - d_x], dim=-1)

        norm_x = x.norm(2, dim=-1, keepdim=True)
        norm_x *= math.sqrt(d_x)

        x /= norm_x

        return x * self.gamma

class GQA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        #Number of  K and V heads
        self.n_kv_heads = [args.n_kv_heads, args.n_heads][args.n_kv_heads is None]
        
        #Number of  Q heads
        self.n_heads_q = args.n_heads

        self.n_rep = self.n_heads_q // self.n_kv_heads

        self.head_dim = args.d_model // args.n_heads

        self.wq = nn.Linear(args.d_model, self.n_rep * self.head_dim, bias=False)
        self.wk = nn.Linear(args.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.d_model, self.n_kv_heads * self.head_dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

        def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
            pass