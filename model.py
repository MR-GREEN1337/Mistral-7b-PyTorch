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
    ffn_dim_multiplier: Optional[float] = None
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

class RotaryPostionalEncoding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model

    @staticmethod
    def compute_theta_params(d_model: int, end: int, theta: float = 10000.):
        assert d_model % 2 ==0, "can't use an odd dimension!"
        
        freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2)[: (d_model // 2)].float() / d_model))
        t = torch.arange(end, device=freqs.device)
        # Calculate m * theta_i
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cisp = torch.polar(torch.ones_like(freqs), freqs)  # we don't care about the scalars for now, we set them to 1
        return freqs_cisp
    
    def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
        x_rotated = x_complex * freq_complex
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)

        return x_out.type_as(x).to(device)

    def forward(self, x):
        R = torch.zeros(2, self.d_model)
        R += x*torch.cos(compute_theta_params(self.d_model, ))



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