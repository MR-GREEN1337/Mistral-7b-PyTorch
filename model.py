import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    d_model: int = 4096
    n_layers: int = 32
    head_dim: int = 128
    hidden_dim: int = 14336
    n_heads: int = 32
    n_kv_heads: int = 8
    window_size: int = 4096
    context_len: int = 8192
    vocab_size: int = 32000
    num_of_experts_per_tok: Optional[int] = None
    num_experts: Optional[int] = None
    norm_eps: float = 1e-5
    p_RMSNorm: float = -1.
    # Needed for the KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048 # For the rolling buffer part

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
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model
        self.gamma = nn.Parameter(torch.ones(self.d_model))
        self.p = args.p_RMSNorm
        self.eps = args.norm_eps

        assert self.p != 0, "What do you want me to calculate! p!=0"

    def forward(self, x):
        criterion = self.p**2 <= 1 and self.p > 0
        d_choice = [self.d_model, int(self.d_model * self.p)]
        d_x = d_choice[criterion]

        if criterion:
            norm_x, _ = torch.split(x, [d_x, self.d_model - d_x], dim=-1)

        norm_x = x.norm(2, dim=-1, keepdim=True)
        norm_x /= math.sqrt(d_x)

        x /= norm_x

        return x * self.gamma

class RotaryPostionalEncoding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.d_model = args.d_model

    @staticmethod
    def compute_theta_params(d_model: int, seq_len: int, device: str, theta: float = 10000.):
        assert d_model % 2 ==0, "can't use an odd dimension!"
        
        freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2)[: (d_model // 2)].float() / d_model))
        t = torch.arange(seq_len, device=freqs.device)
        # Calculate m * theta_i
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cisp = torch.polar(torch.ones_like(freqs), freqs)  # we don't care about the scalars for now, we set them to 1
        return freqs_cisp

    @staticmethod
    def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
        x_rotated = x_complex * freq_complex
        x_out = torch.view_as_real(x_rotated)
        x_out = x_out.reshape(*x.shape)

        return x_out.type_as(x).to(device)

    def forward(self):
        pass 



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
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = RotaryPostionalEncoding().apply_rotary_embeddings(xq, freq_complex, device=x.device)
        xk = RotaryPostionalEncoding().apply_rotary_embeddings(xk, freq_complex, device=x.device)

        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        keys = self.cache_k[:batch_size, :start_pos+seq_len]
        values = self.cache_v[:batch_size, :start_pos+seq_len]
        
        #Apply unrotation
        keys = torch.cat([GQA().unrotate(self.cache_k[i, :, :, :] for i in range(self.cache_k.shape[0]))], dim=0)
        value = torch.cat([GQA().unrotate(self.cache_v[i, :, :, :] for i in range(self.cache_v.shape[0]))], dim=0)

        # repeat k/v heads if n_kv_heads < n_heads
        # make the number of heads in kv and q the same
        keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_rep)
        values = torch.repeat_interleave(values, dim=2, repeats=self.n_rep)

        # Self-attention
        xq = xq.transpose(1, 2) # (batch_size, n_local_heads, seqlen, head_dim)

        keys = keys.transpose(1, 2) # (batch_size, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (batch_size, n_local_heads, cache_len + seqlen, head_dim)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)  # (batch_size, n_local_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)
    
    '''    Unrotate the K and V sequence when it's saturated, unpluck the first element, 
        shift to the last and put the element in queue last
    '''
    @staticmethod
    def unrotate(cache: torch.Tensor, seq_len: int) -> torch.Tensor:
        assert cache.ndim == 3, "Dimensionality problem"
        position = seq_len % cache.shape[0]
        if seq_len < cache.shape[0]:
            return cache[:seq_len]
        elif position == 0:
            return cache
        else:
            return torch.cat([cache[position:], cache[:position]], dim=0)

    def sliding_window_attention(x: torch.Tensor, w: int):
        pass