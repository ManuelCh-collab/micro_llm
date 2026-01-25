import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self , d_model: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model: int = d_model  
        self.num_heads: int = num_heads 
        self.d_k: int = d_model//num_heads # Dimension por cabeza
        
        # proyecciones lineales para Queries, keys y values 
        self.w_q: nn.Linear = nn.Linear(d_model, d_model)
        self.w_k: nn.Linear = nn.Linear(d_model, d_model)
        self.w_v: nn.Linear = nn.Linear(d_model, d_model)
        
        self.w_o: nn.Linear = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) ->torch.Tensor:
        
        batch_size,seq_len, _ = x.size()
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores.masked_fill(mask== 0, float('-inf'))
            
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        out = out.transponse(1, 2).contiguos().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(out)