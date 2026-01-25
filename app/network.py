import torch
import torch.nn as nn
from typing import Optional

class FeedForward(nn.Module):
    

    def init(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1) -> None:
       
        super().init()
        if d_ff is None:
            d_ff = 4 * d_model

        self.w_1: nn.Linear = nn.Linear(d_model, d_ff)

        
        self.activation: nn.GELU = nn.GELU()

        self.w_2: nn.Linear = nn.Linear(d_ff, d_model)

        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        
     
        x = self.activation(self.w_1(x))

        
        x = self.dropout(x)

       
        x = self.w_2(x)

        return x