import torch
import torch.nn as nn
from app.attention import Multipleattention

class TransformerBlock(nn.Module):
    """
    Un bloque estándar de Transformer que combina Atención y Feed-Forward.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # 1. Capa de Atención (la que ya revisamos)
        self.attention = Multipleattention(d_model, num_heads)
        
        # 2. Red Feed-Forward (dos capas lineales con una activación ReLU o GeLU)
        # GPT usa 4 veces el tamaño de d_model en la capa intermedia
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        
        # 3. Normalización de Capas (LayerNorm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Dropout para evitar sobreajuste (overfitting)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Estructura Residual: x + Subcapa(Norm(x))
        
        # Paso 1: Atención con conexión residual
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        
        # Paso 2: Feed-Forward con conexión residual
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        
        return x