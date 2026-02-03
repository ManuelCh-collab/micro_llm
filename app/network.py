import torch
import torch.nn as nn
from app.embedding_logic import TransformerEmbedding
from app.transformer_block import TransformerBlock # El que creamos antes

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        # 1. Capa de entrada (lo que ya tenías en embedding_logic)
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_seq_len)
        
        # 2. El cuerpo del modelo: Una lista de bloques de Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # 3. Capa final de normalización
        self.ln_f = nn.LayerNorm(d_model)
        
        # 4. Cabezal de salida: Convierte vectores de vuelta a probabilidades de palabras
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, mask=None):
        # Pasar por el embedding
        x = self.embedding(input_ids)
        
        # Pasar por cada piso (bloque) del Transformer
        for block in self.blocks:
            x = block(x, mask)
        
        # Normalizar y proyectar al vocabulario
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits