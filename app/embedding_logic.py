import torch
import torch.nn as nn
import math
from typing import Optional

class TransformerEmbedding(nn.Module):
    """Capa de Embedding que combina representaciones semánticas y posicionales.
    
    Esta clase transforma IDs de tokens en vectores densos y añade información
    sobre la posición de cada token en la secuencia.
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 512) -> None:
        """Inicializa las capas de embedding.
        
        Args:
            vocab_size: Tamaño del vocabulario del tokenizer.
            d_model: Dimensión de los vectores de embedding (espacio latente).
            max_seq_len: Longitud máxima de las secuencias de entrada.
        """
        super().__init__()
        self.d_model: int = d_model
        
        # Token Embedding: Matriz de tamaño [vocab_size, d_model]
        self.token_embedding: nn.Embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding: Matriz de tamaño [max_seq_len, d_model]
        # Usaremos embeddings aprendidos (estilo GPT) en lugar de senos/cosenos fijos
        self.position_embedding: nn.Embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Realiza el paso hacia adelante (forward pass).
        
        Args:
            input_ids: Tensor de forma (batch_size, sequence_length) con IDs de tokens.
            
        Returns:
            Tensor de forma (batch_size, sequence_length, d_model).
        """
        seq_length: int = input_ids.size(1)
        
        # Generamos las posiciones [0, 1, 2, ..., seq_length-1]
        positions: torch.Tensor = torch.arange(0, seq_length, device=input_ids.device).unsqueeze(0)
        
        # Sumamos ambos embeddings: Semántico + Posicional
        # El escalado por sqrt(d_model) ayuda a la estabilidad del gradiente
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)
        
        return x