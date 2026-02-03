import torch
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class GPTDataset(Dataset):
    """
    Versión optimizada: usa 'stride' para reducir el número de muestras
    y acelerar el entrenamiento en CPU.
    """
    def __init__(self, file_path: str, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 1. Leer el archivo
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 2. Tokenizar todo el texto una sola vez
        self.logger.info(f"Tokenizando texto...")
        self.tokens = tokenizer.encode(text)
        
        # ESTRATEGIA DE VELOCIDAD: 
        # En lugar de movernos de 1 en 1 (174k muestras), 
        # nos movemos en bloques de tamaño max_seq_len.
        self.stride = 8 
        
        self.logger.info(f"Total tokens: {len(self.tokens)}")

    def __len__(self) -> int:
        # Esto calcula cuántos bloques completos caben en el texto
        return (len(self.tokens) - self.max_seq_len - 1) // self.stride

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculamos la posición inicial basada en el salto (stride)
        start_idx = idx * self.stride
        
        chunk = self.tokens[start_idx : start_idx + self.max_seq_len + 1]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

def create_dataloader(file_path: str, tokenizer, batch_size: int, max_seq_len: int):
    dataset = GPTDataset(file_path, tokenizer, max_seq_len)
    # num_workers=0 es más seguro para evitar errores de memoria en Windows
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)