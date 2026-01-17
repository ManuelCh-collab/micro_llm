import os
from typing import List, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:
    """Esta clase encapsula la lógica de entrenamiento, codificación y decodificación
    utilizando implementaciones optimizadas en Rust.
    """

    def __init__(self, vocab_size: int = 50257) -> None:
        """Inicializa el tokenizer con un modelo BPE vacío.
        
        Args:
            vocab_size: Tamaño máximo del vocabulario (50257 es el estándar de GPT-2).
        """
        self.vocab_size: int = vocab_size
        # Inicializamos el modelo BPE
        self.tokenizer: Tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        
        # El Pre-tokenizer de nivel de byte es esencial para manejar 
        # cualquier carácter UTF-8 y espacios correctamente
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

    def train(self, files: List[str]) -> None:
        """Entrena el tokenizer a partir de una lista de archivos de texto.
        
        Args:
            files: Lista de rutas a los archivos .txt para el entrenamiento.
        """
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        
        # Este proceso en Rust usará tus 2 núcleos al 100% de forma eficiente
        self.tokenizer.train(files, trainer)

    def encode(self, text: str) -> List[int]:
        """Codifica texto plano en una secuencia de IDs enteros.
        
        Args:
            text: La cadena de texto a procesar.
            
        Returns:
            Una lista de IDs representativos de las subpalabras.
        """
        output = self.tokenizer.encode(text)
        return output.ids

    def decode(self, ids: List[int]) -> str:
        """Decodifica una secuencia de IDs de vuelta a una cadena de texto.
        
        Args:
            ids: Lista de identificadores de tokens.
            
        Returns:
            El texto reconstruido.
        """
        return self.tokenizer.decode(ids)

    def save(self, path: str) -> None:
        """Guarda el tokenizer entrenado en un archivo JSON."""
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Carga un tokenizer previamente entrenado desde un archivo."""
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        return instance