"""
Modulo de configuracion
"""
from pathlib import Path

class Config:
    """
    Configuracion general del modelo
    """
    # Tokenizacion
    VOCAB_SIZE: int = 5000
    D_MODEL: int = 256
    MAX_SEQ_LEN: int = 512

    # Directorios
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_TOKEN_DIR: Path = BASE_DIR / "kracker"

    @classmethod
    def create_dirs(cls) -> None:
        """
        Crea los directorios si no existen
        """
        cls.MODEL_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
