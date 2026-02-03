import json
import torch
import logging
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from typing import Optional
from torch.utils.data import DataLoader

# Importamos tus clases reales
from app.network import GPTModel
from app.tokenizer_logic import BPETokenizer
from app.dataset import GPTDataset
from app.config import Config

class TrainModule:
    """
    Encapsula la lógica de entrenamiento del modelo GPT adaptado a tu proyecto.
    """
    def __init__(self, config: Config, dataset_path: Path, model_name: str = "kracker_model"):
        self.config = config
        self.dataset_path = dataset_path
        self.model_name = model_name
        
        # Aseguramos que existan las carpetas de guardado
        self.config.create_dirs()
        self.checkpoint_dir = self.config.MODEL_TOKEN_DIR / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 1. Cargar Tokenizer (usando tu método load)
        # Asumimos que ya lo entrenaste y guardaste con el main.py
        tokenizer_path = self.config.MODEL_TOKEN_DIR / "kracker2.json"
        self.tokenizer = BPETokenizer.load(str(tokenizer_path))
        
        # 2. Inicializar Modelo con tus parámetros de Config
        self.model = self._initialize_model()
        
        # 3. Componentes de Entrenamiento (AdamW es el estándar para Transformers)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.start_epoch = 0

    def _initialize_model(self) -> GPTModel:
        self.logger.info("Instanciando el Modelo GPT...")
        return GPTModel(
            vocab_size=self.config.VOCAB_SIZE,
            d_model=self.config.D_MODEL,
            n_heads=8,      # Valores estándar para tu d_model de 256
            n_layers=4,     # Puedes subir esto en Config si deseas
            max_seq_len=self.config.MAX_SEQ_LEN
        )
    
    def _save_model_metadata(self, final_loss: float, dataset_size_chars: int) -> None:
        """Genera el JSON que resume las características de tu modelo."""
        model_data = {
            "metadata": {
                "model_name": self.model_name,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device": "cpu"
            },
            "architecture": {
                "vocab_size": self.config.VOCAB_SIZE,
                "d_model": self.config.D_MODEL,
                "max_seq_len": self.config.MAX_SEQ_LEN,
                "n_layer": self.config.N_LAYERS,
                "total_params": sum(p.numel() for p in self.model.parameters())
            },
            "training_results": {
                "final_loss": round(final_loss, 4),
                "dataset_size_chars": dataset_size_chars
            }
        }

        json_path = self.config.MODEL_TOKEN_DIR / f"{self.model_name}_info.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=4)
        self.logger.info(f"Metadatos guardados en: {json_path}")

    def save_checkpoint(self, epoch: int, loss: float) -> None:
        """Guarda el estado para poder continuar después si se apaga la PC."""
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        self.logger.info(f"Checkpoint guardado: {path}")
        
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Carga un estado guardado (pesos, optimizador y época)."""
        if checkpoint_path and checkpoint_path.exists():
            # weights_only=False es necesario para cargar el estado del optimizador
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.logger.info(f"¡Éxito! Reanudando desde época {self.start_epoch}")
        else:
            self.logger.info("No se encontró checkpoint, iniciando desde cero.")

    def train(self, epochs: int = 5, batch_size: int = 8):
        """Bucle principal de entrenamiento."""
        # Cargar datos usando tu GPTDataset
        dataset = GPTDataset(str(self.dataset_path), self.tokenizer, self.config.MAX_SEQ_LEN)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.logger.info(f"Iniciando entrenamiento. Total de muestras: {len(dataset)}")
        self.model.train()

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            content_size = len(f.read())

        for epoch in range(self.start_epoch, epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                self.optimizer.zero_grad()
                
                logits = self.model(x)
                # Re-formatear para la pérdida: (Batch*Seq, Vocab)
                loss = self.criterion(logits.view(-1, self.config.VOCAB_SIZE), y.view(-1))
                
                loss.backward()
                self.optimizer.step()
                

                if batch_idx % 20 == 0:
                    progreso = (batch_idx / len(train_loader)) * 100
                    self.logger.info(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Progreso: {progreso:.1f}% | Loss: {loss.item():.4f}"
                    )

            # Guardar checkpoint al final de cada época
            self.save_checkpoint(epoch, loss.item())

        # Guardado final de pesos
        final_path = self.config.MODEL_TOKEN_DIR / f"{self.model_name}_final.pth"
        torch.save(self.model.state_dict(), final_path)
        self._save_model_metadata(final_loss=loss.item(), dataset_size_chars=content_size)
        self.logger.info(f"Modelo final guardado en {final_path}")

# Ejemplo de como ejecutarlo:
if __name__ == "__main__":
    config = Config()
    # Asegúrate de que el archivo entrenamiento.txt exista en la raíz
    trainer = TrainModule(config, dataset_path=Path("entrenamiento.txt"))
    trainer.train(epochs=3)