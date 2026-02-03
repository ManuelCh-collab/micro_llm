import logging
from pathlib import Path

from trainer import TrainModule
from app.config import Config

model_name = "ChefcitoGPT"

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f'{model_name}.log', encoding="utf-8"),
            logging.StreamHandler()
        ]
)
        
logger = logging.getLogger(model_name)

chefcito_dataset = Path(__file__).resolve().parent/ 'entrenamiento.txt'

if not chefcito_dataset.exists():
    logger.error(f"No se encontró el archivo en {chefcito_dataset}")
    
chefcitotrainer = TrainModule(
    config=Config,
    dataset_path=chefcito_dataset,
    model_name=model_name
)

# Aquí le decimos a Remy que "recuerde" lo que aprendió antes
# Asegúrate de que el nombre coincida con el archivo en tu carpeta checkpoints
ultimo_checkpoint = Config.MODEL_TOKEN_DIR / "checkpoints" / "checkpoint_epoch_10.pth"

if ultimo_checkpoint.exists():
    logger.info(f"Cargando memoria de Remy desde {ultimo_checkpoint}...")
    try:
        chefcitotrainer.load_checkpoint(ultimo_checkpoint)
        logger.info("¡Memoria cargada con éxito!")
    except Exception as e:
        logger.error(f"No se pudo cargar el checkpoint: {e}")
else:
    logger.info("No se encontró checkpoint anterior, Remy empezará desde cero.")
    
try:
    logger.info("Iniciando entrenamiendo de Remy... (aprendiendo a comer queso)")
    chefcitotrainer.train(epochs=20)
    logger.info("Exito, Remy le gano a Gustov")
except Exception as e:
    logger.error(f"Error al tratar de entrenar a Remy: {e}")
    pass
except KeyboardInterrupt:
    logger.warning("A chefcito le dieron veneno")