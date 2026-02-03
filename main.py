import os
from pathlib import Path
from app.config import Config
from app.tokenizer_logic import BPETokenizer

def preparar_entorno():
    """
    Se encarga de la logística inicial: carpetas y vocabulario.
    NO carga el modelo ni usa tensores.
    """
    print("=== INICIALIZANDO ENTORNO KRACKER GPT ===")
    
    # 1. Crear directorios (kracker/, etc.)
    Config.create_dirs()
    print(f"Carpeta de salida verificada: {Config.MODEL_TOKEN_DIR}")
    
    # 2. Gestionar el Tokenizer (Diccionario de la IA)
    tokenizer = BPETokenizer(vocab_size=Config.VOCAB_SIZE)
    
    ruta_entrenamiento = "entrenamiento.txt"
    ruta_salida_json = os.path.join(Config.MODEL_TOKEN_DIR, "kracker2.json")

    if os.path.exists(ruta_entrenamiento):
        print(f"Leyendo '{ruta_entrenamiento}' para crear el vocabulario...")
        # Entrena el algoritmo BPE para entender las palabras del archivo
        tokenizer.train([ruta_entrenamiento])
        # Guarda el archivo .json que necesitarán el Trainer y el Generator
        tokenizer.save(ruta_salida_json)
        print(f"Vocabulario guardado con éxito en: {ruta_salida_json}")
    else:
        print(f"ERROR: No se encontró '{ruta_entrenamiento}'.")
        print("Crea el archivo con texto para poder continuar.")

    print("\nPreparación completada.")
    print("Próximos pasos:")
    print("1. Ejecutar 'python trainer.py' para entrenar la red neuronal.")
    print("2. Ejecutar 'python generator.py' para hablar con la IA.")

if __name__ == "__main__":
    preparar_entorno()