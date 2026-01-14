from tokenizers import ByteLevelBPETokenizer
import os

# 1. Configuración: Pongan aquí el nombre de su archivo
NOMBRE_ARCHIVO = "entrenamiento.txt"  # <--- Cambien esto por el nombre de su txt
CARPETA_SALIDA = "tokenizador_estudiantes"

# Verificación de seguridad
if not os.path.exists(NOMBRE_ARCHIVO):
    print(f"Error: No encuentro el archivo {NOMBRE_ARCHIVO}")
else:
    # 2. Inicializar el tokenizador
    tokenizer = ByteLevelBPETokenizer()

    # 3. ENTRENAMIENTO
    # Nota: vocab_size=5000 es un buen punto de partida para archivos medianos.
    # Si su archivo es gigante (megabytes), pueden subirlo a 30000 o 50000.
    tokenizer.train(
        files=[NOMBRE_ARCHIVO],
        vocab_size=5000, 
        min_frequency=2,
        show_progress=True,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    # 4. GUARDAR LOS RESULTADOS
    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
    
    tokenizer.save_model(CARPETA_SALIDA)
    print(f"Tokenizador entrenado y guardado en la carpeta: {CARPETA_SALIDA}")

    # 5. PROBAR CON UNA FRASE NUEVA
    texto_prueba = "el 2 de enero en el territorio mexicano se sintio un sismo de 6.4 de magnitud"
    codificado = tokenizer.encode(texto_prueba)

    print("\n--- RESULTADOS DE LA PRUEBA ---")
    print(f"Texto: {texto_prueba}")
    print(f"Tokens: {codificado.tokens}")
    print(f"IDs: {codificado.ids}")