# llm 

En este script se tiene como objetivo entrenar un tokenizador personalizado utilizando lo que es el algoritmo BPE o Byte Pair Encoding, siendo este proceso fundamental para el proceso de lenguaje natural( NLP) para convertir lo que es texto en una sevuencia de numeros que un modelo de LLM pueda entender 

## Paso 1:  Estructura
Se utiliza la libreria tokenizers de Hugging Face, la cual es altamente eficiente, donde al ejercutar el codigo se creara un archivo y las siguientes carpetas:

entrenamiento.txt: Donde se le dara al tokenizador todo el texto con el que se quiera entrenar a la LLM.

tokenizer_estudiantes: Es la carpeta oficial definida en la variable CARPETA_SALIDA de tu script.

Contenido: Contiene los archivos vocab.json y merges.txt.

Función: Es el resultado directo del comando tokenizer.save_model(CARPETA_SALIDA). Es la que debes entregar como el "producto final" de tu entrenamiento.

Carpeta: modelo_tokenizer (Opcional/Respaldo)
Si tu código actual no la menciona explícitamente, su aparición puede deberse a dos razones:

Ejecuciones previas: Si probaste un código anterior que tenía ese nombre, la carpeta permanece ahí.

Compatibilidad de la librería: Algunas versiones de la librería tokenizers crean una carpeta temporal o de respaldo con nombres genéricos si no se especifica bien la ruta.

## Paso 2: ¿Que es el BPE?

El como trabaja el BPE(Byte Pair Encoding) en lo que es el codigo:

Este construye las palabras estadísticamente basándose en el archivo entrenamiento.txt siguiendo estos pasos:

Representación a Nivel de Bytes: En lugar de ver letras, el tokenizador ve bytes. Esto es fundamental porque permite que el modelo pueda leer cualquier símbolo, emoji o carácter especial sin fallar.

Aprendizaje por Fusión (Merges): Durante el entrenamiento (tokenizer.train), el algoritmo escanea el texto buscando los pares de bytes que aparecen uno al lado del otro con más frecuencia.

Ejemplo: Si en el texto aparece mucho la palabra "sismo", el algoritmo primero verá s, i, s, m, o.

Notará que s + i es muy común y los unirá: si.

Luego unirá si + s = sis.

Este proceso continuaria hasta que se alcanza el límite de 5000 tokens que se definieron en el codigo.

Manejo de Espacios con Ġ: Notarás en la prueba final que el BPE a nivel de bytes usa un carácter especial (como un punto o una Ġ) para marcar dónde había un espacio. Esto permite que el proceso sea reversible: puedes pasar de números a texto sin perder el formato original.

Manejo de los espacios con Ġ: En donde habian espacios se representaria con la Ġ para poder permitir que el proceso sea reversible: Dando la oportinudad de poder cambiar de numeros a texto sin perder lo que es el formato original.

## Paso 3: entrenamiento del modulo

En esta parte, el método .train() analiza el texto y construye el vocabulario siguiendo estos parámetros:

vocab_size=5000: Define el número máximo de tokens únicos que el modelo recordará.

min_frequency=2: Solo se crean tokens para secuencias que aparecen al menos 2 veces, evitando ruidos o errores de escritura.

special_tokens: Son marcadores de control esenciales:

<s> y </s>: Inicio y fin de frase.

<pad>: Relleno para que todas las secuencias tengan la misma longitud.

<unk>: Token para caracteres totalmente desconocidos.

<mask>: Utilizado para tareas de entrenamiento donde el modelo debe "adivinar" una palabra oculta.

## Paso 4: Guardado y verificación

En esta parte del código, se asegura de que el trabajo realizado durante el entrenamiento no se pierda. Básicamente, haciendo dos cosas:

Verificación de la carpeta: Se usa la librería os para revisar si ya existe la carpeta tokenizador_estudiantes. Si no existe, el script la crea automáticamente. Esto se hace para evitar errores al intentar guardar archivos en una ruta que no existe.

Exportación del modelo: Una vez que el entrenamiento termina, el comando save_model "exporta" o guarda el aprendizaje en dos archivos (vocab.json y merges.txt).

## Paso 5: Funcionamiento

Para poder realizar la prueba de funcionamiento se le implemento un texto sobre un sismo ocurrido en mexico el 2 de enero para probar asi si el script pasa lo que es el lenguaje humano a una secuencia de IDs(numeros), siendo estos numeros los que "alimentan" a la LLM.

## Paso 6: Estructura

C:.
│   .gitignore
│   entrenamiento.txt
│   llm.py
│   README.md
│
├───modelo_tokenizer
│       merges.txt
│       vocab.json
│
└───tokenizador_estudiantes
        merges.txt
        vocab.json

llm.py: Es el archivo principal donde se escribio todo el código de Python para el entrenamiento y las pruebas.

entrenamiento.txt: Es el documento de texto que contiene toda la información que se uso para que el tokenizador aprendiera el lenguaje.

tokenizador_estudiantes/: Es la carpeta de salida donde el código guardó el "cerebro" del tokenizador (vocab.json y merges.txt). Esta es la carpeta que se debe usar para futuros proyectos.

modelo_tokenizer/: Una carpeta generada durante el proceso de desarrollo que contiene los archivos de configuración del modelo.

.gitignore: Un archivo que configuré para que, al subir el trabajo a GitHub, no se suban archivos innecesarios o muy pesados (como el contenido del venv).

README.md: El archivo de documentación (este informe) donde se explica paso a paso todo el funcionamiento del proyecto.

```python

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

```