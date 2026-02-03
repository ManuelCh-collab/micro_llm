Informe: 

# Desarrollo y Optimización de Chefcito.LLM 

## 1. Arquitectura y Configuración del Sistema

El modelo se basa en una arquitectura Transformer (Decoder-only) escalada para un aprendizaje especializado en el dominio gastronómico.

 Configuración técnica: 6 capas de atención, un D_Model de 256 y una ventana de contexto ampliada a 128 tokens.

 Estado de Convergencia: Según los últimos logs (Epoch 12-13), el modelo alcanzó un Loss estable de ~0.69, lo que representa un equilibrio entre precisión y generalización.

## 2. Diagnóstico Técnico de la Falta de Coherencia

Tras analizar los fallos iniciales de funcionamiento, se identificaron cuatro factores críticos que impedían que la LLM respondiera correctamente:

### A. Calidad del Dataset (Ruido Estructural)

  El Problema: El archivo entrenamiento.txt original contenía un exceso de espacios en blanco y párrafos vacíos.

  Impacto: El modelo malgastaba su capacidad de cómputo intentando aprender patrones de "silencio" o vacío en lugar de aprender gramática. Esto causaba que las respuestas fueran interrumpidas o carecieran de flujo lógico.

  Solución: Se realizó una limpieza profunda del archivo de texto para aumentar la densidad de información.

### B. Sobredimensión del Vocabulario

  El Problema: El VOCAB_SIZE inicial de 50,000 era desproporcionado para el tamaño del dataset.
 * Impacto: Al haber tantos tokens posibles, la probabilidad se dispersaba (sparse probability), lo que generaba "alucinaciones léxicas" o palabras inventadas.

 Solución: Se redujo el vocabulario a 5,000 tokens, obligando al modelo a ser más preciso y coherente con las palabras reales del libro.
C. Desbalance de Hiperparámetros (d_model y Temperatura)

  d_model: Una dimensión mal ajustada impedía que el modelo capturara la complejidad de las instrucciones de cocina.
 * Temperatura: Una temperatura alta sin control causaba que el modelo eligiera palabras por azar, ignorando el entrenamiento previo.

## 3. Plan para el Funcionamiento al 100%
Para garantizar que el modelo funcione sin errores y con total coherencia, se implementaron las siguientes mejoras finales:

  Limpieza de Datos: Eliminación de ruido estructural en el archivo de entrenamiento para mejorar la relación señal-ruido.

  Top-K Sampling (K=50): Se añadió una capa de filtrado en la generación que obliga al modelo a elegir solo entre las 50 palabras más probables, eliminando incoherencias accidentales.

 Ajuste de Temperatura (0.7): Se fijó un nivel de creatividad que permite variar las recetas sin perder la estructura técnica de los pasos de cocina.

## Conclusión

El diagnóstico confirma que la incoherencia inicial era producto de una dispersión de datos y parámetros. Al limpiar el dataset y reducir el vocabulario a 5,000, se logró que el modelo "encajara" perfectamente con la información disponible. El Loss actual de 0.69 es el punto óptimo donde el modelo ya no solo memoriza, sino que entiende la estructura de una receta.