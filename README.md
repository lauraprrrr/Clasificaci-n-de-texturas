# Tarea 4: Clasificación de Texturas con Selección de Características

Este proyecto implementa un flujo completo de procesamiento de imágenes para la clasificación de texturas. El proceso implementa extracción de características, normalización y reducción de dimensionalidad. Posteriormente realiza una evaluación comparativa de los clasificadores Naive Bayes, Árbol de Decisión y KNN.

# Flujo de ejecución 

## Procesamiento

### 1. Crear la Base de Datos
Este script lee las imágenes de la carpeta  BD y crea el archivo `texturas_features_24.csv`.

- Entrada: Imágenes en archivo BD.
- Salida: texturas_features_24.csv.

```bash
python crear_bd.py
```

### 2. Normalizar los Datos
Este script toma el archivo anterior y normaliza los datos, resultando en el archivo `texturas_scaled_24.csv`.

- Entrada: texturas_features_24.csv.
- Salida: texturas_scaled_24.csv.

```bash
python normalizar_bd.py
```

### 3. Seleccionar Características
Ejecuta el algoritmo SFS para encontrar el subconjunto óptimo de características maximizando el Índice de Fisher. Genera un gráfico de rendimiento y crea el dataset final filtrado.

- Entrada: texturas_scaled_24.csv.
- Salida: texturas_seleccionadas_optimas.csv.

```bash
python fisher_sfs.py
```

## Clasificadores

### Naive Bayes, Árbol de decisión y KNN
Estos scripts entrenan y evalúan los modelos individualmente utilizando una división 70% entrenamiento / 30% prueba. Retornan matrices de confusión y reportes de métricas.

```bash
python bayes.py
```
```bash
python arbol.py
```
```bash
python knn.py
```

### Comparación con K-fold cross-validation
Este ejecuta 10 iteraciones para cada algoritmo, calculando la media y desviación estándar del F-Score. Finalmente, genera un diagrama de cajas para comparar visualmente la estabilidad y el rendimiento de los tres modelos.

```bash
python kfold.py
```



