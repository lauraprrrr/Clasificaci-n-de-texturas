import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, trace, det
from tqdm import tqdm
import warnings


def calculate_fisher_score(X, y):
    """
    Calcula el Índice de Fisher 
    """

    # garantizar que la matriz sea 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
        
    n_features = X.shape[1]
    class_labels = np.unique(y)
    n_classes = len(class_labels)
    
    # Calcular la media global
    global_mean = np.mean(X, axis=0)
    
    # Inicializar matrices Cw y Cb
    Cw = np.zeros((n_features, n_features))
    Cb = np.zeros((n_features, n_features))
    
    for label in class_labels:
        Xi = X[y == label]
        n_samples_k = Xi.shape[0]
        
        p_k = n_samples_k / X.shape[0]
        
        
        class_mean_k = np.mean(Xi, axis=0)
        
        diff_cb = (class_mean_k - global_mean).reshape(n_features, 1)
        Cb += p_k * (diff_cb @ diff_cb.T)
        
        Cw += p_k * np.cov(Xi.T)


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        if det(Cw) < 1e-10:
            Cw_reg = Cw + np.eye(n_features) * 1e-6
        else:
            Cw_reg = Cw

        try:
            # Calcular el score
            J = trace(inv(Cw_reg) @ Cb)
        except np.linalg.LinAlgError:
            J = 0 
    
    return J

# Carga de datos normalizados generados en normalizar_bd.py

ARCHIVO_ENTRADA = 'texturas_scaled_24.csv'

try:
    df = pd.read_csv(ARCHIVO_ENTRADA)
except FileNotFoundError:
    print(f"Error al cargar datos")
    exit()

# Separar características y etiquetas 
columnas_features = df.columns.drop(['clase', 'archivo'])
etiquetas = df['clase']

# Convertir a arrays numpy para los calculos
X_data = df[columnas_features].to_numpy()
y_data = etiquetas.to_numpy()

print(f"Datos listos: {X_data.shape[0]} muestras, {X_data.shape[1]} características, {len(np.unique(y_data))} clases.")

# Uso de SFS

# se quiere ncontrar el mejor conjunto hasta un máximo de N características
# se usaron las 24 para observar mejor el comportamiento
MAX_FEATURES_TO_SELECT = 24 

# Lista de índices de todas las características 
available_features_idx = list(range(X_data.shape[1]))
# Lista de índices de las características que se van seleccionando
selected_features_idx = []

# para guardar los resultados de cada iteración
best_scores_history = []
best_features_history = []


for k in range(MAX_FEATURES_TO_SELECT):
    
    scores_this_iter = []
    features_this_iter = []

    # probar CADA característica disponible
    for idx in tqdm(available_features_idx, desc=f"  Iteración {k+1}/{MAX_FEATURES_TO_SELECT}", leave=False):
        
        # crear el conjunto de características de la itereacuion
        current_selection_idx = selected_features_idx + [idx]
        
        # extraer los datos de ese subconjunto
        X_subset = X_data[:, current_selection_idx]
        
        # calcular el score de Fisher para este subconjunto
        score = calculate_fisher_score(X_subset, y_data)
        
        # guardar el score y el índice
        scores_this_iter.append(score)
        features_this_iter.append(idx)
        
    # decidir cual fue el mejor
    
    # encontrar el score más alto de esta iteración
    best_score_index = np.argmax(scores_this_iter)
    best_score_this_iter = scores_this_iter[best_score_index]
    
    # identificar la característica q produjo ese score
    best_feature_to_add_idx = features_this_iter[best_score_index]
    
    selected_features_idx.append(best_feature_to_add_idx)
    available_features_idx.remove(best_feature_to_add_idx)
    
    # Guardar los resultados
    best_scores_history.append(best_score_this_iter)
    
    # NOMBRES de las características
    selected_names = [columnas_features[i] for i in selected_features_idx]
    best_features_history.append(selected_names)
    
    print(f"Iter {k+1}: Score = {best_score_this_iter:.4f} | Mejor nueva: '{columnas_features[best_feature_to_add_idx]}' | Total: {selected_names}")

print("\n--- SFS Completado ---")

# Encontrar el mejor score y el número óptimo de características
final_best_score = np.max(best_scores_history)
n_features_optim = np.argmax(best_scores_history) + 1 
optimal_feature_set = best_features_history[np.argmax(best_scores_history)]

print("\nResultado")
print(f"El máximo rendimiento (Score de Fisher = {final_best_score:.4f}) se alcanza con:")
print(f"**{n_features_optim} características.**")
print("\nEl conjunto óptimo de características es:")
for i, feat in enumerate(optimal_feature_set):
    print(f"  {i+1}. {feat}")

# graficar el rendimiento
x_axis = range(1, MAX_FEATURES_TO_SELECT + 1)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, best_scores_history, marker='o', linestyle='-')
plt.title('Rendimiento de Características vs. Score de Fisher (SFS)', fontsize=16)
plt.xlabel('Número de Características Seleccionadas', fontsize=12)
plt.ylabel('Índice de Fisher (J)', fontsize=12)
plt.xticks(x_axis)

plt.axvline(x=n_features_optim, color='red', linestyle='--', 
            label=f'Óptimo ({n_features_optim} feats) J={final_best_score:.2f}')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# =============================================================================
#  EXPORTACIÓN DE DATOS FILTRADOS (Iteración 17)
# =============================================================================

print("\n--- Generando archivo con características seleccionadas ---")

# 1. Recuperar la lista de características
# Como listas en Python parten de 0, la iteración 17 corresponde al índice 16
INDICE_OPTIMO = 16 

if INDICE_OPTIMO < len(best_features_history):
    features_optimas = best_features_history[INDICE_OPTIMO]
    print(f"Características recuperadas de la iteración {INDICE_OPTIMO + 1}:")
    print(features_optimas)

    # 2. Filtrar el DataFrame original
    # IMPORTANTE: Concatenamos las features seleccionadas con 'clase' y 'archivo'
    # para no perder las etiquetas ni los identificadores.
    columnas_finales = features_optimas + ['clase', 'archivo']
    
    # Creamos el nuevo dataframe filtrado
    df_filtrado = df[columnas_finales]

    # 3. Generar el nuevo archivo CSV
    NOMBRE_SALIDA = 'texturas_seleccionadas_optimas.csv'
    df_filtrado.to_csv(NOMBRE_SALIDA, index=False)

    print(f"\n¡Éxito! Archivo guardado como: '{NOMBRE_SALIDA}'")
    print(f"Dimensiones del nuevo dataset: {df_filtrado.shape}")
    print("Este es el archivo que debes cargar para entrenar tus clasificadores (Naive Bayes, etc).")

else:
    print(f"Error: El historial solo tiene {len(best_features_history)} iteraciones. No se puede acceder al índice {INDICE_OPTIMO}.")