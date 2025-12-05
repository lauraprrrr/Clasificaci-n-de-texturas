import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Archivo de entrada (generado en crear_bd)
ARCHIVO_ENTRADA = 'texturas_features_24.csv'
# Archivo de salida 
ARCHIVO_SALIDA = 'texturas_scaled_24.csv'

#leer archivo
df = pd.read_csv(ARCHIVO_ENTRADA)


# Separar las características de las etiquetas
columnas_metadatos = ['clase', 'archivo']
metadatos_df = df[columnas_metadatos]

# obtener la lista de columnas de características
columnas_features = df.columns.drop(columnas_metadatos)
features_df = df[columnas_features]


"""
Normalización: Media=0, Desviación=1
"""
#Inicializar el escalador
scaler = StandardScaler() #StandardScaler calcula la media (mu) y la desviación estándar (sigma) de cada columna

# fit calcula mu y sigma de cada columna
# transform aplica la fórmula z = (x - mu) / sigma
X_scaled = scaler.fit_transform(features_df)

# convertir resultado de la normalización a df
df_scaled = pd.DataFrame(X_scaled, columns=columnas_features)

# se vuelve a unir las características con sus etiquetas
df_final_normalizado = pd.concat([df_scaled, metadatos_df], axis=1)

# guardar el nuevo dataset normalizado
df_final_normalizado.to_csv(ARCHIVO_SALIDA, index=False)

print("\n✦ . ⁺  . ✦ . ₊✩‧₊˚౨ৎ˚₊✩‧₊⁺ ✦ Primeras 5 filas de los datos normalizados ✦ ₊✩‧₊˚౨ৎ˚₊✩‧₊⁺ . ✦ . ⁺  . ✦")
print(df_final_normalizado.head())