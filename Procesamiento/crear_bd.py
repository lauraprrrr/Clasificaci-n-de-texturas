import os
import glob
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm # Para visualizar progreso con una barra

"""
Función que extrae las 6 características 
"""
def extraccion_textura_canal(canal):
    # Definir internamente los descriptores base
    descriptores = ['contrast', 'energy', 'ASM',
                    'homogeneity', 'correlation', 'dissimilarity']

    # Cuantización de niveles de gris
    columna = canal.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 10))
    img_cuantizada = scaler.fit_transform(columna).astype(int)
    img_cuantizada = img_cuantizada.reshape(canal.shape)

    # Cálculo de la matriz de co-ocurrencia
    glcm = graycomatrix(img_cuantizada,
                        distances=[2],
                        angles=[0],
                        levels=11,
                        symmetric=True,
                        normed=True)

    # Extracción de las propiedades
    features_canal = [graycoprops(glcm, d)[0, 0] for d in descriptores]
    return features_canal



"""
Construcción de la Base de Datos
"""
descriptores = ['contrast', 'energy', 'ASM',
                'homogeneity', 'correlation', 'dissimilarity']

# Lista para guardar todos los datos 
datos = []

# Nombres de las columnas
nombres_columnas = []
for prefijo in ['R', 'G', 'B', 'Gris']:
    for desc in descriptores:
        nombres_columnas.append(f'{prefijo}_{desc}')
        
# columnas para la clase (etiqueta) y el archivo (ID)
nombres_columnas.append('clase')
nombres_columnas.append('archivo')


# Buscar todas las carpetas de clases (textura_01, textura_02, ...)
try:
    clases_carpetas = sorted(glob.glob(os.path.join('BD', 'textura_*')))
    if not clases_carpetas:
        print("ERROR AL CARGAR BD")
        exit()

except Exception as e:
    print(f"Error al buscar carpetas: {e}")
    exit()


# Recorrer cada carpeta de clase
for ruta_clase in tqdm(clases_carpetas, desc="Procesando Clases"):
    nombre_clase = os.path.basename(ruta_clase)
    
    # Buscar todas las imágenes .jpg dentro de la carpeta de la clase
    imagenes_archivos = glob.glob(os.path.join(ruta_clase, '*.jpg'))
    
    # Recorrer cada imagen dentro de la clase
    for ruta_imagen in tqdm(imagenes_archivos, desc=f"  {nombre_clase}", leave=False):
        try:
            # Cargar imagen en formato BGR
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"Advertencia: No se pudo leer la imagen {ruta_imagen}")
                continue

            # Separar canales (BGR)
            canal_B = imagen[:, :, 0]
            canal_G = imagen[:, :, 1]
            canal_R = imagen[:, :, 2]
            
            # Convertir a escala de grises
            img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
            # Extraer características de cada canal
            features_R = extraccion_textura_canal(canal_R)
            features_G = extraccion_textura_canal(canal_G)
            features_B = extraccion_textura_canal(canal_B)
            features_Gris = extraccion_textura_canal(img_gris)
            
            # Construir el vector final
            vector_final = (features_R + 
                            features_G + 
                            features_B + 
                            features_Gris + 
                            [nombre_clase] + 
                            [os.path.basename(ruta_imagen)])
            
            # Añadir a la lista de datos
            datos.append(vector_final)
            
        except Exception as e:
            print(f"\nError procesando la imagen {ruta_imagen}: {e}")

"""
Creación del DataFrame y guardado
"""

if not datos:
    print("DF vacio")
else:
    df = pd.DataFrame(datos, columns=nombres_columnas)
    
    # Guardar la base de datos en un archivo CSV
    archivo = 'texturas_features_24.csv'
    df.to_csv(archivo, index=False)
    
    print(f"\nBase de datos creada")
    print(f"Total de imágenes procesadas: {len(df)}")
    print(f"Total de características extraídas por imagen: {len(df.columns) - 2}")
    print(f"Archivo guardado como: '{archivo}'")
    
    print("\n₊˚ ✧ ‿︵‿୨୧‿︵‿ ✧ ₊˚ Primeras 5 filas de la Base de Datos ₊˚ ✧ ‿︵‿୨୧‿︵‿ ✧ ₊˚")
    print(df.head())
    
