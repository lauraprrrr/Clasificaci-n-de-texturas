import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

ARCHIVO_ENTRADA = './Procesamiento/texturas_scaled_24.csv' 
N_FOLDS = 10       
K_VECINOS = 5      
SEED = 42          

try:
    df = pd.read_csv(ARCHIVO_ENTRADA)
except FileNotFoundError:
    exit()

X = df.drop(['clase', 'archivo'], axis=1)
y = df['clase']

print(f"Dimensiones: {X.shape[0]} muestras, {X.shape[1]} características.")
print("-" * 60)

modelos = []

# Naive Bayes
modelos.append(('Naive Bayes', GaussianNB()))

# Árbol de Decisión
modelos.append(('Árbol Decisión', DecisionTreeClassifier(criterion='gini', random_state=SEED)))

# KNN 
modelos.append((f'KNN (k={K_VECINOS})', KNeighborsClassifier(n_neighbors=K_VECINOS, metric='euclidean')))



resultados = []
nombres = []

print(f"{'MODELO':<20} | {'F-SCORE PROM.'} | {'DESVIACIÓN'}")
print("-" * 60)

for nombre, modelo in modelos:
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    cv_results = cross_val_score(modelo, X, y, cv=kfold, scoring='f1_weighted')
    
    resultados.append(cv_results)
    nombres.append(nombre)
    
    msg = f"{nombre:<20} | {cv_results.mean():.4f}        | {cv_results.std():.4f}"
    print(msg)

print("-" * 60)



plt.figure(figsize=(10, 6))
plt.title('Comparación de Rendimiento (K-Fold Cross Validation)', fontsize=16)

bplot = plt.boxplot(resultados, patch_artist=True, labels=nombres)

colores = ['lightblue', 'lightgreen', 'orange']
for patch, color in zip(bplot['boxes'], colores):
    patch.set_facecolor(color)

plt.ylabel('F-Score ', fontsize=12)
plt.xlabel('Modelos', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)


for i in range(len(resultados)):
    y_vals = resultados[i]
    x_vals = np.random.normal(i + 1, 0.04, size=len(y_vals))
    plt.plot(x_vals, y_vals, 'r.', alpha=0.5)

plt.tight_layout()
plt.show()