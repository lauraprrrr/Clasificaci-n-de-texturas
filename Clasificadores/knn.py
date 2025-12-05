import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix

archivo_entrada =  './Procesamiento/texturas_scaled_24.csv'

try:
    df = pd.read_csv(archivo_entrada)
except FileNotFoundError:
    exit()


X = df.drop(['clase', 'archivo'], axis=1)
y = df['clase']


X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y
)

K_VECINOS = 5
clf_knn = KNeighborsClassifier(n_neighbors=K_VECINOS, metric='euclidean')

#guarda los datos de entrenamiento
clf_knn.fit(X_train, y_train)
# Predecir
y_pred = clf_knn.predict(X_test)

fscore = f1_score(y_test, y_pred, average='weighted')

print(f"\n--- Resultados KNN (K={K_VECINOS}) ---")
print(f"F-Score Promedio: {fscore:.4f}")
print("-" * 30)


print(classification_report(y_test, y_pred))


plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Matriz de Confusión - KNN (K={K_VECINOS})\nF-Score: {fscore:.2f}')
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.tight_layout()
plt.show()