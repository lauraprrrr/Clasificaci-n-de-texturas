import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# ÁRBOL DE DECISIÓN
clf_tree = DecisionTreeClassifier(criterion='gini', random_state=42)

# Entrenar
clf_tree.fit(X_train, y_train)

# Predecir
y_pred = clf_tree.predict(X_test)

# fscore
fscore = f1_score(y_test, y_pred, average='weighted')

print("\n--- Resultados Árbol de Decisión ---")
print(f"F-Score Promedio: {fscore:.4f}")
print("-" * 30)


print(classification_report(y_test, y_pred))

# MATRIZ DE CONFUSIÓN
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Matriz de Confusión - Árbol de Decisión\nF-Score: {fscore:.2f}')
plt.ylabel('Clase Real')
plt.xlabel('Clase Predicha')
plt.tight_layout()
plt.show()

