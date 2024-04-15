import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data  # características
y = iris.target  # etiquetas

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el clasificador SVM
clf = SVC()

# Entrenar el clasificador
clf.fit(X_train, y_train)

# Hacer una predicción con nuevas características
nuevas_caracteristicas = np.array([[5, 3.1, 2.3, 0.9]])
prediccion = clf.predict(nuevas_caracteristicas)

# Imprimir la predicción
print("La predicción para las características proporcionadas es:", iris.target_names[prediccion[0]])
