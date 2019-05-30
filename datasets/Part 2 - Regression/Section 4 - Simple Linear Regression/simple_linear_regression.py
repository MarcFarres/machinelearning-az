#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 12:07:43 2019

@author: juangabriel
"""

# Regresión Lineal Simple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Salary_Data.csv')
# Todas menos la última, la variable independiente (el índice no es columna real)
X = dataset.iloc[:, :-1].values
# Tendremos una única variable dependiente (años de experiencia)
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
# como es un modelo muy simple haremos un tercia para testing y dos tercios para training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Escalado de variables
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# El modelo de regresión lineal ya lleva incorporado el modelo de escalado
# además solo teneoms una variable

# Crear modelo de Regresión Lienal Simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

# Predecir el conjunto de test
#deberemos especificar que observaciones utilizamos para realizar la predicción
# En este caso usaremos obviamente las variables que no hemos guardado para test
y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamiento
# Pintamos una nube de puntos (scatterplot)
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")

# Visualizar los resultados de test
# con la recta calculada usando los puntos de learning comprobamos como quedan los puntos de testing
plt.scatter(X_test, y_test, color = "green")
plt.scatter(X_test, y_pred, color = "yellow")
plt.plot(X_train, regression.predict(X_train), color = "blue")
plt.title("Sueldo vs Años de Experiencia (Conjunto de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()

