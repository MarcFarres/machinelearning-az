#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:45:44 2019

@author: juangabriel
"""

# Regresión polinómica

# en este ejemplo queremos disponer de datos para negociar el salario de un nuevo
# empleado en función del cargo que ejercia en su antigua empresa

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
# colocando 1:2 es como cojer solo 1 columna (un vector) per al hacerlo así conseguimos
# codificar ese vector en una matriz (matriz de caractrísticas)
X = dataset.iloc[:, 1:2].values
# X = variables independientes
y = dataset.iloc[:, 2].values
# y = variables dependientes
plt.scatter(X, y, color = "red")
plt.plot(X, y, color = "blue")


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# no usaremos conjunto de predicción, ya que disponemos de pocos datos

# Escalado de variables
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
# Ajustamos los coeficientes con los datos de los que disponemos
lin_reg.fit(X, y)
# Esto nos servirá para ver que ocurre al intentar ajustar linealmente un conjunto
# de datos que claramente sigue un patrón polinómico

# Ajustar la regresión polinómica con el dataset
from sklearn.preprocessing import PolynomialFeatures
# Trnasformamos la matriz X incluyendo las potencias de X que queremos usar para obtener
# el modelo
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
# Como el fit que hemos echo nos prepara los datos de forma polinómica, la misma función
# "LinearRegression" que antes nos calculaba la regresión lineal, ahora nos arrojará
# la regresión polinómica
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
# crear el modelo lineal que se ajuste con variables polinomiales

# Visualización de los resultados del Modelo Lineal
plt.scatter(X, y, color = "red")
# scatter genera la gráfica con nuestros puntos reales (nube de puntos)
plt.plot(X, lin_reg.predict(X), color = "blue")
# plot lo usamos para visualizar la gráfica de nuestro modelo de regresión
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X), max(X), 0.1)
# X_grid lo usamos para crear puntos intermedios y evitar así que la gráfica se
# visualize como un conjunto de lineas rectas entre los puntos discretos
# en este caso: desde el mínimo de X hasta el máximo de X con intérvalos de grosor 0.1
# El tercer 
X_grid = X_grid.reshape(len(X_grid), 1)

# Predicción de nuestros modelos
test_value = 6.5
test_value = np.array(test_value)
test_value = test_value.reshape(1, -1)
# al introducir el valor -1 en las columnas, estas se distribuyen automáticamente
lin_reg.predict(test_value)
# predict nos da los valores de la función que onstituye nuestro modelo
lin_reg_2.predict(poly_reg.fit_transform(6.5))


# El reshape lo usamos para pasar de un vector fila a un vector columna
# filas = len(X_grid) , columnas = 1
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()








