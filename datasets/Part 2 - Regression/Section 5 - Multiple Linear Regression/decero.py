# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:57:21 2019

@author: wheelhub
"""

# Regresión lineal múltiple
# manejo de estructuras de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_startups.csv')
# Recojemos en una matriz todos los datos independientes
X = dataset.iloc[:, :-1].values
# La variable dependiente
y = dataset.iloc[:, 4].values
# Hay una categoria que tendrá que ser transformada a variable dummy

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# transformamos las columnas con string en numéricas
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# Hot encoder divide en n-columnas las distintas categorias que hemos numerizado
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#hay que eliminar una de las columnas de las variables dummy, ya que son dependientes
#eliminamos la primera columna
X = X[:,1:]

# Ajustamos el modelo de regresión usando el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as statsmodel
# usamos la misma clase para una lineal simple que una múltiple
regression = LinearRegression()
regression.fit(X_train, y_train)
# después de hacer el fit ya estaremos en condiciones de realizar las transformaciones
# creamos ahora un vector de predicciones
y_pred = regression.predict(X_test)
# Usamos la eliminación hacia atrás
# El p-valor en nuestro caso se usa para cuantificar la probabilidad de que uno de los 
# coeficientes se acerque mucho a 0
# agregamos una columna de números 1 y lo asociamos a nuestro término independiente

# X = np.append(arr = X, values = np.ones((50, 1)).astype(int), axis = 1)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# añadimos una columna nuevo con 50 filas todas con valor 1
# axis = 1 nos indica "en columna"  axis = 0  nos indicaria en fila
#permutamos el dataset por el valor nuevo a añadir para que se añada al inicio en lugar
# de al final
SL = 0.05
# Establecemos el nivel de significación óptimo (valor normal es de 0.05, un 5%)
X_opt = X[:, [0,1,2,3,4,5]]
# creamos la matriz de variables 
# al principio tenemos en cuenta todas las varibales independientes
# para ir descartando las menos significativos
regressionOLS = statsmodel.OLS(endog = y, exog = X_opt).fit()
#ordinari list squares = OLS (método de los mímimos cuadrados)
# endog = variable endógena. Se trata de la variable que queremos predecir (las variables independientes)
# exog = variable exógena. la matriz de características. 
# El numero de filas debe ser igual que en el vector y de las observaciones
# teenemos que indicar todas las columnas que forman parte de las columnas regresoras
# Debemos de haber añadido las ordenadas en origen en forma de columna de unos (ya echo)
# regressionOLS.fit()
# ajustamos el modelo. Se vuelve a generar la regresión linear múltiple que ya habiamos echo
# con la función LinearRegression de sklearn, pero ahora dispondrá de info extra: 
# el cálculo del p-valor
regressionOLS.summary()
# eliminamos la columna num. 2 , que es la que tiene un p-valor mas grande
X_opt = X[:, [0,1,3,4,5]]
regressionOLS = statsmodel.OLS(endog = y, exog = X_opt).fit()
regressionOLS.summary()

X_opt = X[:, [0,3,4,5]]
regressionOLS = statsmodel.OLS(endog = y, exog = X_opt).fit()
regressionOLS.summary()

X_opt = X[:, [0,3,5]]
regressionOLS = statsmodel.OLS(endog = y, exog = X_opt).fit()
regressionOLS.summary()
# en este paso deberemos valorar que el 0 de la variable x2 entra en el intervalo de confianza
# eso significa que es muy probable que el coeficiente de ese término acabe siendo 0
# Pero entra por muy poco ...  es buena decisión también dejarla y quedarnos aquí
regression.fit(X_opt, y_train)
final_pred = regression.predict(X_opt)
# Como acaba esto ????