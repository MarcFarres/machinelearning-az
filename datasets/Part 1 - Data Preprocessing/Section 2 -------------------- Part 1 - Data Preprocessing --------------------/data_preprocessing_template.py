#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:43:11 2019

@author: juangabriel
"""

# Plantilla de Pre Procesado

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Data.csv')
# index localization = iloc
# selección de filas => ":"  desde el inicio hasta el final
# selección de columnas => ":-1"  Todas menos la última
# els .values indica que queremos extraer solo los valores, ni la posición etc ...
X = dataset.iloc[:, :-1].values
# Obtenemos una única columna con todas sus filas, que corresponderá con la 
# variable independiente que querremos predecir
# Todas las filas, de la quarta columna ( Las columnas comienzan en el 0 ) 
y = dataset.iloc[:, 3].values

#---------------------------------------
# Dividimos el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
# X variables independientes de training y test
# y variables dependientes de training y de test donde veremos si las predicciones
# son correctas
# Reservamos un 20% para testing
# Cuantos mas datos tenemos para testear, menos datos tendremos para aprender
# training se usa para que el algoritmo aprenda
Xtraining, Xtesting, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
# importamos solo una parte de una libreria
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X[:,1:3] = imputer.transform(X[:,1,3])

#escalado de variables
from sklearn.preprocessing import StandardScaler
escalableX = StandardScaler()
# utilizamos fit_transform para aplicar directamente el escalado sobre la variable original
# si no usariamos primer fit y luego transform
Xtraining = escalableX.fit_transform(Xtraining)
# Para los datos de testing no usamos fit, ya que tendriamos diferentes escalados, en su lugar
# usaremos la misma transofrmacion que ha detectado para los valores de training
Xtesting = escalableX.transform(Xtesting)
