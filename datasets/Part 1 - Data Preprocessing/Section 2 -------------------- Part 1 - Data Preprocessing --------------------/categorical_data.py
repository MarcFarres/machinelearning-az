#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:19:06 2019

@author: juangabriel
"""

# Plantilla de Pre Procesado - Datos Categóricos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
# Se utiliza para cargar y manipular datos (csv, xml, ...)
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
# Importar el data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0);
# reemplazamos los valores NaN por las medias de su columna (mean of axis 0)
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
# Codificar datos categóricos
# Decodificador de las características de los usuarios
# Le decimos: oye creame un codificador de datos ... vale ok aquí lo tienes
labelencoder_X = LabelEncoder()
labelencoder_Y = LabelEncoder()
# toma directamente las columnas que le indicamos y las transforma 
# en datos numéricos
# Cojeme todas las filas de la primera columna (en python es la 0)
# Subsituimos la primera columna en todas sus filas por la transformación
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
y = labelencoder_Y.fit_transform(y)
# A label encoder le da igual que el dato sea categórico u ordinal
# nos trasforma cada valor en un entero que lo caracteriza
# categorical_features nos dice en que columna se halla nuestra variable
# categórica que queremos codificar
Xhotencoder = OneHotEncoder(categorical_features=[0,3])
# OneHotEncoder en cambio, nos genera una columna extra para cada categoria detectada
# donde cada row sera 1 o 0 en función de si ese label está o no presente en dicha row
# solo tendremos un uno por fila, de aquí el término 'one hot'
X = Xhotencoder.fit_transform(X).toarray()
# Antes de aplicar onehotencoder necesitamos transformar la primera columna como integers
# para ellos usamos el labelEncoder


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


#escalado de variables
from sklearn.preprocessing import StandardScaler
escalableX = StandardScaler()
# utilizamos fit_transform para aplicar directamente el escalado sobre la variable original
# si no usariamos primer fit y luego transform
Xtraining = escalableX.fit_transform(Xtraining)
# Para los datos de testing no usamos fit, ya que tendriamos diferentes escalados, en su lugar
# usaremos la misma transofrmacion que ha detectado para los valores de training
Xtesting = escalableX.transform(Xtesting)


