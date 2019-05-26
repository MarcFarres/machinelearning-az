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




