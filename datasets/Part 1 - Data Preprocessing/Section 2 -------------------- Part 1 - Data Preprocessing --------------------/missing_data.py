#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:19:21 2019

@author: juangabriel
"""

# Plantilla de Pre Procesado - Datos faltantes

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
# Se utiliza para cargar y manipular datos (csv, xml, ...)
import pandas as pd
# Tratamiento de los NAs
from sklearn.preprocessing import Imputer

# Importar el data set
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Le decimos que tipo de datos vamos a procesar
# El segundo parámetro es el método que vamos a utilizar para substituir esos valores
# En python cuando se quiere aplicar algo a una columna se usa "axis = 0" !!!
# En python para aplicar un calculo a toda una fila se usa (axis = 1) !!
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# reemplazamos los valores NaN por las medias de su columna (mean of axis 0)
#fit se usa para aplicar, ejecutar la función que hemos asignado a la variable imputer.
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
# El echo de indicar 1:3  es porque python no incluye la última columna especificada
# así pues para cojer las columnas 1 y 2  deberemos especificar 1:3
# Codificar datos categóricos
# Decodificador de las características de los usuarios
# Le decimos: oye creame un codificador de datos ... vale ok aquí lo tienes
