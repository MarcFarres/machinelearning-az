dataset = read.csv('Data.csv')

#tratamiento de valores NA
dataset$Age = ifelse(is.na(dataset$Age), 
                     # aplicamos para la columna de edades la funcion en que para cada x es reemplazada por la media de todas las x sin
                     # contar los valores que son iguales a na
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), 
                     # Los valores que no son na se quedan igual
                     dataset$Age) 

dataset$Salary = ifelse(is.na(dataset$Salary), 
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), 
                        dataset$Salary)

# codificamos la variables categóricas

#Un factor no serán números, sino que serán números directamente
# no tedrán orden
dataset$Country = factor(dataset$Country,
                         levels = c("France", "Spain", "Germany"),
                         labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased,
                           levels = c("No", "Yes"),
                           labels = c(0, 1))

# Dividir los datos en conjunto de training y test
# install.packages("caTools")
library(caTools)
# Elegimos la semilla para calcular los random
set.seed(3141516)
# decidimos que variables se usarán para entrenar y cuales para testing
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)


# Escalado de valores
# a diferencia de python el 3 si que se tiene en cuenta al establecer la última columna
# y no hace falta inlcuir el ":" para especificar que se usaran todas las filas
training_set = scale(training_set[, 2:3])
testing_set = scale(testing_set[, 2:3])
# los factores pese a que los visualizamos como números, realmente son como strings
# para solventar este problema hemos tenido que especificarle a la función
# scale, que solo trabaje con las columnas numéricas