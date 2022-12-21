#Importando bibliotecas

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

#importando dataframe da UCI
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df.columns = ['sepal-length','sepal-width','petal-length','petal-width','class']

print('\n *** *** *** - - - - - - - - - - Base de dados Iris - - - - - - - - - - - *** *** ***\n')
print(df.head())
print()

#Análise Descritiva do dataframe
print('\n - - - - - - - - - -  Análise Descritiva  - - - - - - - - - - - \n')
print(df.describe().transpose())

#Info
print('\n - - - - - - - - - - Info - - - - - - - - - - - \n')
print(df.info())
print('- '*27)

#Buscando valores nulos
print('\n - - - - - - - - - - Valores nulos - - - - - - - - - - - \n')
print(df.isnull().sum())
print('- '*30)

# Excluindo uma das colunas
x = df.drop('class', axis=1)
print('\n - - - - - - - - - - Dataframe atualizado - - - - - - - - - - - \n')
print(x.head())
print('- '*30)

