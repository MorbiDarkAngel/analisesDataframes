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
print(df.head())

#Análise Descritiva do dataframe


print(df.describe().transpose())