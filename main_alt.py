#Importar librerías

import warnings
warnings.filterwarnings('ignore')

import regex as re

import pandas as pd
import numpy as np

import pylab as plt
import seaborn as sns

%matplotlib inline

#---Limpiar y transformar train---

#Importa csv
train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')

#Concatena DFs
tt = pd.concat([train,test], axis=0)

#Tirar columnas con baja correlacion con y, coeficiente bajo en modelo OLS o alta colinealidad.
tt.drop(['Manufacturer', 'Model Name', 'Category', 'Screen Size', 'Operating System Version'], axis=1, inplace=True)

#Quita KG
tt['Weight'] = tt['Weight'].str.replace('kg', '')

# Cambia valor cheeky de ValueError: Unable to parse string "4s" at position 83 por media.
tt['Weight'] = tt['Weight'].str.replace('4s', '2')

#Convierte la columna a ordinal
tt['Weight'] = pd.to_numeric(tt['Weight'])

#Renombrar columna Storage
tt.rename(columns={' Storage': 'Storage'}, inplace=True)

# Extrace valores numericos. Valor bajo = SSD, medio = Hybrid, alto = HDD.
tt['Storage'] = tt['Storage'].str.extract('(\d+)').astype(int)

# Extrace valores numericos de RAM. Capacidad en GB.
tt['RAM'] = tt['RAM'].str.extract('(\d+)').astype(int)

# Extrace marca GPU.
tt['GPU'] = tt['GPU'].str.split().str[0]

#Carga archivo con info de procesadores.
with open('data/message.txt', "r") as file:
    contenido=file.read()
dictio={i.split(':')[0]:i.split(':')[1].split() for i in contenido.replace('    ','').replace(',','').replace("'",'').replace(')','').replace('(','').split('\n')}

#Crea columnas con frecuencia y número de nucleos del procesador.
tt['CPU_Frecuencia']=tt['CPU'].apply(lambda x : ''.join(x.split()[-1])).str.replace('GHz','').astype(float)
tt['CPU_Nucleos']=tt['CPU'].apply(lambda x:dictio[x][0]).astype(int)

#Tira columna CPU.
tt.drop('CPU', axis=1, inplace=True)

#Transforma columna Screen.
tt['Screen'] = tt['Screen'].str.rsplit(' ', 1).str[-1]

#Separar valores para multiplicar después.
tt[['Width', 'Height']] = tt['Screen'].str.extract(r'(\d+)x(\d+)', expand=True)
#Pasar a ordinal
tt['Width'] = pd.to_numeric(tt['Width'])
tt['Height'] = pd.to_numeric(tt['Height'])
#Pasar Screen a ordinal con total de pixeles
tt['Screen'] = tt['Width'] * tt['Height']
tt.drop(['Width', 'Height'], axis=1, inplace=True)

#Juntar valores de MACOS
tt['Operating System'] = tt['Operating System'].replace("Mac OS", "macOS")

#One_hot_encoding de GPU y Operating System.
tt = pd.get_dummies(train, columns=['GPU', 'Operating System'])

#Separar train y test.
traineo = tt.head(977)
testeo = tt.tail(325)

#---Entrenar y predecir modelo---
#Importar librerías.
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

#Dividir X e y.
X = traineo.drop('Price', axis=1) 
y = traineo['Price']
#Dividir entrenamiento y prueba.
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = tts(X, y)

#Modelo de Regresión lineal.
modelo_reg = LinearRegression()
modelo_reg.fit(X_train, y_train)
reg_pred = modelo_reg.predict(testeo)
#MSE
reg_mse = mean_squared_error(y_test, reg_pred)
print('MSE: ', reg_mse)
#R2 Score
reg_r2 = r2_score(y_test, reg_pred)
print('R2 Score: ', reg_r2)

#MSE suggests this is not the best model to consider for this case.

# Modelo de Bosque.
modelo_bosque = RandomForestRegressor()
modelo_bosque.fit(X_train, y_train)
bosque_pred = modelo_bosque.predict(testeo)
#MSE
bosque_mse = mean_squared_error(y_test, bosque_pred)
print('MSE: ', bosque_mse)
#R2 Score
bosque_r2 = r2_score(y_test, bosque_pred)
print('R2 Score: ', bosque_r2)

#Inicio DF para muestra.
muestra3 = pd.DataFrame()
#Añado id
muestra3['id'] = range (0, len(test))
#Precio prediccion.
muestra3['Price'] = bosque_pred
#Reseteo Index
muestra3.reset_index(drop = True, inplace = True)
#Exportar muestra2
muestra3.to_csv('muestra3.csv', index = False)
