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

#Quita KG
train['Weight'] = train['Weight'].str.replace('kg', '')

#Convierte la columna a ordinal
train['Weight'] = pd.to_numeric(train['Weight'])

#Renombrar columna Storage
train.rename(columns={' Storage': 'Storage'}, inplace=True)

# Extrace valores numericos. Valor bajo = SSD, medio = Hybrid, alto = HDD.
train['Storage'] = train['Storage'].str.extract('(\d+)').astype(int)

# Extrace valores numericos de RAM. Capacidad en GB.
train['RAM'] = train['RAM'].str.extract('(\d+)').astype(int)

# Extrace marca GPU.
train['GPU'] = train['GPU'].str.split().str[0]

#Carga archivo con info de procesadores.
with open('data/message.txt', "r") as file:
    contenido=file.read()
dictio={i.split(':')[0]:i.split(':')[1].split() for i in contenido.replace('    ','').replace(',','').replace("'",'').replace(')','').replace('(','').split('\n')}

#Crea columnas con frecuencia y número de nucleos del procesador.
train['CPU_Frecuencia']=train['CPU'].apply(lambda x : ''.join(x.split()[-1])).str.replace('GHz','').astype(float)
train['CPU_Nucleos']=train['CPU'].apply(lambda x:dictio[x][0]).astype(int)

#Tira columna CPU.
train.drop('CPU', axis=1, inplace=True)

#Dejar solo valor numérico en Screen Size
train['Screen Size'] = train['Screen Size'].str.replace('"', '').astype(float)

#Transforma columna Screen.
train['Screen'] = train['Screen'].str.rsplit(' ', 1).str[-1]

#Separar valores para multiplicar después.
train[['Width', 'Height']] = train['Screen'].str.extract(r'(\d+)x(\d+)', expand=True)
#Pasar a ordinal
train['Width'] = pd.to_numeric(train['Width'])
train['Height'] = pd.to_numeric(train['Height'])
#Pasar Screen a ordinal con total de pixeles
train['Screen'] = train['Width'] * train['Height']
train.drop(['Width', 'Height'], axis=1, inplace=True)

#Tiro la columna OS Version porque no me parece que afecte el precio desde el estudio del problema
train.drop('Operating System Version', axis=1, inplace=True)

#Juntar valores de MACOS
train['Operating System'] = train['Operating System'].replace("Mac OS", "macOS")

#Tirar columnas con baja correlacion con y, coeficiente bajo en modelo OLS o alta colinealidad.
train.drop(['Manufacturer', 'Model Name', 'Category', 'Screen Size'], axis=1, inplace=True)

#One_hot_encoding de GPU y Operating System.
train = pd.get_dummies(train, columns=['GPU', 'Operating System'])

#Añade columna GPU_ARM con todo 0 para matchear test.
train = train.assign(GPU_ARM=0)

#---Limpiar y transformar test---

#Importa csv
test=pd.read_csv('data/test.csv')

#Quita KG
test['Weight'] = test['Weight'].str.replace('kg', '')

# Cambia valor cheeky de ValueError: Unable to parse string "4s" at position 83 por media.
test['Weight'] = test['Weight'].str.replace('4s', '2')

#Convierte la columna a ordinal
test['Weight'] = pd.to_numeric(test['Weight'])

#Renombrar columna Storage
test.rename(columns={' Storage': 'Storage'}, inplace=True)

# Extrace valores numericos. Valor bajo = SSD, medio = Hybrid, alto = HDD.
test['Storage'] = test['Storage'].str.extract('(\d+)').astype(int)

# Extrace valores numericos de RAM. Capacidad en GB.
test['RAM'] = test['RAM'].str.extract('(\d+)').astype(int)

# Extrace marca GPU.
test['GPU'] = test['GPU'].str.split().str[0]

#Carga archivo con info de procesadores.
with open('data/message.txt', "r") as file:
    contenido=file.read()
dictio={i.split(':')[0]:i.split(':')[1].split() for i in contenido.replace('    ','').replace(',','').replace("'",'').replace(')','').replace('(','').split('\n')}

#Crea columnas con frecuencia y número de nucleos del procesador.
test['CPU_Frecuencia']=test['CPU'].apply(lambda x : ''.join(x.split()[-1])).str.replace('GHz','').astype(float)
test['CPU_Nucleos']=test['CPU'].apply(lambda x:dictio[x][0]).astype(int)

#Tira columna CPU.
test.drop('CPU', axis=1, inplace=True)

#Dejar solo valor numérico en Screen Size
test['Screen Size'] = test['Screen Size'].str.replace('"', '').astype(float)

#Transforma columna Screen.
test['Screen'] = test['Screen'].str.rsplit(' ', 1).str[-1]

#Separar valores para multiplicar después.
test[['Width', 'Height']] = test['Screen'].str.extract(r'(\d+)x(\d+)', expand=True)
#Pasar a ordinal
test['Width'] = pd.to_numeric(test['Width'])
test['Height'] = pd.to_numeric(test['Height'])
#Pasar Screen a ordinal con total de pixeles
test['Screen'] = test['Width'] * test['Height']
test.drop(['Width', 'Height'], axis=1, inplace=True)

#Tiro la columna OS Version porque no me parece que afecte el precio desde el estudio del problema
test.drop('Operating System Version', axis=1, inplace=True)

#Juntar valores de MACOS
test['Operating System'] = test['Operating System'].replace("Mac OS", "macOS")

#Tirar columnas con baja correlacion con y, coeficiente bajo en modelo OLS o alta colinealidad.
test.drop(['Manufacturer', 'Model Name', 'Category', 'Screen Size'], axis=1, inplace=True)

#One_hot_encoding de GPU y Operating System.
test = pd.get_dummies(test, columns=['GPU', 'Operating System'])

#---Entrenar y predecir modelo---
#Importar librerías.
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

#Dividir X e y.
X = train.drop('Price', axis=1) 
y = train['Price']
#Dividir entrenamiento y prueba.
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = tts(X, y, test_size=325)

#Modelo de Regresión lineal.
modelo_reg = LinearRegression()
modelo_reg.fit(X_train, y_train)
reg_pred = modelo_reg.predict(X_test)
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
bosque_pred = modelo_bosque.predict(X_test)
#MSE
bosque_mse = mean_squared_error(y_test, bosque_pred)
print('MSE: ', bosque_mse)
#R2 Score
bosque_r2 = r2_score(y_test, bosque_pred)
print('R2 Score: ', bosque_r2)

#Inicio DF para muestra.
muestra2 = pd.DataFrame()
#Añado id
muestra2['id'] = range (0, len(X_test))
#Precio prediccion.
muestra2['Price'] = bosque_pred
#Reseteo Index
muestra2.reset_index(drop = True, inplace = True)
#Exportar muestra2
muestra2.to_csv('muestra2.csv', index = False)
