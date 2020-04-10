# -*- coding: utf-8 -*-
"""
Mi primer desarrollo con machine learning
"""

"------------------------------Importar librerias------------------------"
"pandas permite manipular la data facilmente"
import pandas as pd  
"scikit-learn permite crear modelos"
from sklearn.ensemble import RandomForestRegressor
"MAE"
from sklearn.metrics import mean_absolute_error
"dividir los datos de entrenamiento y validacion"
from sklearn.model_selection import train_test_split

"---------------------Cargar la data u obtener los datos-----------------"
melbourne_file_path = "/Users/juanmendezcuartas/Downloads/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)

"------------------------------Analizar la data--------------------------"
melbourne_data.shape
melbourne_data.describe()
melbourne_data.head()
melbourne_data.columns

"---------------------------Eliminar valores faltantes-------------------"
melbourne_data_drop = melbourne_data.dropna(axis=0)

"-----------------------Seleccionar que se quiere predecir---------------"
y = melbourne_data_drop["Price"]

"---------------Seleccionar los features para predecir-------------------"
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data_drop[features]

"---------------Dividir datos de entrenamiento y valicacion--------------"
train_X , test_X, train_y, test_y = train_test_split(X, y, random_state = 1)

"""
"------------------------Creacion del modelo-----------------------------"
melbourne_data_model = RandomForestRegressor(random_state = 0)

"-------------------------Entrenamiento del modelo-----------------------"
melbourne_data_model.fit(train_X, train_y)

"---------------------------Predicion------------------------------------"
melbourne_data_predict = melbourne_data_model.predict(test_X)

"----------------------MAE : Mean Absolute Error-------------------------"
melbourne_mae= mean_absolute_error(melbourne_data_predict, test_y)
print("El MAE es : ")
print(melbourne_mae)
"""

"------------------Verificacion de cantidad de leaf----------------------"
def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    melbourne_data_model = RandomForestRegressor(random_state=0)
    melbourne_data_model.fit(train_X, train_y)
    melbourne_data_predict = melbourne_data_model.predict(test_X)
    melbourne_mae = mean_absolute_error(test_y, melbourne_data_predict)
    return(melbourne_mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
