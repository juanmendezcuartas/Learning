# -*- coding: utf-8 -*-
"""
Mi primer desarrollo con machine learning
Mejor resultado : 180949
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
"imputar datos nulos"
from sklearn.impute import SimpleImputer

"---------------------Cargar la data u obtener los datos-----------------"
melbourne_file_path = "/Users/juanmendezcuartas/Documents/MachineLearning/Data/melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)

"------------------------------Analizar la data--------------------------"
melbourne_data.shape
melbourne_data.describe()
melbourne_data.head()
melbourne_data.columns

"-----------------------Seleccionar que se quiere predecir---------------"
y = melbourne_data["Price"]

"---------------Seleccionar los features para predecir-------------------"
features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[features]

"---------------Dividir datos de entrenamiento y valicacion--------------"
train_X , test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, 
                                                     test_size=0.2,random_state = 1)
"""---------------------------Eliminar valores faltantes-------------------"
"Verificar las filas que le faltan datos"
cols_with_missing = [col for col in train_X.columns
                     if train_X[col].isnull().any()]
train_X_reduce = train_X.drop(cols_with_missing)
test_X_reduce = test_X.drop(cols_with_missing)
"Drop columns: eliminar las columnas con datos faltantes"
#melbourne_data_drop = melbourne_data.dropna(axis=0)
"""
"""
"Imputation: llenar datos faltantes con un promedio de la columna"
# Imputation
my_imputer = SimpleImputer()
imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
imputed_test_X = pd.DataFrame(my_imputer.transform(test_X))

# Imputation removed column names; put them back
imputed_train_X.columns = train_X.columns
imputed_test_X.columns = test_X.columns
"""
"Extend imputation: lo mismo que el imputation pero adicionando un dato de true"
# Make copy to avoid changing original data (when imputing)
train_X_plus = train_X.copy()
test_X_plus = test_X.copy()

cols_with_missing = [col for col in train_X.columns
                     if train_X[col].isnull().any()]

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    train_X_plus[col + '_was_missing'] = train_X_plus[col].isnull()
    test_X_plus[col + '_was_missing'] = test_X_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_train_X_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_test_X_plus = pd.DataFrame(my_imputer.transform(test_X_plus))

# Imputation removed column names; put them back
imputed_train_X_plus.columns = train_X_plus.columns
imputed_test_X_plus.columns = test_X_plus.columns
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
def score_dataset(n_estimators_nodes,train_X, test_X, train_y, test_y):
    melbourne_data_model = RandomForestRegressor(n_estimators = n_estimators_nodes,
                                                 random_state=1)
    melbourne_data_model.fit(train_X, train_y)
    melbourne_data_predict = melbourne_data_model.predict(test_X)
    melbourne_mae = mean_absolute_error(test_y, melbourne_data_predict)
    return(melbourne_mae)

for n_estimators_nodes in [5, 50, 500, 5000]:
    my_mae = score_dataset(n_estimators_nodes,imputed_train_X_plus, imputed_test_X_plus,
                           train_y, test_y)
    print("Max trees nodes: %d  \t\t Mean Absolute Error:  %d" %(n_estimators_nodes,
                                                                my_mae))
