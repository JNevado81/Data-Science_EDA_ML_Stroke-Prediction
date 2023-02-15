# Importamos librerías
from utils.funciones import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


# Guardamos en una variable el directorio del script
dir_script = os.path.dirname(os.path.abspath(__file__))

# Importamos el modelo de producción
modelo_importado = importar_modelo(dir_script)

# Cargamos el dataset con los datos normalizados
df_test = pd.read_csv(dir_script + '/data/test.csv')
df_test.reset_index(inplace=True)

# Comprobamos que tiene el número de filas que tiene que tener
print("-"*37)
print("Registros a procesar para Test:", len(df_test))
print("-"*37)

# Separa la Target del resto de columnas
X_test = df_test.drop(columns=['index', 'Infarto'])
y_test = df_test['Infarto']

# Predicción de Test
y_pred = predic_modelo(modelo_importado, X_test, y_test)

# Exportamos csv con las predicciones
df_test['Predicción'] = y_pred
predictions = pd.DataFrame({'Id': df_test['index'], 'Predicción': df_test['Predicción']})
exporta_predicciones(predictions, dir_script)
