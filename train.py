# Importamos librerías
from utils.funciones import *
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


# Guardamos en una variable el directorio del script
dir_script = os.path.dirname(os.path.abspath(__file__))

# Cargamos el dataset
df = pd.read_csv(dir_script + '/data/train.csv')
df.reset_index(inplace=True)

# Comprobamos que tiene el número de filas que tiene que tener
print("-"*50)
print("Registros a procesar para Train, Val y Test:", len(df))
print("-"*50)

# Realiza las transformaciones de los datos
df = features_engineering(df, convertir_valor=True)

# Genera fichero csv de 'Test' para la evaluación final (20%)
df = genera_test(df, dir_script)

# Entrena el modelo con un algoritmo de clasificación
logit, X_val, y_val = entrena_modelo(df)

# Predicción de Validation
predic_modelo(logit, X_val, y_val)

# Exporta el modelo
exporta_modelo(logit, dir_script)
