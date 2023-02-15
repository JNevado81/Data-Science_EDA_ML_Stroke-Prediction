# Importamos librerías
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import pickle
from datetime import datetime


# Función para realizar las transformaciones de los datos
def features_engineering(df, convertir_valor=False, campos_nuevos=True, escalar=True):
    df.rename(columns={'sex':'Género', 'age':'Edad', 'hypertension':'Hipertensión', 'heart_disease':'Cardiopatía', 'ever_married':'Casado', 'work_type':'Tipo de trabajo', 
                        'Residence_type':'Tipo de residencia', 'avg_glucose_level':'AVG-Nivel de glucosa', 'bmi':'IMC', 'smoking_status':'Fumador', 
                        'stroke':'Infarto'}, inplace=True)

    df.dropna(inplace=True) # Eliminamos las 3 filas con valores null en sex (género).
    df.drop(df[df['Edad'] < 0].index, inplace=True) # Eliminamos las 58 filas con edades menor a cero.

    if campos_nuevos:
        df['Glucosa_IMC'] = (df['AVG-Nivel de glucosa'] * df['IMC']) / 1000 # Obesidad

        df['Hiper_Cardiopatía'] = df['Hipertensión'] + df['Cardiopatía']

        my_dict = {
                0:'Ninguna',
                1:'Hiper o Cardio',
                2:'Ambas'
                }
        df['Hiper-Cardiopatía'] = df['Hiper_Cardiopatía'].map(my_dict)

    if convertir_valor == False:
        df['Género'] = np.where(df['Género'] == 0, 'Femenino', 'Masculino')

        df['Hipertensión'] = np.where(df['Hipertensión'] == 0, 'No', 'Sí')

        df['Cardiopatía'] = np.where(df['Cardiopatía'] == 0, 'No', 'Sí')

        df['Casado'] = np.where(df['Casado'] == 0, 'No', 'Sí')

        my_dict = {
                0:'Nunca',
                1:'Niños',
                2:'Funcionario',
                3:'Autónomo',
                4:'Privado'
                }
        df['Tipo de trabajo'] = df['Tipo de trabajo'].map(my_dict)

        df['Tipo de residencia'] = np.where(df['Tipo de residencia'] == 0, 'Rural', 'Urbana')

        df['Fumador'] = np.where(df['Fumador'] == 0, 'No', 'Sí')

        df['Infarto'] = np.where(df['Infarto'] == 0, 'No', 'Sí')

    if escalar:
        scaler = MinMaxScaler()
        df[['Edad', 'AVG-Nivel de glucosa', 'IMC', 'Glucosa_IMC']] = scaler.fit_transform(df[['Edad', 'AVG-Nivel de glucosa', 'IMC', 'Glucosa_IMC']])

    return df


# Función para generar el fichero csv de 'Test' para la evaluación final (20%)
def genera_test(df, dir_script):
    X = df[['index', 'Género', 'Hipertensión', 'Cardiopatía', 'Casado', 'AVG-Nivel de glucosa', 'Glucosa_IMC', 'Hiper_Cardiopatía']]
    y = df['Infarto']

    # Separa Train y Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    X_test['Infarto'] = y_test

    # Eliminamos del df_Train las filas generadas para Test
    df['test'] = np.where(df['index'].isin(X_test['index']), 1, 0)
    df.drop(df[(df['test'] == 1)].index, inplace=True)

    # Eliminamos los campos 'index' y 'test' del df_Train
    df.drop(columns=['index', 'test'], inplace=True)

    # Eliminamos los campos 'index' y 'test' del df de test
    X_test.drop(columns=['index'], inplace=True)

    # Generamos fichero Test_Evaluate
    X_test.to_csv(dir_script + '/data/test.csv', index = False) # Muy importante --> index=False

    # Retornamos el df_Train con los cambios generados
    return df


# Función para entrenar el modelo con un algoritmo de clasificación
def entrena_modelo(df):
    X = df[['Género', 'Hipertensión', 'Cardiopatía', 'Casado', 'AVG-Nivel de glucosa', 'Glucosa_IMC', 'Hiper_Cardiopatía']]
    y = df['Infarto']

    # Separa Train y Validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Comprobamos que tiene el número de filas que tiene que tener
    print("-"*40)
    print("Registros a procesar para Train:", len(X_train))
    print("Registros a procesar para Val:", len(X_val))
    print("-"*40)

    # Generamos el modelo
    logit = KNeighborsClassifier(n_neighbors=5)

    # Escalamos los datos
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # Entrenamos el modelo
    logit.fit(X_train_scaled, y_train)

    # Predecimos el modelo
    y_pred = logit.predict(X_train_scaled)

    # Mostramos el scoring
    fpr, tpr, thresh = roc_curve(y_train, y_pred)
    print("-"*30)
    print("Train auc:",auc(fpr, tpr).round(3))
    print("Train roc_auc_score:",roc_auc_score(y_train, y_pred).round(3))
    print("Train accuracy_score:",accuracy_score(y_train, y_pred).round(3))
    print("-"*30)


    return logit, X_val, y_val


# Función para hacer la predicción de los datos
def predic_modelo(logit, X, y):
    # Escalamos los datos
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # Predecimos el modelo
    y_pred = logit.predict(X_scaled)

    # Mostramos el scoring
    fpr, tpr, thresh = roc_curve(y, y_pred)
    print("-"*30)
    print("Predict auc:",auc(fpr, tpr).round(3))
    print("Predict roc_auc_score:",roc_auc_score(y, y_pred).round(3))
    print("Predict accuracy_score:",accuracy_score(y, y_pred).round(3))
    print("-"*30)

    return y_pred


# Función para exportar el modelo
def exporta_modelo(logit, dir_script):
    # Formateamos la fecha y hora actual para el renombrado del fichero a exportar
    now = datetime.now()
    año = now.strftime('%Y')[2:]
    timestamp = año + now.strftime('%m%d%H%M%S')

    # Exportamos el modelo con el renombrado al directorio genérico
    filename = dir_script + '/model/model_' + timestamp
    with open(filename, 'wb') as archivo_salida:
        pickle.dump(logit, archivo_salida)

    # Exportamos el modelo al directorio de producción
    filename_prod = dir_script + '/model/production/model'
    with open(filename_prod, 'wb') as archivo_salida:
        pickle.dump(logit, archivo_salida)
    
    # Mostramos mensaje de info
    print("-"*46)
    print("Modelo exportado a /model y /model/production")
    print("-"*46)


# Función para importar el modelo de producción
def importar_modelo(dir_script):
    filename_prod = dir_script + '/model/production/model'
    with open(filename_prod, 'rb') as archivo_entrada:
        modelo_importado = pickle.load(archivo_entrada)

    return modelo_importado


# Función para exportar un fichero csv con las predicciones
def exporta_predicciones(predictions, dir_script):
    predictions.to_csv(dir_script + '/data/predictions.csv', index = False) # Muy importante --> index=False

    # Mostramos mensaje de info
    print("-"*42)
    print("Fichero de predicciones exportado a /data")
    print("-"*42)
