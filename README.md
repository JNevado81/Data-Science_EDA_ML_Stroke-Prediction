# DATA SCIENCE - EDA - ML - PREDICCIÓN DEL INFARTO CEREBRAL
* Un accidente cerebrovascular o ataque cerebral sucede cuando se detiene el flujo sanguíneo a parte del cerebro. Al no poder recibir el oxígeno y nutrientes que necesitan, las células cerebrales comienzan a morir en minutos. Esto puede causar un daño severo al cerebro, discapacidad permanente e incluso la muerte.

* **¿Y si una persona pudiera saber que tiene una alta probabilidad de sufrir un ictus en los próximos años?**

* Vamos a diseñar un algoritmo que vaticine si un paciente es propenso a desarrollar un infarto cerebral.

* El objetivo es encontrar el algoritmo de clasificación que mejor encaje en los problemas de predicción dadas las características del problema, por lo que se estudiará prácticamente cada tipo de algoritmo de aprendizaje supervisado viendo cómo se comporta.
## Características del conjunto de datos
* Contamos con un dataset de 40.910 registros para realizar el Train, Validation y Test.

* Cada registro muestra información de un paciente.

* Disponemos de 11 columnas: 
    * 4 columnas de tipo float64.
    * 7 columnas de tipo int64, incluida la target --> Infarto (stroke).
### Campos
>**campo_original (campo_renombrado)**

* sex (Género): 1 - Hombre, 0 - Mujer.

* age (Edad): en años.

* hypertension (Hipertensión): indica si el paciente ha tenido alguna vez hipertensión (1 - Sí, 0 - No).

* heart_disease (Cardiopatía): indica si el paciente ha tenido alguna vez enfermedades del corazón (1 - Sí, 0 - No).

* ever_married (Casado): indica si el paciente ha estado casado alguna vez (1 - Sí, 0 - No).

* work_type (Tipo de trabajo): 0 - Nunca trabajó (Nunca), 1 - Trabaja con niños (Niños), 2 - Funcionario, 3 - Autónomo, 4 - sector privado (Privado).

* Residence_type (Tipo de residencia): 1 - Urbana, 0 - Rural.

* avg_glucose_level (AVG-Nivel de glucosa): nivel medio de azúcar en sangre.

* bmi (IMC): índice de masa corporal.

* smoking_status (Fumador): 1 - Sí, 0 - No.

---

* stroke (Infarto): si el paciente ha sufrido un infarto/ictus (1 - Sí, 0 - No).

---
> Género, Tipo de trabajo y Tipo de residencia, aun teniendo valores numéricos, son del tipo categórico nominal. 

> Edad, Nivel de glucosa en sangre y IMC son de tipo decimal.

> El resto de columnas de tipo entero son binarias.
### Información del fichero
* Origen: https://www.kaggle.com/datasets/prosperchuks/health-dataset?select=stroke_data.csv

* Nombre: stroke_data.csv (en el proyecto train.csv)
## Análisis exploratorio de datos: missing, duplicados, eliminación y feature engineering
* Encontramos valores negativas para las edades. Como tenemos bastantes registros para realizar el modelado vamos a eliminar las 58 filas afectadas.

* Para la Edad, Nivel de glucosa en sangre e IMC (índice de masa corporal), debemos aplicar un escalado para conseguir una normalización frente al resto de datos del dataset.

* No hay duplicados.

* Observamos 3 filas con valores nulos en la columna de Género.

* Tanto las 58 filas de edades negativas como las 3 de Género con valores nulos serán eliminadas.

* Transformación a binario: Hipertensión, Cardiopatía, Casado, Fumador e Infarto.

* Transformación a categórico nominal: Género, Tipo de trabajo y Tipo de residencia.

* Campos nuevos:
    * Glucosa_IMC: el producto entre la media del nivel de glucosa en sangre y el IMC.
    
    * Hiper_Cardio: la suma de los valores binarios de Hipertensión y Cardiopatía. Resultado: ninguna de las dos enfermedades (0), una de las dos (1) o ambas (2).

- Al terminar el EDA realizamos los siguientes pasos:

    * Las columnas de tipo categórico nominal las volvemos a transformar en numéricas.

    * Escalamos para dejar los datos acotados entre 0 y 1 (MinMaxScaler): Edad, Nivel de glucosa en sangre, IMC y Glucosa_IMC.

    * Aplicamos un StandardScaler a todos los campos para entrenar y predecir.
## Análisis univariante y bivariante, anomalías, errores, sesgos y outliers
### Univariante
* Los valores en la columna a predecir (target) están bastante equilibrados, al igual que en 'Géneros', 'Tipos de residencia' y 'Fumadores'.

* Por otro lado, hay un porcentaje muy alto de casados. Al contrario para las personas con hipertensión y cardiopatías.

* En Tipos de trabajo, el sector privado se lleva más de la mitad de los datos con un poco más del 62%, seguido por autónomos (22.6%) y funcionarios (13.7%), en total suman casi el 100%, para ser más exactos 98.8% de lo datos totales. Las personas que cuidan a niños (1.1%) y los que nunca han trabajado tienen un porcentaje muy bajo (0.2%). Vemos que la suma total es 100.1% debido a los redondeos.

* Las personas sin hipertensión ni cardiopatías suman 69.7%, casi el 70% de los datos totales. Las que tienen solo una de las dos están en el 26.5%. Sin embargo las que tienen las dos enfermedades no llegan al 4%.
### Bivariante
* El género no influye mucho. Hay más mujeres con riesgo de infarto a que no, al contrario de los hombres, pero en proporción de infartos están muy igualados.

* La vida matrimonial y el tabaquismo tienen mayor porcentaje de sufrir un infarto, al igual que las enfermedades de hipertensión y cardiopatía. Podemos observar que las personas que tienen una de las dos enfermedades o las dos, tienen mayor riesgo, se podría decir que es el índice con mayor riesgo de todos los analizados.

* El trabajo por cuenta propia tiene mayor riesgo que el resto de trabajos.

* Ampliamos la gráfica para los que nunca han trabajado y los que cuidan a niños.
Ninguno de los dos grupos tuvo nunca un infarto. Estos datos pueden causar sesgos, por lo que sería conveniente eliminarlos.

* Para el tipo de residencia hay una leve tendencia a sufrir un infarto en los entornos urbanos.

* Las personas comprendidas entre los 35 y los 65 años tiene un mayor índice de infarto.

* La distribución de la edad y el IMC respecto a la target es bastante similar, vemos que la mediana y los percentiles están altamente igualados.

* Se ve claramente que las personas con el nivel de glucosa alto en sangre tienen más probabilidades de sufrir un infarto que las que tienen valores normales.

* Observamos un outlier en el IMC (no infarto).

* Las columnas Tipo de residencia, IMC, Tipo de trabajo, Edad y Fumador son las menos correlacionadas con la target Infarto (0.01, 0.02, 0.03, 0.06 y 0.07 respectivamente). El resto tienen una mejor correlación con la target, aunque tampoco es muy buena.

* La correlación mayor es de 0.33, le siguen otras de 0.27, 0.26, 0.23 y 0.22 (Hiper_Cardiopatía, AVG-Nivel de glucosa, Hipertensión, Glucosa_IMC y Cardiopatía).

* Vemos una fuerte correlación de Hiper-Cardiopatía con Hipertensión y Cardiopatía. Es normal ya que Hiper-Cardiopatía está creada en base a las otras dos.\
Lo mismo ocurre con Glucosa_IMC y AVG-Nivel de glucosa e IMC.
## Dividir Test para evaluación
* Generamos fichero csv de 'Test' para evaluar (20%) --> test.csv
## Separa Train y Validation
* De los datos restantes (80% del total), un 80% para Train y un 20% para Validation.
## Modelos
* Probamos con los siguientes modelos de **clasificación**: LogisticRegression, GradientBoostingClassifier, XGBClassifier, SVC, RandomForestClassifier, DecisionTreeClassifier, KNeighborsClassifier, BaggingClassifier y AdaBoostClassifier.

* Aplicamos GridSearchCV para conseguir los mejores parámetros.

* El modelo que mejor genera es --> **KNeighborsClassifier(n_neighbors=5)**

## Exportamos el mejor modelo con pickle para poder realizar la evaluación
* El mejor modelo 'KNeighborsClassifier' lo exportamos a un fichero llamado 'model' junto con los datos a predecir 'test.csv'. En 'predict.py' podemos lanzar el proceso.

## Resultados
* Comprobamos los resultados con las siguientes métricas: accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score

* Graficamos los resultados.
## Conclusión
* **Podemos afirmar que el modelo seleccionado hace muy buena predicción de los datos, obteniendo un scoring de 0.991 en el Train y un 0.958 en el Validation.**
