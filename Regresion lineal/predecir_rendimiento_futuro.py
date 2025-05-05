#aca realizamos la prediccion con el modelo ya entrenado guardado en modelo_desempeño_futuro.pkl
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import pickle

# Define la ruta del archivo donde guardaste el modelo entrenado
modelo_guardado_path = 'backend-ml/Regresion lineal/modelo_desempeño_futuro.pkl'

# Define la ruta del nuevo archivo CSV CON la columna 'desempeño'
nuevo_data_filepath = 'backend-ml/Regresion lineal/DataSet_para_predecir.csv'

try:
    # Carga el modelo entrenado desde el archivo
    with open(modelo_guardado_path, 'rb') as archivo_cargado:
        datos_cargados = pickle.load(archivo_cargado)

    modelo_cargado = datos_cargados['modelo']
    columnas_entrenamiento = datos_cargados['columnas']
    ohe = datos_cargados['encoder']

    # Lee el nuevo archivo CSV con los datos de los nuevos empleados
    nuevos_df = pd.read_csv(nuevo_data_filepath, encoding="utf-8")

    # --- Preprocesamiento de los nuevos datos ---
   #como desempeño tiene orden logico uso codificacion manual para que el modelo entienda que alto>medio>bajo
    orden_desempeño = {'bajo': 0, 'medio': 1, 'alto': 2}
    nuevos_df['desempeño_ordinal'] = nuevos_df['desempeño'].map(orden_desempeño)
    #lo mismo con jerarquia
    orden_jerarquia = {'trainee': 0, 'junior': 1, 'senior': 2}
    nuevos_df['jerarquia_ordinal'] = nuevos_df['jerarquia'].map(orden_jerarquia)
    #uso astype para la columna binaria, TRUE y FALSE
    nuevos_df['horas_extra'] = nuevos_df['horas_extra'].astype(int)
    #uso one hot encoder para convertir lo datos categoricos en numeros
    area_encoded = ohe.transform(nuevos_df[['area']])
    area_encoded_df = pd.DataFrame(area_encoded, columns=ohe.get_feature_names_out(['area']), index=nuevos_df.index)

    df_final = pd.concat([nuevos_df.drop(
        ['area', 'jerarquia', 'desempeño'], axis=1), area_encoded_df],
        axis=1)

    # 4. Seleccionar las columnas de características (en el mismo orden que se usaron para entrenar)
    x_nuevos = df_final[columnas_entrenamiento]

    # --- Realizar la predicción ---
    predicciones_futuras = modelo_cargado.predict(x_nuevos)

    # --- Mostrar las predicciones ---
    print("\nPredicciones de desempeño futuro para los nuevos empleados:")
    for i, prediccion in enumerate(predicciones_futuras):
        print(f"Empleado {i+1}: {prediccion:.2f}")

except FileNotFoundError as e:
    print(f"Error: No se encontró el archivo '{e.filename}'. Asegúrate de que las rutas sean correctas.")
except KeyError as e:
    print(f"Error de clave al procesar los datos: La columna '{e}' no se encontró en el archivo.")
except Exception as e:
    print(f"Ocurrió un error: {e}")