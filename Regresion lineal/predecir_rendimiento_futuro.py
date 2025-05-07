import pandas as pd
import pickle
import sys
import json  # Importa el módulo json

def predecir_rendimiento_futuro(archivo_csv):
    """
    Realiza la predicción del rendimiento futuro de los empleados basado en los datos proporcionados en un archivo CSV.

    Args:
        archivo_csv (str): La ruta al archivo CSV que contiene los datos de los empleados.

    Returns:
        dict: Un diccionario que contiene los resultados de la predicción.  Si hay un error,
              devuelve un diccionario con una clave "error" y el mensaje de error.
    """
    try:
        # Define la ruta del archivo donde guardaste el modelo entrenado
        modelo_guardado_path = 'modelo_desempenio_futuro.pkl'

        # Carga el modelo entrenado desde el archivo
        with open(modelo_guardado_path, 'rb') as archivo_cargado:
            datos_cargados = pickle.load(archivo_cargado)

        modelo_cargado = datos_cargados['modelo']
        columnas_entrenamiento = datos_cargados['columnas']
        ohe = datos_cargados['encoder']

        # Lee el nuevo archivo CSV con los datos de los nuevos empleados
        nuevos_df = pd.read_csv(archivo_csv, encoding="utf-8")

        # --- Preprocesamiento de los nuevos datos ---
        # como desempeño tiene orden logico uso codificacion manual para que el modelo entienda que alto>medio>bajo
        orden_desempeño = {'bajo': 0, 'medio': 1, 'alto': 2}
        nuevos_df['desempenio_ordinal'] = nuevos_df['desempenio'].map(orden_desempeño)
        # lo mismo con jerarquia
        orden_jerarquia = {'trainee': 0, 'junior': 1, 'senior': 2}
        nuevos_df['jerarquia_ordinal'] = nuevos_df['jerarquia'].map(orden_jerarquia)
        # uso astype para la columna binaria, TRUE y FALSE
        nuevos_df['horas_extra'] = nuevos_df['horas_extra'].astype(int)
        # uso one hot encoder para convertir lo datos categoricos en numeros
        area_encoded = ohe.transform(nuevos_df[['area']])
        area_encoded_df = pd.DataFrame(area_encoded, columns=ohe.get_feature_names_out(['area']), index=nuevos_df.index)

        df_final = pd.concat([nuevos_df.drop(
            ['area', 'jerarquia', 'desempenio'], axis=1), area_encoded_df],
            axis=1)

        # 4. Seleccionar las columnas de características (en el mismo orden que se usaron para entrenar)
        x_nuevos = df_final[columnas_entrenamiento]

        # --- Realizar la predicción ---
        predicciones_futuras = modelo_cargado.predict(x_nuevos)

        # --- Devolver las predicciones como un diccionario ---
        resultados = {"predicciones": [float(pred) for pred in predicciones_futuras]}  # Convierte a float estándar de Python
        return json.dumps(resultados) # Devuelve el diccionario como un string JSON

    except FileNotFoundError as e:
        return json.dumps({"error": f"No se encontró el archivo: {e.filename}"})
    except KeyError as e:
        return json.dumps({"error": f"Error de clave al procesar los datos: La columna '{e}' no se encontró en el archivo."})
    except Exception as e:
        return json.dumps({"error": f"Ocurrió un error inesperado: {e}"})

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: Se debe proporcionar la ruta al archivo CSV como argumento.")
        sys.exit(1)

    archivo_csv = sys.argv[1]
    resultado = predecir_rendimiento_futuro(archivo_csv)
    print(resultado) # Imprime el resultado como JSON
