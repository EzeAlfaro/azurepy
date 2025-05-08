import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import pickle
import json
import os
import logging
import sys

#Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rutas relativas
#ruta_entrenamiento = os.path.join(os.path.dirname(__file__), "DataSet_entrenamiento.csv")
ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_desempenio_futuro.pkl")  # Sin acento

#logging.info(f"Intentando cargar dataset desde: {ruta_entrenamiento}")  # Log

def entrenar_modelo(archivo_csv):

    try:
        df = pd.read_csv(archivo_csv, encoding="utf-8")

        # --- Código original de Santi (sin caracteres especiales) ---
        orden_desempeno = {'bajo': 0, 'medio': 1, 'alto': 2}  # Sin acento
        df['desempenio_ordinal'] = df['desempenio'].map(orden_desempeno)

        orden_jerarquia = {'trainee': 0, 'junior': 1, 'senior': 2}
        df['jerarquia_ordinal'] = df['jerarquia'].map(orden_jerarquia)

        df['horas_extra'] = df['horas_extra'].astype(int)

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        area_encoded = ohe.fit_transform(df[['area']])
        area_encoded_df = pd.DataFrame(area_encoded, columns=ohe.get_feature_names_out(['area']), index=df.index)

        df_final = pd.concat([df.drop(['area', 'jerarquia', 'desempenio'], axis=1), area_encoded_df], axis=1)

        x = df_final.drop(['nombre', 'desempenio_futuro'], axis=1)
        y = df_final['desempenio_futuro']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Guardar modelo
        with open(ruta_modelo, 'wb') as archivo:
            pickle.dump({'modelo': model, 'columnas': list(x.columns), 'encoder': ohe}, archivo)

        # Salida JSON
            resultados = {
                "r2_score": f"{r2_score(y_test, model.predict(x_test))*100:.2f}%",
                "status": "Modelo entrenado y guardado"
            }
            return json.dumps(resultados)
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
resultado = entrenar_modelo(archivo_csv)
print(resultado) # Imprime el resultado como JSON

