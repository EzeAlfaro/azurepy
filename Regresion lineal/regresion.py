import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import pickle
import json
import os
import logging

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Rutas relativas
ruta_entrenamiento = os.path.join(os.path.dirname(__file__), "DataSet_entrenamiento.csv")
ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_desempeno_futuro.pkl")  # Sin acento

logging.info(f"Intentando cargar dataset desde: {ruta_entrenamiento}")  # Log

try:
    df = pd.read_csv(ruta_entrenamiento, encoding="utf-8")

    # --- Código original de Santi (sin caracteres especiales) ---
    orden_desempeno = {'bajo': 0, 'medio': 1, 'alto': 2}  # Sin acento
    df['desempeno_ordinal'] = df['desempeno'].map(orden_desempeno)

    orden_jerarquia = {'trainee': 0, 'junior': 1, 'senior': 2}
    df['jerarquia_ordinal'] = df['jerarquia'].map(orden_jerarquia)

    df['horas_extra'] = df['horas_extra'].astype(int)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    area_encoded = ohe.fit_transform(df[['area']])
    area_encoded_df = pd.DataFrame(area_encoded, columns=ohe.get_feature_names_out(['area']), index=df.index)

    df_final = pd.concat([df.drop(['area', 'jerarquia', 'desempeno'], axis=1), area_encoded_df], axis=1)

    x = df_final.drop(['nombre', 'desempeno_futuro'], axis=1)
    y = df_final['desempeno_futuro']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Guardar modelo
    with open(ruta_modelo, 'wb') as archivo:
        pickle.dump({'modelo': model, 'columnas': list(x.columns), 'encoder': ohe}, archivo)

    # Salida JSON
    if __name__ == "__main__":
        resultados = {
            "r2_score": r2_score(y_test, model.predict(x_test)),
            "status": "Modelo entrenado y guardado"
        }
        print(json.dumps(resultados))

except FileNotFoundError:
    logging.error(f"No se encontró el archivo: {ruta_entrenamiento}")
    print(json.dumps({"error": f"No se encontró el archivo: {ruta_entrenamiento}"}))
    exit()

except Exception as e:
    logging.exception("Ocurrió un error inesperado")
    print(json.dumps({"error": str(e)}))
