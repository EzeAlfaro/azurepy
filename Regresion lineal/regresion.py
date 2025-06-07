import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

def entrenar_modelo(archivo_csv, columnas_excluir=None):

    try:
        df = pd.read_csv(archivo_csv, encoding="utf-8")
        if columnas_excluir is None:
            columnas_excluir = []

        # 1. Eliminar columnas según el usuario
        df = df.drop(columns=columnas_excluir, errors='ignore')

        # 2. Validaciones mínimas
        if 'desempenio_futuro' not in df.columns:
            return json.dumps({"error": "La columna 'desempenio_futuro' es obligatoria"})

            # 3. Transformaciones (solo si las columnas están)
        if 'jerarquia' in df.columns:
            orden_jerarquia = {'trainee': 0, 'junior': 1, 'senior': 2}
            df['jerarquia_ordinal'] = df['jerarquia'].map(orden_jerarquia)

        if 'horas_extra' in df.columns:
            df['horas_extra'] = df['horas_extra'].astype(int)

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        if 'area' in df.columns:
            area_encoded = ohe.fit_transform(df[['area']])
            area_encoded_df = pd.DataFrame(area_encoded, columns=ohe.get_feature_names_out(['area']), index=df.index)
            df = pd.concat([df.drop('area', axis=1), area_encoded_df], axis=1)

        if 'desempenio' in df.columns:
            desempenio_encoded = ohe.fit_transform(df[['desempenio']])
            desempenio_encoded_df = pd.DataFrame(desempenio_encoded, columns=ohe.get_feature_names_out(['desempenio']), index=df.index)
            df = pd.concat([df.drop('desempenio', axis=1), desempenio_encoded_df], axis=1)

        scaler = StandardScaler()

        x = df.drop(['nombre', 'desempenio_futuro'], axis=1)
        y = df['desempenio_futuro']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        model = LinearRegression()
        model.fit(x_train_scaled, y_train)
        coeficientes = dict(zip(x.columns, model.coef_))
        lista_coeficientes = []
        for col, val in coeficientes.items():
            mensaje = f"Coeficiente para {col}: {val:.4f}"
            logging.info(lista_coeficientes)
            lista_coeficientes.append(mensaje)

        # Guardar modelo
        with open(ruta_modelo, 'wb') as archivo:
            pickle.dump({'modelo': model, 'columnas': list(x.columns), 'encoder': ohe}, archivo)

            # Salida JSON
            resultados = {
                "r2_score": f"{r2_score(y_test, model.predict(x_test_scaled))*100:.2f}%",
                "status": "Modelo entrenado y guardado",
                "coeficientes": lista_coeficientes
            }
            return json.dumps(resultados)
    except FileNotFoundError as e:
        return json.dumps({"error": f"No se encontró el archivo: {e.filename}"})
    except KeyError as e:
        return json.dumps({"error": f"Error de clave al procesar los datos: La columna '{e}' no se encontró en el archivo."})
    except Exception as e:
        return json.dumps({"error": f"Ocurrió un error inesperado: {e}"})

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Se debe proporcionar la ruta al archivo CSV como argumento.")
        sys.exit(1)

    archivo_csv = sys.argv[1]
    columnas = []

    if len(sys.argv) >= 3:
        with open(sys.argv[2], 'r') as f:
            columnas = json.load(f)

    resultado = entrenar_modelo(archivo_csv, columnas)
    print(resultado)

archivo_csv = sys.argv[1]
resultado = entrenar_modelo(archivo_csv)
print(resultado) # Imprime el resultado como JSON

