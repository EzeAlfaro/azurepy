from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, auth
import subprocess
import os
import json
import logging

# Configura Flask y CORS
app = Flask(__name__)
CORS(app)

# Inicializa Firebase
cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), "firebase-service.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Variable de entorno para controlar la autenticación
ENABLE_AUTH = os.environ.get("ENABLE_AUTH", "False").lower() == "true"  # Cambiado a False por defecto

# --- Ruta de inicio ---
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Bienvenido a la API de predicción",
        "status": "OK",
        "endpoints_disponibles": ["/health", "/api/predict/rotation", "/api/predict/performance", "/test", "/api/predict/future_performance"]
    }), 200

# --- Ruta de Salud ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "message": "El servidor está funcionando correctamente",
        "endpoints": {
            "kmeans": "/api/predict/rotation",
            "regresion": "/api/predict/performance",
            "future_performance": "/api/predict/future_performance" #agrego el nuevo endpoint al diccionario
        }
    }), 200

# --- Función para ejecutar scripts ---
def run_script(script_path, archivo_path=None): # Ahora acepta un segundo argumento opcional
    try:
        command = ["python", script_path]
        if archivo_path:
            command.append(archivo_path)  # Agrega el argumento si se proporciona
        result = subprocess.run(
            command,
            cwd=os.path.dirname(script_path),
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)  # Se espera que el script devuelva un JSON
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error en el script: {e.stderr}")
    except json.JSONDecodeError:
        raise Exception("El script no devolvió un JSON válido")

# --- Ruta K-Means ---
@app.route('/api/predict/rotation', methods=['POST'])
def predict_rotation():
    if ENABLE_AUTH:
        try:
            token = request.headers.get('Authorization', '').split(" ")[1]
            auth.verify_id_token(token)
        except Exception as e:
            return jsonify({"error": f"Error de autenticación: {str(e)}"}), 401
    else:
        logging.info("Autenticación deshabilitada para /api/predict/rotation")

    try:
        script_path = os.path.join(os.path.dirname(__file__), "K-Means", "K-Means-Rotacion.py")
        output = run_script(script_path)
        return jsonify(output), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Ruta Regresión ---
@app.route('/api/predict/performance', methods=['POST'])
def predict_performance():
    if ENABLE_AUTH:
        try:
            token = request.headers.get('Authorization', '').split(" ")[1]
            auth.verify_id_token(token)
        except Exception as e:
            return jsonify({"error": f"Error de autenticación: {str(e)}"}), 401
    else:
        logging.info("Autenticación deshabilitada para /api/predict/performance")

    try:
        script_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "regresion.py")
        output = run_script(script_path)
        return jsonify(output), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Ruta para predecir el rendimiento futuro ---
@app.route('/api/predict/future_performance', methods=['POST'])
def predict_future_performance():
    if ENABLE_AUTH:
        try:
            token = request.headers.get('Authorization', '').split(" ")[1]
            auth.verify_id_token(token)
        except Exception as e:
            return jsonify({"error": f"Error de autenticación: {str(e)}"}), 401
    else:
        logging.info("Autenticación deshabilitada para /api/predict/future_performance")

    try:
        # Verifica si se envió un archivo CSV
        if 'file' not in request.files:
            return jsonify({"error": "No se proporcionó ningún archivo CSV"}), 400

        archivo_csv = request.files['file']

        # Verifica si el archivo tiene un nombre y es un CSV
        if archivo_csv.filename == '' or not archivo_csv.filename.endswith('.csv'):
            return jsonify({"error": "Por favor, sube un archivo CSV válido"}), 400

        # Guarda el archivo CSV temporalmente
        archivo_temporal_path = os.path.join(os.path.dirname(__file__), 'temp.csv')
        archivo_csv.save(archivo_temporal_path)

        script_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "predecir_rendimiento_futuro.py")
        # Llama al script y pasa la ruta del archivo CSV como argumento
        output = run_script(script_path, archivo_temporal_path)

        # Si la salida del script es una cadena, conviértela a un objeto JSON
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                # Si no se puede convertir a JSON, devuelve un mensaje de error
                return jsonify({"error": "El script no devolvió un JSON válido: " + output}), 500

        # Elimina el archivo temporal
        os.remove(archivo_temporal_path)

        return jsonify(output), 200
    except Exception as e:
        # Registra el error para ayudar a depurar
        logging.error(f"Error al predecir el rendimiento futuro: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_page():
    return send_file('test.html')

# --- Ejecutar la app ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
