from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, auth
import subprocess
import os
import json
from config_postgres import get_connection
from psycopg2.extras import execute_values
import logging
import jwt
import time

# Configura Flask y CORS
app = Flask(__name__)
CORS(app)

# =========================================================================
# === INICIO DE CONFIGURACIÓN CON DATOS HARDCODEADOS (NO RECOMENDADO PARA PRODUCCIÓN) ===
# =========================================================================

# Configuración de Firebase
# ASEGÚRATE de que 'firebase-service.json' esté en la misma carpeta que este 'app.py'
FIREBASE_SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(__file__), "firebase-service.json")

# Configuración de Metabase
# Asegúrate de que esta URL sea la de tu instancia de Metabase (cambia si no es localhost:3000)
METABASE_SITE_URL = "http://localhost:3000"
# ¡MANTÉN ESTA CLAVE SÚPER SEGURA! Obtenla de la configuración de incrustación de Metabase.
METABASE_SECRET_KEY = "940957bdbb152c1ff9c817f89d16f68e81b6b30e39b362567a4c23a71a4dd388"
# El ID de la pregunta/gráfico de Metabase que quieres incrustar.
# Encuéntralo en la URL de tu gráfico en Metabase (e.g., /question/42-tu-grafico)
METABASE_QUESTION_ID = 42 # <-- ¡CAMBIA ESTO POR EL ID REAL DE TU GRÁFICO!

# Variable para controlar la autenticación (True/False)
# Si lo pones en True, necesitarás manejar la autenticación de Firebase.
ENABLE_AUTH = False # Cambiado a False para simplificar las pruebas iniciales

# =========================================================================
# === FIN DE CONFIGURACIÓN CON DATOS HARDCODEADOS ===
# =========================================================================


# Inicializa Firebase
try:
    cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logging.info("Firebase inicializado correctamente.")
except Exception as e:
    logging.error(f"Error al inicializar Firebase: {e}")
    # Considera una forma de manejar esto si Firebase es crítico para la app.

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Función para generar la URL de incrustación de Metabase ---
def generate_metabase_iframe_url(question_id):
    """
    Genera una URL de incrustación segura para un gráfico de Metabase.
    """
    payload = {
      "resource": {"question": question_id},
      "params": {}, # Puedes añadir parámetros de filtro aquí si los necesitas
      "exp": round(time.time()) + (60 * 10) # 10 minutos de expiración
    }
    # Asegúrate de que la clave secreta no esté vacía o None
    if not METABASE_SECRET_KEY:
        raise ValueError("METABASE_SECRET_KEY no está configurada. No se puede generar el token JWT.")
    
    token = jwt.encode(payload, METABASE_SECRET_KEY, algorithm="HS256")
    iframe_url = f"{METABASE_SITE_URL}/embed/question/{token}#bordered=true&titled=true"
    return iframe_url

# --- Ruta de inicio ---
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Bienvenido a la API de predicción",
        "status": "OK",
        "endpoints_disponibles": ["/health", "/api/predict/rotation", "/api/predict/performance_train", "/test", "/api/predict/future_performance", "/interfaz"]
    }), 200

# --- Ruta de Salud ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "message": "El servidor está funcionando correctamente",
        "endpoints": {
            "kmeans": "/api/predict/rotation",
            "entrenar_regresion": "/api/predict/performance_train",
            "future_performance": "/api/predict/future_performance"
        }
    }), 200

# --- Función para ejecutar scripts ---
def run_script(script_path, archivo_path=None):
    try:
        command = ["python", script_path]
        if archivo_path:
            command.append(archivo_path)
        result = subprocess.run(
            command,
            cwd=os.path.dirname(script_path),
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
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
@app.route('/api/predict/performance_train', methods=['POST'])
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
        if 'file' not in request.files:
            return jsonify({"error": "No se proporcionó ningún archivo CSV"}), 400
        archivo_csv = request.files['file']
        if archivo_csv.filename == '' or not archivo_csv.filename.endswith('.csv'):
            return jsonify({"error": "Por favor, sube un archivo CSV válido"}), 400
        archivo_temporal_path = os.path.join(os.path.dirname(__file__), 'temp.csv')
        archivo_csv.save(archivo_temporal_path)

        script_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "regresion.py")
        output = run_script(script_path, archivo_temporal_path)

        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                return jsonify({"error": "El script no devolvió un JSON válido: " + output}), 500
        os.remove(archivo_temporal_path)
        return jsonify(output), 200
    except Exception as e:
        logging.error(f"Error al predecir el rendimiento futuro: {e}")
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
        if 'file' not in request.files:
            return jsonify({"error": "No se proporcionó ningún archivo CSV"}), 400
        archivo_csv = request.files['file']
        if archivo_csv.filename == '' or not archivo_csv.filename.endswith('.csv'):
            return jsonify({"error": "Por favor, sube un archivo CSV válido"}), 400
        archivo_temporal_path = os.path.join(os.path.dirname(__file__), 'temp.csv')
        archivo_csv.save(archivo_temporal_path)

        script_path = os.path.join(os.path.dirname(__file__), "Regresion lineal", "predecir_rendimiento_futuro.py")
        output = run_script(script_path, archivo_temporal_path)

        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                return jsonify({"error": "El script no devolvió un JSON válido: " + output}), 500
        os.remove(archivo_temporal_path)
        return jsonify(output), 200
    except Exception as e:
        logging.error(f"Error al predecir el rendimiento futuro: {e}")
        return jsonify({"error": str(e)}), 500
    
# -- Ruta para guardar resultados de regresión lineal en BD --
@app.route('/api/predict/save_results', methods=['POST'])
def guardar_regresión():
    datos = request.json.get('resultados')
    if not datos:
        return jsonify({"error": "No hay datos para guardar"}), 400
    
    # Conexión y query para insertar varios registros rápido
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Asumiendo que cada dict en datos tiene las columnas: Nombre, AusenciasInjustificadas, etc.
        # Ajustá los nombres y orden según tu tabla
        query = """
            INSERT INTO regresion
            (nombre, area, jerarquia, desempenio, cantidad_proyectos, personas_equipo, horas_extra, rendimiento_futuro, puntaje, asistencia_puntualidad)
            VALUES %s
        """
        
        valores = [
            (
                d['nombre'],
                d['area'],
                d['jerarquia'],
                d['desempenio'],
                d['cantidad_proyectos'],
                d['personas_equipo'],
                d['horas_extra'],
                d['rendimiento_futuro'],
                d['puntaje'],
                d['asistencia_puntualidad']
            ) for d in datos
        ]

        execute_values(cursor, query, valores)
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"mensaje": "Datos guardados en PostgreSQL exitosamente"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['GET'])
def test_page():
    return send_file('test.html')

# --- Modificación del endpoint /interfaz ---
@app.route('/interfaz', methods=['GET'])
def interfaz_page():
    # Usa el ID de la pregunta de Metabase hardcodeado
    metabase_iframe_url = generate_metabase_iframe_url(METABASE_QUESTION_ID)

    # Lee el contenido de tu HTML
    with open('interfaz.html', 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Reemplaza el marcador de posición en el HTML con la URL generada.
    html_content = html_content.replace('METABASE_IFRAME_URL_PLACEHOLDER', metabase_iframe_url)

    return render_template_string(html_content)

# --- Ejecutar la app ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)