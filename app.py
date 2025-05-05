from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, auth
import subprocess
import os
import json

# Configura Flask y CORS
app = Flask(__name__)
CORS(app)

# Inicializa Firebase
cred = credentials.Certificate(os.path.join(os.path.dirname(__file__), "firebase-service.json"))
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Ruta de Salud ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "message": "El servidor esta funcionando correctamente",  # Sin acentos
        "endpoints": {
            "kmeans": "/api/predict/rotation",
            "regresion": "/api/predict/performance"
        }
    }), 200

# --- Función para ejecutar scripts ---
def run_script(script_path):
    try:
        result = subprocess.run(
            ["python", script_path],
            cwd=os.path.dirname(script_path),
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error en el script: {e.stderr}")
    except json.JSONDecodeError:
        raise Exception("El script no devolvio un JSON valido")  # Sin acentos

# --- Ruta K-Means ---
@app.route('/api/predict/rotation', methods=['POST'])
def predict_rotation():
    try:
        token = request.headers.get('Authorization', '').split(" ")[1]
        auth.verify_id_token(token)
    except Exception as e:
        return jsonify({"error": f"Error de autenticacion: {str(e)}"}), 401  # Sin acentos

    try:
        script_path = os.path.join(os.path.dirname(__file__), "kmeans", "K-Means-Rotacion.py")
        output = run_script(script_path)
        return jsonify(output), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Ruta Regresion ---
@app.route('/api/predict/performance', methods=['POST'])
def predict_performance():
    try:
        token = request.headers.get('Authorization', '').split(" ")[1]
        auth.verify_id_token(token)
    except Exception as e:
        return jsonify({"error": f"Error de autenticacion: {str(e)}"}), 401  # Sin acentos

    try:
        script_path = os.path.join(os.path.dirname(__file__), "regresion", "regresion.py")
        output = run_script(script_path)
        return jsonify(output), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)