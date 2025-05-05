from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Hola desde Flask en Azure 🚀"

if __name__ == "__main__":
    app.run()
