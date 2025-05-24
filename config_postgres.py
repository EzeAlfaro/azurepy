import psycopg2

def get_connection():
    return psycopg2.connect(
        host="ainabi-ainabi.g.aivencloud.com",
        port=14186,
        dbname="defaultdb",
        user="avnadmin",
        password="AVNS_-cITT1QVqP0nWCD-E9E",
        sslmode="require"
    )
