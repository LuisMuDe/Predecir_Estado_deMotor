import sqlite3

# ========================
# BASE DE DATOS
# ========================
conn = sqlite3.connect("motor_data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL,
    rms REAL,
    freq REAL,
    temp REAL,
    voltaje REAL,
    corriente REAL,
    prediction TEXT,
    binary INTEGER
)
""")

conn.commit()