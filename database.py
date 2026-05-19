import sqlite3

conn = sqlite3.connect("riwayat.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS riwayat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT,
    umur INTEGER,
    gender TEXT,
    gejala TEXT,
    alergi TEXT,
    obat TEXT,
    dosis TEXT,
    efek TEXT
)
""")

conn.commit()
conn.close()

print("Database berhasil dibuat / terhubung.")