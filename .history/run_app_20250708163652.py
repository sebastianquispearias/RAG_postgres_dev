# test_connection.py
import os
import psycopg

# lee las mismas vars que tú pusiste en .env
host     = "zane-linave.postgres.database.azure.com"
port     = 5432
dbname   = "dev_linave"
user     = "sebastian"
password = "Dtj78frWb2"

conn = psycopg.connect(
    host=host,
    port=port,
    dbname=dbname,
    user=user,
    password=password,
    sslmode="require"
)

with conn.cursor() as cur:
    cur.execute("SELECT NOW(), COUNT(*) FROM abastecimento;")
    now, cnt = cur.fetchone()
    print(f"✓ Conectado a linave @ {now}, tiene {cnt} filas en abastecimento")

conn.close()
