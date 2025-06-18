#!/usr/bin/env python3
import os
import psycopg
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAICOM_KEY"))  # o usa AZURE creds

DSN = dict(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    dbname=os.getenv("PG_DATABASE"),
    sslmode="require",
)

TABLES = {
    "abastecimento": ["placa", "data", "custo_combustivel"],
    "manutencao":     ["placa", "item", "valor"],
    "quilometragem":  ["placa", "odometro_final"],
    "retorno_socorro":["placa", "tipo_retorno_socorro"],
    "veiculos":       ["placa", "fabricante", "modelo_chassi"],
}

def make_text(table, row):
    cols = TABLES[table]
    parts = [f"{col}:{row[col]}" for col in cols]
    return " | ".join(parts)

def get_embedding(text):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return resp.data[0].embedding

def main():
    conn = psycopg.connect(**DSN)
    for table in TABLES:
        print(f"\n➡️  Procesando tabla {table}")
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(f"SELECT ctid, * FROM public.{table} WHERE embedding_3l IS NULL;")
            rows = cur.fetchall()
        for row in rows:
            txt = make_text(table, row)
            emb = get_embedding(txt)
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE public.{table} SET embedding_3l = %s WHERE ctid = %s",
                    (emb, row["ctid"]),
                )
            conn.commit()
        print(f"   ✓ {len(rows)} filas actualizadas")
    conn.close()

if __name__ == "__main__":
    main()
