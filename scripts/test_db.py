#!/usr/bin/env python3
"""
scripts/test_db.py

Diagnóstico de tu base PostgreSQL:
 - Conexión
 - Listado de tablas
 - Columnas y tipos
 - Conteo de filas
 - Detección y saneamiento de embeddings
 - (Opcional) Búsqueda vectorial de ejemplo
"""
import os
import sys
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
load_dotenv()   # carga el .env de la carpeta raíz

# Lee parámetros de conexión de tu .env o del entorno
HOST     = os.getenv("POSTGRES_HOST", "localhost")
PORT     = os.getenv("POSTGRES_PORT", 5432)
DBNAME   = os.getenv("POSTGRES_DATABASE") or os.getenv("PG_DB") or os.getenv("PG_NAME")
USER     = os.getenv("PG_USER") or os.getenv("POSTGRES_USERNAME")
PASSWORD = os.getenv("POSTGRES_PASSWORD") or os.getenv("PG_PASS")
# Needed for Azure:
if not all([DBNAME, USER, PASSWORD]):
    print("⚠️  Faltan variables de entorno: asegúrate de PG_DATABASE, PG_USER y PG_PASSWORD.")
    sys.exit(1)

def main():
    dsn = dict(host=HOST, port=PORT, dbname=DBNAME, user=USER, password=PASSWORD, sslmode="require")
    print(f"🔌 Conectando a {HOST}:{PORT}/{DBNAME} …", end=" ")
    try:
        conn = psycopg.connect(**dsn, row_factory=dict_row)
    except Exception as e:
        print(f"\n Error de conexión: {e}")
        sys.exit(1)
    print("OK ✅")

    cur = conn.cursor()

    # 1) Listado de tablas en public
    cur.execute("""
        SELECT table_name
          FROM information_schema.tables
         WHERE table_schema = 'public'
           AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    tablas = [r["table_name"] for r in cur.fetchall()]
    print(f"\n📋 Tablas en public ({len(tablas)}):\n  " + "\n  ".join(tablas))

    for table in tablas:
        print(f"\n––––––––––––––––––––––––––––––––––––––––––––")
        print(f"🗄 Tabla: {table}")

        # 2) Columnas y tipos
        cur.execute("""
            SELECT column_name, data_type 
              FROM information_schema.columns
             WHERE table_schema = 'public'
               AND table_name   = %s
             ORDER BY ordinal_position
        """, (table,))
        cols = cur.fetchall()
        for c in cols:
            print(f"   • {c['column_name']:20}  {c['data_type']}")

        # 3) Conteo de filas
        cur.execute(f"SELECT COUNT(*) AS cnt FROM public.{table};")
        cnt = cur.fetchone()["cnt"]
        print(f"   ↳ filas: {cnt}")

        # 4) Detección de columnas de tipo vector
        #    (pgvector usa la extensión vector, tipo INTERNAL en information_schema)
        cur.execute("""
            SELECT a.attname AS column_name
              FROM pg_attribute a
              JOIN pg_class c ON a.attrelid = c.oid
              JOIN pg_namespace n ON c.relnamespace = n.oid
             WHERE n.nspname = 'public'
               AND c.relname = %s
               AND a.atttypid = (
                    SELECT oid FROM pg_type WHERE typname = 'vector'
               )
               AND a.attnum > 0;
        """, (table,))
        vec_cols = [r["column_name"] for r in cur.fetchall()]

        if vec_cols:
            print(f"   🎯 Embeddings detectados: {vec_cols}")
            for vc in vec_cols:
                cur.execute(f"""
                    SELECT COUNT(*) AS not_null 
                      FROM public.{table}
                     WHERE {vc} IS NOT NULL;
                """)
                nn = cur.fetchone()["not_null"]
                print(f"     • {vc}: {nn} valores no‐nulos ({nn/cnt:.1%} del total)")

            # 5) Prueba de vector‐vector (elige un embedding aleatorio y busca similitud)
            ejemplo_col = vec_cols[0]
            cur.execute(f"""
                SELECT {ejemplo_col} 
                  FROM public.{table}
                 WHERE {ejemplo_col} IS NOT NULL
                 LIMIT 1;
            """)
            ejemplo_emb = cur.fetchone()[ejemplo_col]
            if ejemplo_emb is not None:
                cur.execute(f"""
                    SELECT *, ({ejemplo_col} <-> %s::vector) AS distancia
                      FROM public.{table}
                     WHERE {ejemplo_col} IS NOT NULL
                     ORDER BY distancia
                     LIMIT 3;
                """, (ejemplo_emb,))
                print(f"\n     ► Los 3 elementos más cercanos (distancia):")
                for r in cur.fetchall():
                    print("       -",
                          {k: v for k, v in r.items() if k not in [ejemplo_col, "distancia"]},
                          f"→ {r['distancia']:.4f}")

    conn.close()
    print("\n Diagnóstico completado.")

if __name__ == "__main__":
    main()
