import os
import subprocess
import sys
import time
from dotenv import load_dotenv

try:
    import psycopg
except ImportError:
    print("La librería 'psycopg' no está instalada. Por favor, ejecute 'pip install psycopg' en su entorno.")
    sys.exit(1)

# --- CONFIGURACIÓN ---
# Carga las variables de entorno del archivo .env
load_dotenv()

# Variables de entorno necesarias
REQUIRED_ENV_VARS = [
    "PG_HOST", "PG_DATABASE", "PG_USER", "PG_PASSWORD",
    "OPENAI_CHAT_HOST", "OPENAICOM_KEY"
]


TARGET_TABLE = "abastecimento"
EMBEDDING_COLUMN = "embedding_main"
VECTOR_DIMENSION = 1536 # La dimensión que decidimos usar

# --- FUNCIONES DE VERIFICACIÓN ---

def check_env_vars():
    """Comprueba si todas las variables de entorno necesarias están definidas."""
    print("1. Verificando variables de entorno...")
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        print(f"\n❌ ERROR: Faltan las siguientes variables en tu archivo .env: {', '.join(missing_vars)}")
        sys.exit(1)
    print("   ✅ Todas las variables requeridas están presentes.")
    return True

def check_db_connection():
    """Comprueba si la conexión a la base de datos es exitosa."""
    print("\n2. Verificando la conexión a la base de datos...")
    try:
        conn = psycopg.connect(
            host=os.getenv("PG_HOST"),
            dbname=os.getenv("PG_DATABASE"),
            user=os.getenv("PG_USER"),
            password=os.getenv("PG_PASSWORD"),
            connect_timeout=10,
            sslmode="prefer"
        )
        conn.close()
        print("   ✅ Conexión a la base de datos exitosa.")
        return True
    except Exception as e:
        print(f"\n❌ ERROR al conectar a la base de datos: {e}")
        print("   Por favor, verifica tus credenciales en el archivo .env y las reglas del firewall de Azure.")
        sys.exit(1)

def check_db_schema():
    """Comprueba si la extensión pgvector está habilitada y si la tabla objetivo tiene la columna de embedding."""
    print("\n3. Verificando la estructura de la base de datos...")
    conn = None
    try:
        conn = psycopg.connect(host=os.getenv("PG_HOST"), dbname=os.getenv("PG_DATABASE"), user=os.getenv("PG_USER"), password=os.getenv("PG_PASSWORD"), sslmode="prefer")
        cur = conn.cursor()

        # Verificar extensión pgvector
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone() is None:
            print("\n❌ ERROR: La extensión 'vector' (pgvector) no está habilitada en la base de datos.")
            print("   Pídele al administrador de Azure que la active.")
            sys.exit(1)
        print("   ✅ Extensión 'vector' encontrada.")

        # Verificar columna y su dimensión
        cur.execute("""
            SELECT atttypmod
            FROM   pg_attribute
            WHERE  attrelid = %s::regclass
            AND    attname = %s;
        """, (f'public.{TARGET_TABLE}', EMBEDDING_COLUMN))
        result = cur.fetchone()
        
        if result is None:
            print(f"\n❌ ERROR: La columna '{EMBEDDING_COLUMN}' no existe en la tabla '{TARGET_TABLE}'.")
            print("   Ejecuta el comando SQL para añadirla: ")
            print(f"   ALTER TABLE public.{TARGET_TABLE} ADD COLUMN {EMBEDDING_COLUMN} vector({VECTOR_DIMENSION});")
            sys.exit(1)
            
        dimension = result[0]
        if dimension != VECTOR_DIMENSION:
            print(f"\n❌ ERROR: La dimensión de la columna '{EMBEDDING_COLUMN}' es {dimension}, pero se esperaba {VECTOR_DIMENSION}.")
            print("   Ejecuta el comando SQL para corregirla: ")
            print(f"   ALTER TABLE public.{TARGET_TABLE} ALTER COLUMN {EMBEDDING_COLUMN} TYPE vector({VECTOR_DIMENSION});")
            sys.exit(1)

        print(f"   ✅ Columna '{EMBEDDING_COLUMN}' con la dimensión correcta ({VECTOR_DIMENSION}) encontrada en la tabla '{TARGET_TABLE}'.")
        return True

    finally:
        if conn:
            conn.close()

def check_embeddings_status():
    """Comprueba cuántas filas ya tienen embeddings generados."""
    print("\n4. Verificando el estado de los embeddings...")
    conn = None
    try:
        conn = psycopg.connect(host=os.getenv("PG_HOST"), dbname=os.getenv("PG_DATABASE"), user=os.getenv("PG_USER"), password=os.getenv("PG_PASSWORD"), sslmode="prefer")
        cur = conn.cursor()

        cur.execute(f"SELECT COUNT(*) FROM public.{TARGET_TABLE};")
        total_rows = cur.fetchone()[0]

        cur.execute(f"SELECT COUNT(*) FROM public.{TARGET_TABLE} WHERE {EMBEDDING_COLUMN} IS NOT NULL;")
        embedded_rows = cur.fetchone()[0]

        if total_rows > 0:
            percentage = (embedded_rows / total_rows) * 100
            print(f"   ✅ {embedded_rows} de {total_rows} filas ({percentage:.1f}%) en la tabla '{TARGET_TABLE}' tienen embeddings.")
            if percentage < 100:
                print("\n   ⚠️ ¡Atención! No todas las filas tienen embeddings.")
                print("      Ejecuta 'python scripts/generate_embeddings.py' para procesar las filas restantes.")
                sys.exit(1)
            return True
        else:
            print("   ✅ La tabla está vacía, no se necesita generar embeddings.")
            return True
            
    finally:
        if conn:
            conn.close()

def launch_servers():
    """Lanza los servidores del backend y del frontend en nuevas terminales."""
    print("\n5. ¡Todo listo! Iniciando la aplicación...")
    
    backend_command = "conda activate rag_env && uvicorn fastapi_app:create_app --factory --host 0.0.0.0 --port 8000"
    frontend_command = "cd src/frontend && npm start"
    
    try:
        if sys.platform == "win32":
            print("   Iniciando backend en una nueva ventana de terminal...")
            subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', backend_command])
            time.sleep(5) # Dar tiempo al backend para que empiece a iniciar
            print("   Iniciando frontend en una nueva ventana de terminal...")
            subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', frontend_command])
        else: # para macOS y Linux
            print("Por favor, ejecuta los siguientes comandos en dos terminales separadas:")
            print(f"Terminal 1 (Backend): {backend_command}")
            print(f"Terminal 2 (Frontend): {frontend_command}")
            
        print("\n🚀 ¡Aplicación iniciada! Revisa las nuevas ventanas de terminal y abre tu navegador.")
        
    except Exception as e:
        print(f"\n❌ ERROR al intentar iniciar los servidores: {e}")
        print("Por favor, ejecuta los siguientes comandos manualmente en dos terminales separadas:")
        print(f"Terminal 1 (Backend): {backend_command}")
        print(f"Terminal 2 (Frontend): {frontend_command}")

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print("--- Iniciando Verificación y Lanzamiento de la Aplicación RAG ---")
    if check_env_vars() and check_db_connection() and check_db_schema() and check_embeddings_status():
        launch_servers()
    print("\n--- Proceso de verificación finalizado. ---")