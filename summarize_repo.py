import os

# --- CONFIGURACIÓN ---
# Directorios que el script ignorará por completo.
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', 'dist', 'build', 'evals', 'infra'}

# Extensiones de archivo que se considerarán "pesados" y se truncarán.
TRUNCATE_EXTENSIONS = {
    '.csv', '.json', '.jsonl', '.lock', '.yaml', '.yml', '.md',
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', '.ipynb',
    '.txt' # Truncamos también los .txt que pueden ser largos
}

# Tamaño máximo en bytes para archivos que no están en la lista de arriba.
# Si un archivo de código es muy grande, también se truncará.
MAX_FILE_SIZE = 150 * 1024  # 150 KB

# Número de líneas que se mostrarán de los archivos truncados.
TRUNCATE_LINES = 10

# Nombre del archivo de salida.
OUTPUT_FILENAME = "repo_summary.txt"

def should_ignore(path, root_dir):
    """Comprueba si una ruta debe ser ignorada."""
    relative_path = os.path.relpath(path, root_dir)
    parts = relative_path.split(os.sep)
    # Ignora si cualquier parte del path está en IGNORE_DIRS o es un venv
    for part in parts:
        if part in IGNORE_DIRS or part.endswith(('_env', '.venv')):
            return True
    return False

def process_repo(root_dir, output_file):
    """
    Recorre el repositorio y crea un archivo de texto con el contenido,
    truncando los archivos pesados o no relevantes para la lógica.
    """
    # Abre el archivo de salida en modo escritura con codificación UTF-8
    with open(output_file, 'w', encoding='utf-8', errors='ignore') as out_f:
        # Recorre todos los directorios y archivos desde la raíz
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
            # Filtra los directorios que deben ser ignorados
            dirnames[:] = [d for d in dirnames if not should_ignore(os.path.join(dirpath, d), root_dir)]

            for filename in filenames:
                file_path = os.path.join(dirpath, filename)

                if should_ignore(file_path, root_dir):
                    continue
                
                # Obtiene la ruta relativa para mostrar en el resumen
                relative_path = os.path.relpath(file_path, root_dir)
                print(f"Procesando: {relative_path}")

                # Escribe una cabecera para cada archivo
                out_f.write("=" * 48 + "\n")
                out_f.write(f"FILE: {relative_path.replace('\\', '/')}\n")
                out_f.write("=" * 48 + "\n")

                try:
                    file_size = os.path.getsize(file_path)
                    _, extension = os.path.splitext(filename.lower())

                    # Decide si truncar el archivo basado en extensión o tamaño
                    if extension in TRUNCATE_EXTENSIONS or file_size > MAX_FILE_SIZE:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                            for i, line in enumerate(in_f):
                                if i >= TRUNCATE_LINES:
                                    break
                                out_f.write(line)
                        out_f.write(f"\n... (Archivo truncado - Tamaño: {file_size} bytes) ...\n\n")
                    else:
                        # Si es un archivo de lógica, escribe el contenido completo
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as in_f:
                            content = in_f.read()
                            out_f.write(content)
                        out_f.write("\n\n")

                except Exception as e:
                    out_f.write(f"... (No se pudo leer el archivo: {e}) ...\n\n")

if __name__ == "__main__":
    # Define la ruta del repositorio ('.' significa el directorio actual)
    repo_path = "."
    print(f"Iniciando el procesamiento del repositorio en: '{os.path.abspath(repo_path)}'")
    
    process_repo(repo_path, OUTPUT_FILENAME)
    
    print("\n¡Proceso completado!")
    print(f"El archivo de resumen '{OUTPUT_FILENAME}' ha sido generado.")