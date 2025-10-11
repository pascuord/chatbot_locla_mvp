import requests
from bs4 import BeautifulSoup
import time
import sqlite3
import os
import json # Necesario para serializar/deserializar el vector
import numpy as np # Necesario para operaciones vectoriales (aunque solo se usa en app.py, es buena práctica aquí)

# --- CONFIGURACIÓN GLOBAL ---
OLLAMA_API_URL_BASE = "http://localhost:11434" # Base para la API de embeddings
DB_NAME = 'db_cosmetica.db' 
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
PRODUCT_URLS_FILE = 'product_urls.txt'
EMBEDDING_MODEL = "nomic-embed-text" # Modelo para generar embeddings

# =======================================================
# FUNCIONES DE OLLAMA Y BASE DE DATOS
# =======================================================

def generate_ollama_embedding(text, model=EMBEDDING_MODEL):
    """
    Genera un vector de embedding para el texto usando el endpoint /api/embeddings de Ollama.
    """
    try:
        url = f"{OLLAMA_API_URL_BASE}/api/embeddings"
        payload = {
            "model": model,
            "prompt": text
        }
        # Intentamos solo una vez con un tiempo de espera razonable
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        # La respuesta contiene el vector en la clave 'embedding'
        return json.dumps(response.json().get('embedding')) # Devolvemos como JSON string
    except Exception as e:
        print(f"ERROR al generar embedding con Ollama. Asegúrate que Ollama está activo y el modelo '{model}' existe. Error: {e}")
        return None

def setup_db():
    """Asegura que la tabla de productos existe, AHORA CON COMPROBACIÓN DE COLUMNA DE VECTORES."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Crear la tabla si no existe (con el esquema vectorial completo)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS productos_scraped (
            nombre TEXT PRIMARY KEY,
            descripcion TEXT,
            ingredientes TEXT,
            embedding_vector TEXT -- Columna clave
        );
    """)
    conn.commit()
    
    # 2. Verificar si la columna vectorial existe (para usuarios que tenían la versión antigua)
    cursor.execute("PRAGMA table_info(productos_scraped)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'embedding_vector' not in columns:
        print("⚠️ ADVERTENCIA: La columna 'embedding_vector' falta en la base de datos existente.")
        try:
            # 3. Intentar añadir la columna usando ALTER TABLE
            cursor.execute("ALTER TABLE productos_scraped ADD COLUMN embedding_vector TEXT;")
            conn.commit()
            print("✅ Columna 'embedding_vector' añadida exitosamente.")
        except sqlite3.OperationalError as e:
            # Esto puede ocurrir si el archivo de DB estaba bloqueado o corrupto.
            print(f"❌ ERROR al añadir la columna: {e}. Se recomienda borrar el archivo '{DB_NAME}' manualmente y reintentar.")
    
    conn.close()

def save_to_db(data):
    """Guarda o actualiza un producto en la base de datos, incluyendo el vector."""
    if not data or data['embedding_vector'] is None:
        return
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO productos_scraped 
            (nombre, descripcion, ingredientes, embedding_vector) 
            VALUES (?, ?, ?, ?);
        """, (data['nombre'], data['descripcion'], data['ingredientes'], data['embedding_vector']))
        conn.commit()
        print(f"✅ Guardado/Actualizado: {data['nombre']}")
    except Exception as e:
        print(f"Error al guardar en BD: {e}")
    finally:
        conn.close()

def load_product_urls(filename=PRODUCT_URLS_FILE):
    """Carga la lista de URLs desde un archivo de texto."""
    try:
        with open(filename, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        return urls
    except FileNotFoundError:
        print(f"❌ ERROR: El archivo '{filename}' no se encontró. Asegúrate de que existe en esta carpeta.")
        return []

# =======================================================
# FUNCIÓN PRINCIPAL DE SCRAPING (Beautiful Soup)
# =======================================================

def scrape_product_data(url):
    """Extrae datos, aplica la corrección de encoding y genera el vector de embedding."""
    print(f"Procesando: {url}")
    try:
        # 1. Hacemos la petición HTTP
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status() 
        
        # --- CORRECCIÓN DE CODIFICACIÓN (IMPORTANTE) ---
        response.encoding = 'utf-8' 
        
        parser = 'lxml' if os.system('pip show lxml > /dev/null 2>&1') == 0 else 'html.parser'
        soup = BeautifulSoup(response.text, parser)
        
        # --- EXTRACCIÓN DE DATOS ---

        # 1. Nombre
        nombre_element = soup.find('h1', id='product-title')
        nombre = nombre_element.text.strip() if nombre_element else 'Producto Desconocido'
        
        # 2. Descripción
        descripcion = 'N/A'
        descripcion_container = soup.find('div', id='product-description-0')
        if descripcion_container:
            descripcion_element = descripcion_container.find('div', class_='attribute-content')
            descripcion = descripcion_element.text.strip() if descripcion_element else 'N/A'

        # 3. Ingredientes
        ingredientes = 'N/A'
        ingredientes_container = soup.find('div', id='product-description-2')
        if not ingredientes_container:
            ingredientes_container = soup.find('div', class_='product-description-1') # Fallback por si usan otro ID/Clase
        
        if ingredientes_container:
            ingredientes_element = ingredientes_container.find('div', class_='attribute-content')
            ingredientes_texto_crudo = ingredientes_element.text.strip() if ingredientes_element else 'N/A'
            
            # Limpieza de espacios y aviso legal
            ingredientes = ' '.join(ingredientes_texto_crudo.split()).strip()
            limpiador_aviso = "For the latest information, it is recommended to review the ingredient list printed on the packaging of the product prior to usage or consumption."
            ingredientes = ingredientes.replace(limpiador_aviso, '').strip()
            
        
        # 4. Generar el Embedding (Clave para la búsqueda semántica)
        embedding_text = f"Producto: {nombre}. DESCRIPCION: {descripcion} INGREDIENTES: {ingredientes}"
        embedding_vector_json = generate_ollama_embedding(embedding_text)
        
        if embedding_vector_json is None:
            return None # Saltamos el producto si falla la generación del vector

        # 5. Retornar el diccionario con el vector
        return {
            'nombre': nombre,
            'descripcion': descripcion,
            'ingredientes': ingredientes,
            'embedding_vector': embedding_vector_json # Ya está en formato JSON string
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Error HTTP/Conexión al acceder a {url}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error interno de parsing en {url}: {e}")
        return None


# =======================================================
# LÓGICA DE EJECUCIÓN PRINCIPAL
# =======================================================

if __name__ == '__main__':
    # Esta función principal ha sido simplificada para solo llamar a main()
    # Si la base de datos tiene la tabla antigua, setup_db la actualizará.
    
    # Asegúrate de que Ollama está activo y el modelo nomic-embed-text está descargado.
    
    setup_db() 
    product_urls = load_product_urls()

    if not product_urls:
        print("No se encontraron URLs en el archivo. Verifica 'product_urls.txt'. Finalizando.")
    else:
        print(f"Total de {len(product_urls)} URLs cargadas. Iniciando el proceso de extracción y vectorización...")
        
        for i, url in enumerate(product_urls):
            product_data = scrape_product_data(url)
            if product_data:
                save_to_db(product_data)
                
            time.sleep(1) # Pausa ética

    print("\nProceso de extracción y vectorización finalizado. RAG semántico listo.")
