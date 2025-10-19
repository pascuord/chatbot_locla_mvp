import requests
from bs4 import BeautifulSoup
import time
import sqlite3
import os
import json # Necesario para serializar/deserializar el vector
import re # Necesario para limpiar el precio

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
    """Asegura que la tabla de productos existe, AHORA CON COMPROBACIÓN DE COLUMNA DE VECTORES Y PRECIO."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Crear la tabla si no existe (con el esquema vectorial completo)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS productos_scraped (
            nombre TEXT PRIMARY KEY,
            descripcion TEXT,
            ingredientes TEXT,
            precio REAL, 
            embedding_vector TEXT
        );
    """)
    conn.commit()
    
    # 2. Verificar si las columnas faltantes existen
    cursor.execute("PRAGMA table_info(productos_scraped)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'embedding_vector' not in columns:
        print("⚠️ ADVERTENCIA: La columna 'embedding_vector' falta en la base de datos existente.")
        try:
            cursor.execute("ALTER TABLE productos_scraped ADD COLUMN embedding_vector TEXT;")
            conn.commit()
            print("✅ Columna 'embedding_vector' añadida exitosamente.")
        except sqlite3.OperationalError as e:
            print(f"❌ ERROR al añadir la columna 'embedding_vector': {e}. Se recomienda borrar el archivo '{DB_NAME}' manualmente y reintentar.")
    
    # --- NUEVO: Comprobación para la columna 'precio' ---
    if 'precio' not in columns:
        print("⚠️ ADVERTENCIA: La columna 'precio' falta en la base de datos existente.")
        try:
            cursor.execute("ALTER TABLE productos_scraped ADD COLUMN precio REAL;")
            conn.commit()
            print("✅ Columna 'precio' añadida exitosamente.")
        except sqlite3.OperationalError as e:
            print(f"❌ ERROR al añadir la columna 'precio': {e}.")

    conn.close()

def save_to_db(data):
    """Guarda o actualiza un producto en la base de datos, incluyendo el vector y el precio."""
    if not data or data['embedding_vector'] is None:
        return
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # --- CAMBIO: Añadida la columna 'precio' al INSERT ---
        cursor.execute("""
            INSERT OR REPLACE INTO productos_scraped 
            (nombre, descripcion, ingredientes, precio, embedding_vector) 
            VALUES (?, ?, ?, ?, ?);
        """, (data['nombre'], data['descripcion'], data['ingredientes'], data['precio'], data['embedding_vector']))
        conn.commit()
        print(f"✅ Guardado/Actualizado: {data['nombre']} | Precio: {data['precio']}")
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
# FUNCIÓN PRINCIPAL DE SCRAPING (Beautiful Soup) - CORREGIDA
# =======================================================

def scrape_product_data(url):
    """Extrae datos, aplica la corrección de encoding y genera el vector de embedding."""
    print(f"Procesando: {url}")
    try:
        # 1. Hacemos la petición HTTP
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status() 
        response.encoding = 'utf-8' 
        
        parser = 'lxml' if os.system('pip show lxml > /dev/null 2>&1') == 0 else 'html.parser'
        soup = BeautifulSoup(response.text, parser)
        
        # --- EXTRACCIÓN DE DATOS ---

        # 1. Nombre
        nombre_element = soup.find('h1', id='product-title')
        nombre = nombre_element.text.strip() if nombre_element else 'Producto Desconocido'
        
        # 2. Descripción
        descripcion = None
        descripcion_container = soup.find('div', id='product-description-0')
        if descripcion_container:
            descripcion_element = descripcion_container.find('div', class_='attribute-content')
            if descripcion_element:
                desc_text = descripcion_element.text.strip()
                if desc_text:
                    descripcion = desc_text

        # 3. Ingredientes
        ingredientes = None 
        
        ingredientes_container = soup.find('div', id='product-description-1')
        if not ingredientes_container:
            ingredientes_container = soup.find('div', id='product-description-2')
        if not ingredientes_container:
            ingredientes_container = soup.find('div', class_='product-description-1')
        
        if ingredientes_container:
            ingredientes_element = ingredientes_container.find('div', class_='attribute-content')
            
            if ingredientes_element:
                first_p = ingredientes_element.find('p')
                ingredientes_texto_crudo = first_p.text.strip() if first_p else ingredientes_element.text.strip()
                
                ingredientes_limpio = ' '.join(ingredientes_texto_crudo.split()).strip()
                limpiador_aviso = "For the latest information, it is recommended to review the ingredient list printed on the packaging of the product prior to usage or consumption."
                ingredientes_limpio = ingredientes_limpio.replace(limpiador_aviso, '').strip()
                
                if ingredientes_limpio:
                    ingredientes = ingredientes_limpio
        
        # --- 4. Extraer Precio (LÓGICA ROBUSCA MEJORADA) ---
        precio = None
        price_text = None
        try:
            price_container = soup.find('div', id='product-price')
            if price_container:
                
                # Intento 1: Buscar el precio de oferta (HTML 2)
                sale_price_span = price_container.find('span', class_='text-gray-900')
                
                if sale_price_span:
                    price_text = sale_price_span.text.strip()
                else:
                    # Intento 2: Buscar el precio normal (HTML 1)
                    p_tag = price_container.find('p', class_='text-2xl font-medium')
                    if p_tag:
                        regular_price_span = p_tag.find('span')
                        if regular_price_span:
                            price_text = regular_price_span.text.strip()

                # Si hemos encontrado texto de precio, lo limpiamos y convertimos
                if price_text:
                    # Limpiamos todo lo que no sea un dígito, una coma o un punto
                    price_clean = re.sub(r"[^0-9,.]", "", price_text).strip()
                    # Reemplazamos la coma decimal por un punto
                    price_clean = price_clean.replace(',', '.')
                    
                    if price_clean: # Asegurarnos de que no esté vacío después de limpiar
                        precio = float(price_clean)
                    else:
                        print(f"⚠️ No se pudo extraer un número del texto de precio '{price_text}' en {url}")
                else:
                    print(f"⚠️ No se pudo encontrar un span de precio válido en {url}")

        except Exception as e:
            print(f"⚠️ Error al procesar el precio para {url}: {e}")
            precio = None # Se guardará como NULL

        
        # --- 5. Generar el Embedding (Mejorado para manejar None y añadir precio) ---
        descripcion_para_embedding = descripcion if descripcion is not None else ""
        ingredientes_para_embedding = ingredientes if ingredientes is not None else ""
        precio_para_embedding = f"Precio: {precio}€" if precio is not None else ""

        embedding_text = f"Producto: {nombre}. {precio_para_embedding}. DESCRIPCION: {descripcion_para_embedding} INGREDIENTES: {ingredientes_para_embedding}"
        embedding_vector_json = generate_ollama_embedding(embedding_text)
        
        if embedding_vector_json is None:
            return None # Saltamos el producto si falla la generación del vector

        # --- 6. Retornar el diccionario con el vector y el precio ---
        return {
            'nombre': nombre,
            'descripcion': descripcion,
            'ingredientes': ingredientes,
            'precio': precio, # <-- ¡Nueva columna añadida!
            'embedding_vector': embedding_vector_json
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
    # Borra la base de datos anterior si quieres empezar de cero
    # (Recomendado después de un cambio de esquema)
    # if os.path.exists(DB_NAME):
    #     os.remove(DB_NAME)
    #     print(f"Base de datos '{DB_NAME}' eliminada para reconstrucción.")
    
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