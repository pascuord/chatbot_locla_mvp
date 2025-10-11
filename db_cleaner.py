import os
import requests 
import sqlite3 
import json
import numpy as np
import time

# --- CONFIGURACIÓN GLOBAL (Debe coincidir con app.py) ---
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
DB_NAME = 'db_cosmetica.db'
EMBEDDING_MODEL = "nomic-embed-text" 

# =======================================================
# FUNCIONES DE UTILERÍA
# =======================================================

def generate_ollama_embedding(text):
    """Genera un vector de embedding para el texto usando Ollama."""
    try:
        payload = {
            "model": EMBEDDING_MODEL,
            "prompt": text,
        }
        # Llama al endpoint de embeddings de Ollama
        response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        return response.json()['embedding']
    except Exception as e:
        print(f"    ❌ ERROR al generar embedding: {e}. Verifique el servidor Ollama.")
        return None

def sanitize_text(text):
    """Normaliza texto para corregir errores comunes de codificación (ej. 'atÃ³pica' a 'atópica')."""
    if not text:
        return ""
    
    # Mapeo de errores comunes de ISO-8859-1 a UTF-8 mal interpretados
    text = text.replace('Ã³', 'ó')
    text = text.replace('Ã¡', 'á')
    text = text.replace('Ã©', 'é')
    text = text.replace('Ã­', 'í')
    text = text.replace('Ãº', 'ú')
    text = text.replace('Ã±', 'ñ')
    text = text.replace('Ã‘', 'Ñ')
    text = text.replace('Ãœ', 'Ü')
    text = text.replace('Ã¼', 'ü')
    text = text.replace('â', "'") # Corregir comillas HTML mal codificadas
    text = text.replace('â', "'") # Corregir comillas HTML mal codificadas
    
    return text.strip()

# =======================================================
# LÓGICA DE LIMPIEZA Y VECTORIZACIÓN
# =======================================================

def clean_and_revectorize_db():
    """Procesa todos los registros, corrige la codificación y genera vectores faltantes."""
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Seleccionamos el ID de la fila (ROWID) y los datos necesarios
        cursor.execute("SELECT rowid, nombre, descripcion, embedding_vector FROM productos_scraped")
        rows_to_process = cursor.fetchall()
        
        if not rows_to_process:
            print("Base de datos vacía. No hay registros para limpiar.")
            return

        print(f"Iniciando limpieza de {len(rows_to_process)} registros...")
        
        count_updated = 0
        count_vectorized = 0
        count_deleted = 0
        
        # Diccionario para rastrear los nombres normalizados que ya existen en el proceso de bucle.
        # Esto previene el error UNIQUE constraint failed.
        seen_normalized_names = {} 

        for rowid, old_nombre, old_descripcion, vector_blob in rows_to_process:
            
            # --- 1. SANITIZACIÓN DE TEXTO ---
            new_nombre = sanitize_text(old_nombre)
            new_descripcion = sanitize_text(old_descripcion)
            
            # Si la codificación se corrigió o si el vector falta, necesitamos actualizar.
            needs_update = (new_nombre != old_nombre) or (new_descripcion != old_descripcion)
            
            # --- 2. VERIFICACIÓN DE DUPLICADOS POR NORMALIZACIÓN ---
            
            # Si el nombre corregido ya lo hemos visto/procesado antes, es un duplicado residual.
            if new_nombre in seen_normalized_names:
                # Es un duplicado. Eliminamos la fila actual (por ROWID)
                print(f"  ❌ DUPLICADO detectado por normalización: '{new_nombre[:50]}...'. Eliminando por ROWID {rowid}.")
                cursor.execute("DELETE FROM productos_scraped WHERE rowid = ?", (rowid,))
                count_deleted += 1
                conn.commit() # Commit inmediato de la eliminación para liberar el conflicto
                continue
            
            # --- 3. VERIFICACIÓN Y REGENERACIÓN DE VECTORES ---
            needs_vector_gen = False
            
            # Comprobamos si el vector es NULL, cadena vacía o corrupto
            if vector_blob is None or len(vector_blob) < 10:
                needs_vector_gen = True
            else:
                try:
                    np.array(json.loads(vector_blob))
                except Exception:
                    needs_vector_gen = True

            # Si necesitamos generar el vector
            if needs_vector_gen:
                print(f"  ➡️ Regenerando vector para: {new_nombre[:50]}...")
                text_to_embed = f"{new_nombre} {new_descripcion}"
                new_vector_list = generate_ollama_embedding(text_to_embed)
                
                if new_vector_list:
                    vector_blob = json.dumps(new_vector_list)
                    needs_update = True
                    count_vectorized += 1
                else:
                    # Si falla la generación, nos saltamos la actualización
                    print(f"  ❌ Falló la generación del vector para {new_nombre}. Saltando.")
                    time.sleep(0.05)
                    continue

            # --- 4. ACTUALIZACIÓN DE LA BASE DE DATOS ---
            if needs_update:
                # Intentamos actualizar. Si la fila se corrigió (new_nombre != old_nombre), 
                # puede haber un error si la versión corregida ya existe (lo cual deberíamos haber detectado antes).
                try:
                    cursor.execute("""
                        UPDATE productos_scraped 
                        SET nombre = ?, descripcion = ?, embedding_vector = ?
                        WHERE rowid = ?
                    """, (new_nombre, new_descripcion, vector_blob, rowid))
                    count_updated += 1
                except sqlite3.IntegrityError:
                    # Capturamos el error residual de UNIQUE constraint, si ocurre inesperadamente.
                    print(f"  ❌ Error de integridad al actualizar '{new_nombre[:50]}...'. El nombre ya existe.")
                    cursor.execute("DELETE FROM productos_scraped WHERE rowid = ?", (rowid,))
                    conn.commit()
                    count_deleted += 1
                    continue
                
            # Marcamos el nombre normalizado como visto/procesado
            seen_normalized_names[new_nombre] = rowid
            
            time.sleep(0.05) # Pequeña pausa
            
        conn.commit() # Commit final de todas las actualizaciones/eliminaciones
        
        print("\n========================================================")
        print("✅ LIMPIEZA COMPLETADA.")
        print(f"  - Registros sanitizados/actualizados: {count_updated}")
        print(f"  - Vectores regenerados: {count_vectorized}")
        print(f"  - Registros duplicados eliminados: {count_deleted}")
        print("========================================================")


    except sqlite3.Error as e:
        print(f"❌ ERROR de SQLite FATAL: {e}")
    except Exception as e:
        print(f"❌ Error general: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    clean_and_revectorize_db()
