import os
import requests 
import sqlite3 
import json
import numpy as np 
import re 
import time 

# --- CONFIGURACIÓN GLOBAL ---
OLLAMA_API_URL = "http://localhost:11434/api/chat" 
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
DB_NAME = 'db_cosmetica.db'
EMBEDDING_MODEL = "nomic-embed-text" 

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# =======================================================
# FUNCIONES DE UTILERÍA (Similitud)
# =======================================================

def generate_ollama_embedding(text):
    """Genera un vector de embedding para el texto usando Ollama."""
    try:
        payload = {"model": EMBEDDING_MODEL,"prompt": text,}
        response = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['embedding']
    except Exception as e:
        print(f"❌ ERROR al generar embedding: {e}")
        return None

def cosine_similarity(v1, v2):
    """Calcula la similitud de coseno entre dos vectores numpy."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))

# =======================================================
# FUNCIÓN DE BÚSQUEDA DE CONTEXTO (RAG VECTORIAL)
# =======================================================

def search_local_db(query):
    # [La lógica de búsqueda vectorial permanece igual, solo devuelve la lista de productos]
    query_vector_list = generate_ollama_embedding(query)
    if query_vector_list is None: return None, [] 

    query_vector = np.array(query_vector_list)

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT nombre, descripcion, ingredientes, embedding_vector FROM productos_scraped")
        products = cursor.fetchall()
        conn.close()

        if not products: return None, []

        scored_products = []
        for nombre, descripcion, ingredientes, vector_blob in products:
            try:
                product_vector = np.array(json.loads(vector_blob))
                similarity = cosine_similarity(query_vector, product_vector)
                
                product_data = {'nombre': nombre,'descripcion': descripcion.strip(),'ingredientes': ingredientes.strip()}
                scored_products.append((similarity, product_data))
            except Exception: continue

        scored_products.sort(key=lambda item: item[0], reverse=True)
        top_context_data = scored_products[:5] 
        
        context_for_llm = ""
        products_to_markup = [] 
        
        MIN_SIMILARITY_THRESHOLD = 0.4 
        
        if top_context_data and top_context_data[0][0] > MIN_SIMILARITY_THRESHOLD: 
            context_for_llm = "\n--- CONTEXTO DE PRODUCTOS DE LA TIENDA ---\n"
            for score, data in top_context_data:
                if score > MIN_SIMILARITY_THRESHOLD: 
                    context_for_llm += (
                        f"PRODUCTO: {data['nombre']}\n"
                        f"DESCRIPCION: {data['descripcion'][:200]}...\n"
                        f"INGREDIENTES: {data['ingredientes'][:100]}...\n\n"
                    )
                    products_to_markup.append(data['nombre'])
            context_for_llm += "--- FIN CONTEXTO ---\n"
        
        return context_for_llm, products_to_markup
    
    except Exception as e:
        print(f"❌ ERROR en search_local_db: {e}")
        return None, []

# =======================================================
# FUNCIÓN DE MARCADO DE RESPUESTA (POST-PROCESAMIENTO)
# =======================================================

def markup_product_names(text, product_names):
    """
    ENCAPSULA los nombres de producto de la lista product_names en el texto del LLM
    utilizando una búsqueda flexible (REGEX).
    """
    if not product_names:
        return text
    
    # Ordenar los nombres por longitud (descendente)
    product_names.sort(key=len, reverse=True)
    
    for name in product_names:
        
        # 1. Crear un patrón flexible basado en las palabras clave iniciales del nombre de la BD
        # Esto soluciona el problema de que el LLM omite el final del nombre ("(75 ml)").
        
        # Tomamos la parte inicial del nombre de la BD (las primeras 5-7 palabras clave)
        key_name = ' '.join(name.split()[:3]) 

        # 2. El patrón busca la clave (ej. 'Aceite corporal nutritivo Rose de Dr. Hauschka')
        # Y permite que le sigan las palabras adicionales que el LLM pudo haber omitido en la respuesta.
        # Buscamos el patrón CON LIMITES DE PALABRA al inicio, pero NO al final del nombre clave.
        pattern = r'\b' + re.escape(key_name) + r'[^a-zA-Z0-9\s,\'-]*?' 
        
        def replace_with_span(match):
            # match.group(0) es el texto que realmente coincidió en la respuesta del LLM (ej. "Aceite corporal nutritivo Rose de Dr. Hauschka")
            matched_text = match.group(0).strip()
            safe_name_for_js = name.replace("'", "\\'") # Usamos el nombre COMPLETO de la BD para la metadata del JS
            
            # Devolvemos el SPAN clicable COMPLETO
            return f'<span class="product-button" onclick="toggleProductCard(\'{safe_name_for_js}\')">{matched_text}</span>'

        # Sustituimos todas las ocurrencias del nombre de este producto
        text = re.sub(pattern, replace_with_span, text, flags=re.IGNORECASE)

    # 4. Limpieza final de espacios extra
    return ' '.join(text.split())

# =======================================================
# FUNCIÓN PRINCIPAL DE CHAT (API /api/chat)
# =======================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    conversation_history = data.get('history', []) 

    if not conversation_history:
         return jsonify({"error": "No se recibió historial de conversación."}), 400

    user_query = conversation_history[-1]['parts'][0]['text']
    
    start_time = time.time()
    
    retrieved_context, products_to_markup = search_local_db(user_query) 

    if retrieved_context is None:
        return jsonify({"error": "Error crítico al generar embeddings. Verifique el servidor Ollama (nomic-embed-text)."}), 503

    # 2. DEFINICIÓN DEL ROL DEL SISTEMA - ELIMINAMOS LA INSTRUCCIÓN DE MARCAR DEL LLM
    SYSTEM_PROMPT = {
        "role": "system",
        "content": (
            "ERES UN ASESOR DE VENTA DE COSMÉTICA. Tu nombre es 'La Experta en Piel'. "
            "TONO: Profesional, amigable, cauto con pieles sensibles. "
            #"**ESTRUCTURA DE RESPUESTA OBLIGATORIA:** Tu respuesta DEBE seguir la siguiente estructura de formato, usando negritas (**):"
            #"1. **Comienza siempre con:** 'Hola! Como La Experta en Piel, estoy aquí para ayudarte...' "
            #"2. **Utiliza títulos con dos puntos:** Emplea títulos como **'Recomendación inicial:'** y **'Paso a paso:'** para organizar tu respuesta."
            #"3. **Formato de Lista:** Cuando enumeres puntos, pasos o métodos, DEBES usar el formato de lista Markdown: **1. Título de paso:** Contenido. (Añade un salto de línea antes de cada punto para la legibilidad)."
            "**PRECISIÓN RAG (OBLIGATORIO):** SIEMPRE que te pregunten por un producto, un ingrediente específico, o hagas una recomendación, DEBES basar tu respuesta ÚNICAMENTE en la información de los productos proporcionada en el 'CONTEXTO DE PRODUCTOS DE LA TIENDA'."
            # INSTRUCCIÓN CRÍTICA: NO USAR HTML/MARKDOWN para nombres
            "**MARCADO DE PRODUCTO:** Cuando menciones un producto de la tienda, DEBES escribir su nombre tal cual, sin añadir ninguna etiqueta HTML o Markdown. El sistema de post-procesamiento lo marcará automáticamente."
            "Si no encuentras el producto en el contexto, informa al cliente que ese producto no está en stock, pero puedes recomendar uno similar."
            "Responde siempre en español."
        )
    }

    # 3. AUMENTO DEL PROMPT
    system_content_augmented = SYSTEM_PROMPT['content']
    if retrieved_context:
        system_content_augmented = f"{system_content_augmented}\n\n{retrieved_context}"

    # 4. Construir el historial para Ollama
    ollama_messages = [{"role": "system", "content": system_content_augmented}]
    
    for entry in conversation_history:
        role = entry['role']
        content = entry['parts'][0]['text']
        if role == 'model':
            role = 'assistant'
        ollama_messages.append({"role": role, "content": content})

    try:
        # 5. Llama a la API /api/chat local de Ollama
        payload = {"model": "llama3:8b", "messages": ollama_messages, "stream": False, "options": {"temperature": 0.3}}
        
        ollama_response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        ollama_response.raise_for_status() 

        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000)
        
        # 6. Procesar y Marcar la respuesta
        response_data = ollama_response.json()
        response_text_crudo = response_data['message']['content'].strip()
        
        # APLICAR MARCADO INTERACTIVO: Usando la lista de nombres completos de la BD
        response_text = markup_product_names(response_text_crudo, products_to_markup) 
        
        # DEVOLVER LA LATENCIA Y LA LISTA DE PRODUCTOS (Para la barra de sugerencias)
        return jsonify({
            "response": response_text,
            "latency": latency_ms,
            "recommended_products": products_to_markup 
        })
    
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Error: El servidor Ollama (LLM Local) no está ejecutándose."}), 503
    
    except requests.exceptions.HTTPError as e:
        error_message = f"Error HTTP de Ollama: {e.response.status_code}. Asegúrate de que el modelo 'llama3:8b' esté instalado y que el servidor Ollama esté activo."
        return jsonify({"error": error_message}), 500
        
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error interno del chatbot: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
