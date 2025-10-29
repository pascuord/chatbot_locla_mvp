import os
import requests 
import sqlite3 
import json
import numpy as np 
import re 
import time 

# --- CONFIGURACIÓN GLOBAL ---
OLLAMA_API_URL_BASE = "http://localhost:11434" # Base para Ollama API
OLLAMA_API_URL = f"{OLLAMA_API_URL_BASE}/api/chat" 
OLLAMA_EMBED_URL = f"{OLLAMA_API_URL_BASE}/api/embeddings"
DB_NAME = 'db_cosmetica.db'
EMBEDDING_MODEL = "nomic-embed-text" 
REWRITE_MODEL = "llama3:8b" # Modelo para reescribir (puede ser uno más pequeño/rápido)
MAIN_MODEL = "mixtral:8x7b-instruct-v0.1-q2_K" # Modelo principal para responder

from flask import Flask, request, jsonify, Response, stream_with_context
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
# NUEVA FUNCIÓN: REESCRITURA DE CONSULTA CON LLM
# =======================================================

def rewrite_query_with_history(user_query, conversation_history, model=REWRITE_MODEL):
    """
    Usa un LLM para reescribir la consulta del usuario incorporando
    contexto del historial reciente.
    """
    print(f"🔄 Reescribiendo consulta: '{user_query}'")
    
    # Extraer los últimos turnos relevantes (ej. último par user/model)
    history_context = ""
    if len(conversation_history) >= 2:
        last_user_turn = conversation_history[-2] # Penúltimo mensaje (usuario)
        last_model_turn = conversation_history[-1] # Último mensaje (modelo)
        
        last_user_text = last_user_turn.get('parts', [{}])[0].get('text', '')
        last_model_text = last_model_turn.get('parts', [{}])[0].get('text', '')
        
        last_model_text_clean = re.sub(r'<[^>]+>', '', last_model_text).strip()

        history_context = f"""
            Contexto de la Conversación Anterior:
            Usuario preguntó: "{last_user_text}"
            Tu respuesta fue (resumida): "{last_model_text_clean[:500]}..." """ # Resumen para no exceder el prompt
        
        # Prompt específico para la tarea de reescritura
        rewrite_prompt = f"""Eres un asistente experto en reescribir consultas de usuario para que sean autocontenidas, usando el contexto de la conversación.
        {history_context}

        Pregunta Actual del Usuario: "{user_query}"

        Tu Tarea: Reescribe la "Pregunta Actual del Usuario" para que sea una consulta clara y completa que pueda entenderse sin necesidad de leer el contexto anterior.
        - Si la pregunta actual ya es completa, devuélvela tal cual.
        - Si hace referencia a elementos anteriores (ej. "esas dos", "la primera opción", "cuál es mejor para X"), incorpóralos explícitamente en la nueva consulta (mencionando los productos si aparecen en 'Tu respuesta fue').
        - Sé conciso y directo.
        - Responde ÚNICAMENTE con la consulta reescrita, sin explicaciones ni saludos.

        Consulta Reescrita:"""

    try:
        ollama_generate_url = f"{OLLAMA_API_URL_BASE}/api/generate" 
        payload = {
            "model": model, 
            "prompt": rewrite_prompt,
            "stream": False,
            "options": {"temperature": 0.0} 
        }
        
        response = requests.post(ollama_generate_url, json=payload, timeout=45) # Timeout un poco más largo
        response.raise_for_status()
        
        rewritten_query = response.json().get('response', '').strip()
        rewritten_query = rewritten_query.strip('"\'') # Limpiar comillas
        
        if rewritten_query and rewritten_query.lower() != user_query.lower(): # Comprobar si realmente cambió
            print(f"✅ Consulta reescrita: '{rewritten_query}'")
            return rewritten_query
        else:
            print(f"ℹ️ Consulta no reescrita. Usando original.")
            return user_query
            
    except Exception as e:
        print(f"❌ Error al reescribir consulta: {e}. Usando consulta original.")
        return user_query

# --- FUNCIÓN DE FILTRO DE INTENCIÓN TÉCNICA (sin cambios) ---
def is_technical_query(query):
    # ... (tu código actual) ...
    query_upper = query.upper()
    technical_patterns = [
        r'SODIUM', 'ALCOHOL', 'GLYCOL', 'PHENOXYETHANOL', 'BENZOATE', 'LINALOOL', 
        'LIMONENE', 'TOCOPHEROL', 'BENZYL', 'HYDROXIDE', 'POTASSIUM', 'DIMETHICONE',
        r'COPOLYMER', 'CHLORIDE', 'CAPRYLYL', 'ACETATE', 'STEARATE', 'CETEARYL',
        r'TRIGLYCERIDE', 'TITANIUM', 'SILICA', 'SALICYLATE', 'OXIDES', r'\sACID\s',
        r'MICA', 'CINNAMAL', 'COUMARIN', 'PALMITATE', 'PHOSPHATE', 'SULFATE', 
        r'IONONE', r'ALKYL', 'LECITHIN', 'METHANEDIBENZOYLMETHANE'
    ]
    definition_keywords = ['COMPONENTE', 'INGREDIENTE']
    definition_seeking_patterns = [
        r'\bQUÉ ES\b', r'\bPARA QUÉ SIRVE\b', r'\bCUÁL ES LA FUNCIÓN\b', r'\bDEFINICIÓN DE\b'
    ]
    if len(query.split()) < 7 and any(re.search(p, query_upper) for p in technical_patterns):
        return True
    contains_keyword = any(keyword in query_upper for keyword in definition_keywords)
    is_seeking_definition = any(re.search(p, query_upper) for p in definition_seeking_patterns)
    if contains_keyword and is_seeking_definition:
        return True
    return False

# =======================================================
# FUNCIÓN DE BÚSQUEDA DE CONTEXTO (RAG VECTORIAL) (sin cambios)
# =======================================================
def search_local_db(query):
    # ... (tu código actual para generar embedding, buscar en DB, calcular similitud, formatear contexto) ...
    # Asegúrate de que esta función selecciona y formatea el PRECIO en el context_for_llm
    query_vector_list = generate_ollama_embedding(query)
    if query_vector_list is None: return None, [] 
    query_vector = np.array(query_vector_list)
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT nombre, descripcion, ingredientes, precio, embedding_vector 
            FROM productos_scraped 
            WHERE ingredientes IS NOT NULL OR descripcion IS NOT NULL 
        """) # Asegurarse de tener al menos ingredientes o descripción
        products = cursor.fetchall()
        conn.close()
        if not products: return None, []

        scored_products = []
        for nombre, descripcion, ingredientes, precio, vector_blob in products:
            try:
                # Evitar error si el vector está mal guardado
                if not vector_blob: continue
                product_vector = np.array(json.loads(vector_blob))
                # Asegurar que ambos vectores tienen la misma dimensión
                if product_vector.shape != query_vector.shape: continue 
                    
                similarity = cosine_similarity(query_vector, product_vector)
                
                product_data = {
                    'nombre': nombre,
                    'descripcion': descripcion.strip() if descripcion else "", # Manejar None
                    'ingredientes': ingredientes.strip() if ingredientes else "", # Manejar None
                    'precio': precio,
                    'similarity': similarity # Guardamos similitud para posible lógica híbrida/filtrado
                }
                scored_products.append(product_data)
            except Exception as e: 
                # print(f"Debug search_local_db loop error: {e} for product {nombre}") # Descomentar para depurar
                continue

        scored_products.sort(key=lambda item: item['similarity'], reverse=True)
        
        # Lógica de combinación (si usas híbrida) o simplemente tomar top N
        top_context_data_list = scored_products[:5] # Tomar los 5 mejores vectoriales
        
        context_for_llm = ""
        products_to_markup = [] 
        MIN_SIMILARITY_THRESHOLD = 0.4 
        
        if top_context_data_list:
             # Usamos la similitud del mejor encontrado
            best_similarity = top_context_data_list[0]['similarity']
            if best_similarity > MIN_SIMILARITY_THRESHOLD:
                print(f"✅ Contexto relevante encontrado (similitud max: {best_similarity:.2f}).")
                context_for_llm = "\n--- CONTEXTO DE PRODUCTOS DE LA TIENDA ---\n"
                for data in top_context_data_list:
                    # Aplicar umbral a cada producto individualmente
                    if data['similarity'] > MIN_SIMILARITY_THRESHOLD: 
                        precio_str = f"{data['precio']:.2f}€" if data.get('precio') is not None else "Precio no disponible"
                        context_for_llm += (
                            f"PRODUCTO: {data['nombre']}\n"
                            f"PRECIO: {precio_str}\n"
                            f"DESCRIPCION: {data.get('descripcion', '')[:200]}...\n"
                            f"INGREDIENTES: {data.get('ingredientes', '')[:100]}...\n\n"
                        )
                        products_to_markup.append(data['nombre'])
                context_for_llm += "--- FIN CONTEXTO ---\n"
            else:
                 print(f"ℹ️ Contexto encontrado pero similitud baja ({best_similarity:.2f}). No se usará.")
        else:
            print("ℹ️ No se encontró contexto relevante.")
            
        return context_for_llm, products_to_markup
    
    except Exception as e:
        print(f"❌ ERROR en search_local_db: {e}")
        return None, []


# =======================================================
# FUNCIÓN DE MARCADO DE RESPUESTA (POST-PROCESAMIENTO) (sin cambios)
# =======================================================
def markup_product_names(text, product_names):
    # ... (tu código actual) ...
    if not product_names: return text
    product_names.sort(key=len, reverse=True)
    for name in product_names:
        words = name.split()
        max_words_to_use = min(5, len(words)) 
        key_name_base = ' '.join(words[:max_words_to_use]).strip()
        pattern = r'\b(' + re.escape(key_name_base) + r'[\s\w,():!\.\-\'\/]*?)' 
        def replace_with_span(match):
            matched_text = match.group(1).strip()
            if len(matched_text.split()) < 3: return match.group(0) 
            safe_name_for_js = name.replace("'", "\\'")
            return f'<span class="product-button" onclick="toggleProductCard(\'{safe_name_for_js}\')">{matched_text}</span>'
        text = re.sub(pattern, replace_with_span, text, flags=re.IGNORECASE, count=1)
    return text # Devolver texto, no ' '.join(text.split()) si causa problemas

# =======================================================
# ENDPOINT DETALLES PRODUCTO (sin cambios)
# =======================================================
@app.route('/api/product/<product_name>', methods=['GET'])
def get_product_details(product_name):    
    # ... (tu código actual) ...
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # --- CAMBIO: Seleccionar precio real ---
        sql_query = "SELECT nombre, descripcion, ingredientes, precio FROM productos_scraped WHERE nombre = ?" # Usar nombre limpio si lo guardaste
        params = (product_name,)
        cursor.execute(sql_query, params)
        product = cursor.fetchone()
        conn.close()
        if product:
            # --- CAMBIO: Devolver precio real ---
             precio_str = f"{product[3]:.2f}€" if product[3] is not None else "No disponible"
             # Asumir una imagen placeholder si no la tienes en la BD
             img_placeholder = "https://placehold.co/120x120/e2f0d9/333?text=Producto"
             return jsonify({
                "nombre": product[0],
                "desc": product[1] if product[1] else "Descripción no disponible",
                "ing": product[2] if product[2] else "Ingredientes no disponibles",
                "price": precio_str, 
                "img": img_placeholder 
            })
        else:
            return jsonify({"error": "Producto no encontrado"}), 404
    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500

# =======================================================
# FUNCIÓN FORMATEO HTML (sin cambios)
# =======================================================
def format_response_for_html(text):
    # ... (tu código actual) ...
    formatted_text = text.strip().replace('\n\n', '</p><p>')
    formatted_text = formatted_text.replace('\n', '<br>')
    return f"<p>{formatted_text}</p>"


# =======================================================
# FUNCIÓN PRINCIPAL DE CHAT (API /api/chat) - MODIFICADA
# =======================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    full_conversation_history = data.get('history', []) 

    if not full_conversation_history:
         return jsonify({"error": "No se recibió historial de conversación."}), 400

    original_user_query = full_conversation_history[-1]['parts'][0]['text']
    
    # --- PASO 0: REESCRITURA DE CONSULTA ---
    if len(full_conversation_history) > 1:
        history_for_rewrite = full_conversation_history[:-1] 
        rewritten_query = rewrite_query_with_history(original_user_query, history_for_rewrite)
    else:
        rewritten_query = original_user_query
        print("ℹ️ Primer mensaje, no se reescribe.")
    
    # 1. FILTRO DE INTENCIÓN (usar original)
    if is_technical_query(original_user_query):
        print(f"⛔ Consulta técnica bloqueada: '{original_user_query}'")
        # ... (respuesta fija) ...
        return jsonify({ ... })
    
    start_time = time.time()
    
    # 2. BÚSQUEDA RAG (usar reescrita)
    retrieved_context, products_to_markup = search_local_db(rewritten_query) 
    
    # Manejo de error si la búsqueda falla completamente
    if retrieved_context is None and products_to_markup is None: # Modificado para checkear ambos
         # Podrías devolver un error 500 o intentar responder sin RAG
         print("❌ Error crítico en RAG o embedding. Intentando responder sin contexto.")
         retrieved_context = "" # Forzar contexto vacío para continuar
         products_to_markup = []


    # 3. DEFINICIÓN DEL ROL DEL SISTEMA (sin cambios, usa tu prompt reforzado)
    SYSTEM_PROMPT = {
        "role": "system",
        "content": (
            "ERES KIM, un asistente virtual especialista en cosmética. Tu objetivo es ayudar a los usuarios a encontrar los productos adecuados basándote en el contexto proporcionado.\n"
            "TONO: Profesional, amigable y servicial. Cauto con pieles sensibles.\n\n"
            "**OBJETIVO PRINCIPAL:** Ayudar a los usuarios a encontrar productos cosméticos adecuados según sus necesidades, basándote **estrictamente** en el 'CONTEXTO DE PRODUCTOS DE LA TIENDA' proporcionado.\n"
            "**RESTRICCIÓN CLAVE:** NO PUEDES inventar información sobre productos, ingredientes o precios que no estén explícitamente en el 'CONTEXTO DE PRODUCTOS DE LA TIENDA'. Tu conocimiento se limita a ese contexto.\n\n"
            "**PRECISIÓN RAG (OBLIGATORIO):**\n"
            "1. Basa **SIEMPRE** tus respuestas (recomendaciones, descripciones, ingredientes) ÚNICAMENTE en la información del 'CONTEXTO DE PRODUCTOS DE LA TIENDA' **proporcionado para esta pregunta específica**.\n"
            "2. **NO menciones ni hagas referencia a otros productos**, incluso si tienen nombres similares, a menos que estén explícitamente listados en el contexto actual.\n" 
            "3. Menciona productos específicos del contexto para apoyar tus recomendaciones.\n"
            "4. Si no encuentras información sobre un producto específico en el contexto, informa al cliente que no tienes detalles sobre él, pero puedes recomendar uno similar **basado en otros productos que sí encuentres en el contexto**.\n\n"
             "**MANEJO DE PREGUNTAS DE SEGUIMIENTO:**\n" # Añadido si no lo tenías
            "1. Presta mucha atención a pronombres o frases que hagan referencia a productos mencionados en turnos anteriores de la conversación (ej. 'esa crema', 'la segunda opción', 'entre esos dos').\n"
            "2. Si la pregunta del usuario es una comparación o una solicitud de más detalles sobre productos específicos ya mencionados, tu respuesta DEBE centrarse en esos productos si están presentes en el 'CONTEXTO DE PRODUCTOS DE LA TIENDA' actual.\n"
            "3. No introduzcas productos completamente nuevos si la pregunta claramente se refiere a los ya discutidos, a menos que el contexto RAG recuperado SÓLO contenga esos nuevos productos (en cuyo caso, indica que no tienes más detalles de los anteriores).\n\n"
            "**MANEJO DE IDIOMAS:** El contexto puede contener descripciones en inglés o español. Basa tu respuesta en la información encontrada, independientemente del idioma original.\n\n" 
            "**USO DEL PRECIO:**\n"
            "1. Si el contexto incluye el precio de un producto, **DEBES** usar ese precio exacto si la consulta lo requiere (ej. comparar precios, buscar opciones económicas).\n"
            "2. **NO asumas ni inventes precios**. Si el precio no está en el contexto, indica explícitamente que no tienes esa información para ese producto.\n"
            "3. Al comparar ('más barato', 'más caro'), basa tu comparación **estrictamente** en los precios listados en el contexto actual.\n"
            "4. Puedes mencionar el precio si es relevante, pero no es obligatorio mencionarlo siempre.\n\n"
            "**INSTRUCCIÓN CRÍTICA DE IDENTIDAD:** Tu identidad es de género neutro. NUNCA uses un lenguaje que revele un género (masculino o femenino). Evita palabras como 'encantado/a', 'contento/a', 'experto/a', etc. Mantén todas tus respuestas de forma impersonal y neutra. Usa términos como 'asistente' o 'especialista'.\n\n"
            "**¡¡IMPORTANTE!! TU RESPUESTA FINAL DEBE ESTAR SIEMPRE Y COMPLETAMENTE EN ESPAÑOL.** NO uses inglés en tu respuesta bajo ninguna circunstancia, incluso si el contexto recuperado está en inglés.\n\n"
            "**FORMATO DE SALIDA (OBLIGATORIO):**\n"
            "1. Responde SIEMPRE en español.\n"
            "2. Cuando menciones un producto de la tienda, escribe su nombre **tal cual aparece en el contexto**, sin añadir ninguna etiqueta HTML o Markdown. El sistema de post-procesamiento lo marcará."
        )
    }

    # 4. AUMENTO DEL PROMPT (sin cambios)
    system_content_augmented = SYSTEM_PROMPT['content']
    if retrieved_context: # Solo añadir si el contexto no está vacío
        system_content_augmented = f"{system_content_augmented}\n\n{retrieved_context}"

    ollama_messages = []
    
    # 5. Construir el historial para Ollama (usar historial original)
    ollama_messages.append({"role": "system", "content": system_content_augmented})
    HTML_TAG_REGEX = re.compile(r'<[^>]+>')
    for entry in full_conversation_history: 
        role = entry['role']
        content = entry['parts'][0]['text'] 
        if role == 'model':
            content = HTML_TAG_REGEX.sub('', content).strip() 
        if content:
             ollama_messages.append({"role": role, "content": content})

    # 6. LLAMADA AL LLM PRINCIPAL (Streaming)
    try:
        # Imprimir contexto final justo antes de la llamada (opcional para depurar)
        # print("--- CONTEXTO FINAL ENVIADO A OLLAMA ---")
        # print(system_content_augmented)
        # print("--- HISTORIAL ENVIADO A OLLAMA ---")
        # print(ollama_messages)
        # print("---------------------------------")
        
        payload = {"model": MAIN_MODEL, "messages": ollama_messages, "stream": True, "options": {"temperature": 0.0}}
        
        ollama_response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=180) # Aumentar timeout para modelos grandes
        ollama_response.raise_for_status() 

        # --- AJUSTE: Pasar args a generate ---
        return Response(stream_with_context(generate(start_time, products_to_markup, ollama_response)), mimetype='text/plain')
    
    # ... (tus bloques except sin cambios) ...
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Error: El servidor Ollama (LLM Local) no está ejecutándose."}), 503
    except requests.exceptions.HTTPError as e:
        error_message = f"Error HTTP de Ollama: {e.response.status_code}. Asegúrate de que el modelo '{MAIN_MODEL}' esté instalado y que el servidor Ollama esté activo."
        return jsonify({"error": error_message}), 500
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error interno del chatbot: {e}"}), 500

# =======================================================
# FUNCIÓN GENERADORA PARA STREAMING (MODIFICADA)
# =======================================================
# --- AJUSTE: Aceptar ollama_response como argumento ---
def generate(start_time, products_to_markup, ollama_response):
    response_text_crudo_parts = []
    try: # Añadir try/except dentro del generador es buena práctica
        for chunk in ollama_response.iter_lines():
            if chunk:
                try:
                    line = chunk.decode('utf-8')
                    data = json.loads(line)
                    # Verificar estructura del chunk de streaming
                    message_data = data.get('message', {})
                    content_piece = message_data.get('content')
                    if content_piece is not None: # Solo procesar si hay contenido
                         response_text_crudo_parts.append(content_piece)
                         yield content_piece 
                except json.JSONDecodeError:
                    # Ignorar líneas inválidas (a veces Ollama envía líneas vacías al final)
                    # print(f"Warning: JSONDecodeError en chunk: {chunk}") # Descomentar para depurar
                    continue 
                except Exception as e_inner:
                     print(f"Error procesando chunk: {e_inner}") # Loguear error inesperado
                     continue

        # --- Post-procesamiento ---
        response_text_crudo = "".join(response_text_crudo_parts).strip()
        print(f"💬 Respuesta completa del LLM (antes de marcar): {response_text_crudo}\n")
        
        response_text_with_spans = markup_product_names(response_text_crudo, products_to_markup)
        response_text_final = format_response_for_html(response_text_with_spans)

        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000)
        
        final_data = {
            "final_response": response_text_final,
            "latency": latency_ms,
            "recommended_products": products_to_markup 
        }
        
        yield f"__END_OF_STREAM__{json.dumps(final_data)}"
    except Exception as e_outer:
         print(f"❌ Error CRÍTICO dentro del generador de streaming: {e_outer}")
         # Intentar enviar un mensaje de error al frontend si es posible
         error_data = {"error": f"Error interno durante el streaming: {e_outer}"}
         yield f"__END_OF_STREAM__{json.dumps(error_data)}"
    finally:
        # Asegurarse de cerrar la respuesta si es necesario (requests suele hacerlo)
        if ollama_response:
             ollama_response.close()


if __name__ == '__main__':
    app.run(debug=True, port=5000)