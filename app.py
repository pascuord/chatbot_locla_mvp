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

# --- FUNCIÓN DE FILTRO DE INTENCIÓN TÉCNICA (MODIFICADA) ---
def is_technical_query(query):
    """Detecta si la consulta se centra en un nombre INCI o un patrón técnico de la BD."""
    query_upper = query.upper()
    
    # 1. Patrones químicos y conservantes comunes en tu TOP 100
    technical_patterns = [
        r'SODIUM', 'ALCOHOL', 'GLYCOL', 'PHENOXYETHANOL', 'BENZOATE', 'LINALOOL', 
        'LIMONENE', 'TOCOPHEROL', 'BENZYL', 'HYDROXIDE', 'POTASSIUM', 'DIMETHICONE',
        r'COPOLYMER', 'CHLORIDE', 'CAPRYLYL', 'ACETATE', 'STEARATE', 'CETEARYL',
        r'TRIGLYCERIDE', 'TITANIUM', 'SILICA', 'SALICYLATE', 'OXIDES', r'\sACID\s',
        r'MICA', 'CINNAMAL', 'COUMARIN', 'PALMITATE', 'PHOSPHATE', 'SULFATE', 
        r'IONONE', r'ALKYL', 'LECITHIN', 'METHANEDIBENZOYLMETHANE' # Añadidos del top 100
    ]
    
    # 2. Palabras clave de intención (Bloqueo por función o definición)
    definition_keywords = ['COMPONENTE', 'INGREDIENTE']

    # 3. VERIFICACIÓN: Si la pregunta es corta Y contiene un patrón técnico, la bloqueamos.
    # Usamos un umbral de 5 palabras para evitar bloquear preguntas conversacionales largas.
    if len(query.split()) < 7 and any(re.search(p, query_upper) for p in technical_patterns):
        return True
    
    # 4. VERIFICACIÓN DE DEFINICIÓN TÉCNICA
    if any(keyword in query_upper for keyword in definition_keywords):
        return True
    
    return False

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
            print(f"✅ Contexto relevante encontrado (similitud: {top_context_data[0][0]:.2f}).")
            
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
    utilizando una búsqueda flexible (REGEX) basada en la parte inicial del nombre.
    """
    if not product_names:
        return text
    
    # 1. Ordenar los nombres por longitud (descendente)
    product_names.sort(key=len, reverse=True)
    
    for name in product_names:
        
        # Estrategia: Tomar la parte inicial del nombre de la BD (las primeras 5-7 palabras clave)
        words = name.split()
        max_words_to_use = min(5, len(words)) 
        key_name_base = ' '.join(words[:max_words_to_use]).strip()
        
        # 2. Patrón Generoso: Busca la clave esencial Y se expande.
        # Patron: \b(ClaveBase[...caracteres_del_nombre...])
        # Buscamos la clave esencial + cualquier caracter válido (alfanumérico, espacios, puntos, comas, guiones)
        # hasta que encuentra un final de oración o un límite de palabra.
        
        # El Regex busca el inicio (\b) del nombre clave, captura esa clave, y luego captura CUALQUIER COSA
        # que lo siga (incluyendo números, comas, etc.) hasta que encuentra una palabra que no coincide.
        pattern = r'\b(' + re.escape(key_name_base) + r'[\s\w,():!\.\-\'\/]*?)' 
        
        def replace_with_span(match):
            # match.group(1) es la parte del texto del LLM que coincidió (el nombre + la descripción parcial)
            matched_text = match.group(1).strip()
            
            # Verificación de sanity check: evitamos marcar texto demasiado corto
            if len(matched_text.split()) < 3:
                return match.group(0) 
            
            safe_name_for_js = name.replace("'", "\\'") # Nombre completo de la BD
            print(f"🔖 Nombre substitucion: '{safe_name_for_js}'")
            
            # Devolvemos el SPAN clicable COMPLETO
            return f'<span class="product-button" onclick="toggleProductCard(\'{safe_name_for_js}\')">{matched_text}</span>'

        # Sustituimos solo la primera ocurrencia del patrón por el SPAN
        text = re.sub(pattern, replace_with_span, text, flags=re.IGNORECASE, count=1)

    # 4. Limpieza final de espacios extra
    return text

# En app.py
@app.route('/api/product/<product_name>', methods=['GET'])
def get_product_details(product_name):    
    try:
        # 1. Conectando a la base de datos
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
               
        # 2. Preparando la consulta SQL (MÁS ROBUSTA)
        #    Usamos TRIM() para ignorar espacios extra en el nombre guardado en la BD.
        sql_query = "SELECT nombre, descripcion, ingredientes FROM productos_scraped WHERE TRIM(nombre) = ?"
        params = (product_name,)
                
        # 3. Ejecutando la consulta
        cursor.execute(sql_query, params)
        product = cursor.fetchone()
        conn.close()
        
        # 4. Verificando el resultado
        if product:
            return jsonify({
                "nombre": product[0],
                "desc": product[1],
                "ing": product[2],
                "price": "19.99€", # Dato de relleno
                "img": "https://placehold.co/120x120/ffb6c1/000000?text=Cosmetica" # Dato de relleno
            })
        else:
            return jsonify({"error": "Producto no encontrado"}), 404
            
    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {e}"}), 500
    
# NUEVA FUNCIÓN DE AYUDA PARA FORMATEAR LA SALIDA FINAL
def format_response_for_html(text):
    #Formatea el texto final para HTML, creando párrafos y saltos de línea
    #de manera consistente con el frontend.
    
    # Reemplazamos dobles saltos de línea por cierres y aperturas de párrafo.
    # Es crucial hacer esto ANTES de reemplazar los saltos de línea simples.
    formatted_text = text.strip().replace('\n\n', '</p><p>')

    # Reemplazamos los saltos de línea simples restantes por <br>.
    formatted_text = formatted_text.replace('\n', '<br>')

    # Envolvemos todo el contenido en una etiqueta <p> para asegurar la consistencia.
    return f"<p>{formatted_text}</p>"

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
    
    # 1. FILTRO DE INTENCIÓN (NUEVO)
    if is_technical_query(user_query):
        print(f"⛔ Consulta técnica bloqueada: '{user_query}'")
        # Respuesta fija y limpia (sin llamar a Ollama)
        return jsonify({
            "response": "**¡Lo siento mucho!** Como asistente especializado, mi conocimiento se enfoca estrictamente en las **recomendaciones de productos y sus beneficios**, no en el análisis químico detallado de ingredientes (INCI). Por favor, pregúntame cómo un producto en particular puede ayudarte con una necesidad (ej. 'piel sensible', 'anti-arrugas'), y con gusto te asistiré.",
            "latency": 0, 
            "recommended_products": []
        })
    
    start_time = time.time()
    
    retrieved_context, products_to_markup = search_local_db(user_query) 
    #print(f"🔍 Productos para marcar: {products_to_markup}\n")

    if retrieved_context is None:
        return jsonify({"error": "Error crítico al generar embeddings. Verifique el servidor Ollama (nomic-embed-text)."}), 503

    # 2. DEFINICIÓN DEL ROL DEL SISTEMA - ELIMINAMOS LA INSTRUCCIÓN DE MARCAR DEL LLM
    SYSTEM_PROMPT = {
        "role": "system",
        "content": (
            "ERES KIM, un asistente virtual especialista en cosmética. Tu objetivo es ayudar a los usuarios a encontrar los productos adecuados basándote en el contexto proporcionado. "            "TONO: Profesional, amigable, cauto con pieles sensibles. "
            "OBJETIVO PRINCIPAL: Ayudar a los usuarios a encontrar productos cosméticos adecuados según sus necesidades, basándote en el 'CONTEXTO DE PRODUCTOS DE LA TIENDA' proporcionado. "
            "NO PUEDES inventar información sobre productos o ingredientes que no estén en el 'CONTEXTO DE PRODUCTOS DE LA TIENDA'. "
            "SIEMPRE que respondas, DEBES mencionar productos específicos del 'CONTEXTO DE PRODUCTOS DE LA TIENDA' para apoyar tus recomendaciones."
            "**PRECISIÓN RAG (OBLIGATORIO):** SIEMPRE que te pregunten por un producto, un ingrediente específico, o hagas una recomendación, DEBES basar tu respuesta ÚNICAMENTE en la información de los productos proporcionada en el 'CONTEXTO DE PRODUCTOS DE LA TIENDA'."
            "**INSTRUCCIÓN CRÍTICA DE IDENTIDAD:** Tu identidad es de género neutro. NUNCA uses un lenguaje que revele un género (masculino o femenino). Evita palabras como 'encantado/a', 'contento/a', 'experto/a', etc. Mantén todas tus respuestas de forma impersonal y neutra. Si tienes que referirte a tu rol, usa términos como 'asistente' o 'especialista'. "
            "**INSTRUCCIÓN CRÍTICA**: NO USAR HTML/MARKDOWN para nombres"
            "**MARCADO DE PRODUCTO:** Cuando menciones un producto de la tienda, DEBES escribir su nombre tal cual, sin añadir ninguna etiqueta HTML o Markdown. El sistema de post-procesamiento lo marcará automáticamente."
            "Si no encuentras el producto en el contexto, informa al cliente que ese producto no está en stock, pero puedes recomendar uno similar."
            "Responde siempre en español."
        )
    }

    # 3. AUMENTO DEL PROMPT
    system_content_augmented = SYSTEM_PROMPT['content']
    
    if retrieved_context:
        system_content_augmented = f"{system_content_augmented}\n\n{retrieved_context}"

    ollama_messages = []
    
    # 4. Construir el historial para Ollama
    ollama_messages.append({"role": "system", "content": system_content_augmented})

    # 1b. Limpiar el historial de HTML y añadirlo
    HTML_TAG_REGEX = re.compile(r'<[^>]+>') # Regex para eliminar todas las etiquetas HTML

    for entry in conversation_history:
        role = entry['role']
        content = entry['parts'][0]['text']
        
        # Si el mensaje proviene del modelo, eliminamos el HTML (los spans clicables)
        if role == 'model':
            # Eliminamos todas las etiquetas HTML del contenido para evitar la corrupción del prompt
            content = HTML_TAG_REGEX.sub('', content).strip() 
        
        if content:
             ollama_messages.append({"role": role, "content": content})

    try:
        # 5. Llama a la API /api/chat local de Ollama en modo streaming
        payload = {"model": "llama3:8b", "messages": ollama_messages, "stream": True, "options": {"temperature": 0.3}}
        
        ollama_response = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=120)
        ollama_response.raise_for_status() 

        # Función generadora que manejará el stream y el post-procesamiento
        def generate():
            response_text_crudo_parts = []
            
            # Itera sobre los chunks de texto del LLM y los envía al frontend
            for chunk in ollama_response.iter_lines():
                if chunk:
                    try:
                        line = chunk.decode('utf-8')
                        data = json.loads(line)
                        content_piece = data['message']['content']
                        
                        response_text_crudo_parts.append(content_piece)
                        yield content_piece # Envía el trozo de texto crudo
                    except json.JSONDecodeError:
                        continue # Ignora líneas inválidas

            # --- Post-procesamiento después de que el stream del LLM haya terminado ---
            
            # 1. Reconstruir la respuesta completa
            response_text_crudo = "".join(response_text_crudo_parts).strip()
            print(f"💬 Respuesta completa del LLM (antes de marcar): {response_text_crudo}\n")
            
            # 2. Aplicar el marcado interactivo de productos
            response_text_with_spans = markup_product_names(response_text_crudo, products_to_markup)

            # 3. --- CAMBIO CLAVE ---
            #    Aplicar el formato de párrafo final a la respuesta que ya tiene los spans.
            response_text_final = format_response_for_html(response_text_with_spans)

            # 4. Calcular la latencia final
            end_time = time.time()
            latency_ms = round((end_time - start_time) * 1000)

            # 5. Crear el payload final con la respuesta ya formateada
            final_data = {
                "final_response": response_text_final,
                "latency": latency_ms,
                "recommended_products": products_to_markup 
            }

            # 6. Enviar el payload final
            yield f"__END_OF_STREAM__{json.dumps(final_data)}"

        # Devuelve la respuesta como un stream de texto plano
        return Response(stream_with_context(generate()), mimetype='text/plain')
    
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Error: El servidor Ollama (LLM Local) no está ejecutándose."}), 503
    
    except requests.exceptions.HTTPError as e:
        error_message = f"Error HTTP de Ollama: {e.response.status_code}. Asegúrate de que el modelo 'llama3:8b' esté instalado y que el servidor Ollama esté activo."
        return jsonify({"error": error_message}), 500
        
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error interno del chatbot: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
