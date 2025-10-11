🧴 Asistente Experto en Cosmética (RAG Local y Coste Cero)
Este proyecto implementa un chatbot conversacional avanzado capaz de actuar como un experto en cosmética. Utiliza la arquitectura RAG (Retrieval-Augmented Generation) para inyectar información específica de productos (scraped de Lookfantastic) en un Modelo de Lenguaje Grande (LLM) de código abierto, Llama 3 (8B).
Ventaja Principal: El 100% de la lógica de IA y la ejecución se realizan en hardware local (GPU NVIDIA), lo que garantiza un Coste Cero de operación continua (sin límites de cuota de API).
🏗️ Arquitectura del Proyecto
El sistema se basa en tres componentes principales que se comunican localmente:
Frontend (index.html): Interfaz de usuario simple para texto y voz (TTS/STT, utilizando las APIs nativas del navegador). Mantiene el historial de la conversación.
Backend (app.py - Flask/Python): Actúa como orquestador. Recibe la pregunta, consulta la base de datos RAG, construye el prompt aumentado y se comunica con Ollama.
Base de Conocimientos (RAG): Archivo db_cosmetica.db (SQLite) con datos de productos extraídos mediante web scraping.
Motor LLM (Ollama/Llama 3): Servidor local que ejecuta Llama 3, utilizando la potencia de la GPU (RTX 3070).
⚙️ Requisitos y Dependencias
Necesitas el siguiente software instalado en tu entorno Ubuntu:
Software Base
Python 3.9+
Servidor Ollama (Instalado y configurado para usar la GPU NVIDIA)
Modelo Llama 3 (8b): Debe estar descargado en Ollama.
ollama run llama3:8b


SQLite3 CLI (Para depuración de la base de datos).
Dependencias de Python
Instala las librerías necesarias en tu entorno virtual (venv):
pip install flask requests beautifulsoup4 flask-cors


▶️ Comandos Esenciales de Inicio Rápido
Para que el asistente esté disponible, debes ejecutar tres comandos en dos terminales separadas, en el orden indicado:
Terminal 1: Iniciar el Motor LLM (Llama 3)
Este comando inicia el servidor de la API que ejecuta Llama 3.
sudo systemctl start ollama 


(Alternativa: Si el servicio está deshabilitado, usa ollama serve)
Terminal 2: Iniciar el Backend (Flask)
Este comando inicia el servidor de orquestación (API del chatbot).
# Asegúrate de que el entorno virtual esté activo (source venv/bin/activate)
python app.py 


Navegador: Iniciar el Frontend
Abre la interfaz para comenzar a conversar.
Abre el archivo index.html directamente en tu navegador.
🚀 Guía de Configuración y Ejecución
Sigue estos pasos para poner en marcha el asistente experto.
Paso 1: Inicializar el Motor LLM (Ollama)
Abre una terminal y mantén el servidor Ollama activo. Este paso es fundamental para el coste cero:
sudo systemctl start ollama # Inicia como servicio (Método más estable)
# Opcional: Si lo haces manualmente para ver logs
# ollama serve


Paso 2: Crear la Base de Conocimientos (RAG)
Primero, creamos el archivo de base de datos vacío y luego lo llenamos con los datos extraídos.
Crear el Archivo db_cosmetica.db:
Nota: Se asume que has ejecutado el script init_db_scraper.py anteriormente.
Si el archivo no existe, la función setup_db() en el scraper lo creará.
Llenar la Base de Datos con Productos:
Asegúrate de que el archivo product_urls.txt (con las URLs de Lookfantastic) exista en el directorio.
Ejecuta el scraper (scraper_cosmetica.py):
python scraper_cosmetica.py


Esto llenará la tabla productos_scraped con los nombres, descripciones e ingredientes, que actuarán como el contexto para Llama 3.
Paso 3: Iniciar el Backend (Flask)
Abre una segunda terminal (asegúrate de que el entorno venv esté activo) e inicia el servidor de orquestación:
python app.py


Paso 4: Iniciar el Frontend y Conversar
Abre el archivo index.html directamente en tu navegador (Chrome o Edge recomendado para la funcionalidad de voz).
Prueba de RAG (Contexto): Pregunta sobre productos o ingredientes específicos que sabes que están en tu base de datos local (ej. "¿Qué dice el sérum hidratante sobre la piel seca?").
🔍 Depuración
Síntoma
Problema (Causa)
Solución
Error 500 en el navegador
Error interno de Flask (app.py).
Revisa la terminal donde se ejecuta app.py para ver el traceback de Python.
Error 'Servidor no responde'
El servidor Flask está inactivo.
Vuelve a ejecutar python app.py.
Error 'Error HTTP de Ollama: 404'
El modelo (llama3:8b) no está cargado o fue mal escrito.
Ejecuta ollama list para verificar el nombre exacto del modelo.
No hay voz (TTS)
Bloqueo del navegador o configuración de idioma.
Confirma que utterance.lang = 'es-ES' está en index.html y haz clic en la página antes de intentar que hable.


