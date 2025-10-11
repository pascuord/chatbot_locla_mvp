import sqlite3

DB_NAME = 'db_cosmetica.db'

def setup_db():
    """Asegura que la tabla de productos_scraped existe."""
    try:
        # 1. Conexión a la base de datos (si no existe, la crea)
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # 2. Creación de la tabla
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS productos_scraped (
                nombre TEXT PRIMARY KEY,
                descripcion TEXT,
                ingredientes TEXT
            );
        """)
        
        conn.commit()
        print(f"✅ La base de datos '{DB_NAME}' y la tabla 'productos_scraped' han sido inicializadas correctamente.")
        
    except sqlite3.Error as e:
        print(f"❌ Error al inicializar la base de datos: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    setup_db()