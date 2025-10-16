import sqlite3
from collections import Counter
import re
import os

DB_NAME = 'db_cosmetica.db'

def get_top_ingredients(limit=100):
    """
    Extrae la columna de ingredientes de todos los productos y cuenta las repeticiones 
    de los componentes principales.
    """
    if not os.path.exists(DB_NAME):
        print(f"❌ ERROR: No se encontró la base de datos '{DB_NAME}'. Asegúrate de ejecutar el scraper primero.")
        return Counter()

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # 1. Recuperar TODOS los ingredientes de la base de datos
        cursor.execute("SELECT ingredientes FROM productos_scraped WHERE ingredientes IS NOT NULL")
        all_ingredients_rows = cursor.fetchall()
        conn.close()
        
        # Inicializar el contador
        ingredient_counts = Counter()
        
        # 2. Procesar y contar
        for (ingredients_list,) in all_ingredients_rows:
            
            # Limpieza y división de la cadena de ingredientes:
            # Reemplaza comas, puntos y paréntesis por espacios para separar componentes.
            cleaned_list = re.sub(r'[,.;()\n\r\s]+', ' ', ingredients_list)
            
            # Divide la cadena en palabras (ingredientes)
            individual_ingredients = cleaned_list.split()
            
            # Usamos un conjunto (set) para evitar contar el mismo ingrediente varias veces
            # dentro del mismo producto.
            unique_ingredients_in_product = set()
            
            for ingredient in individual_ingredients:
                ingredient = ingredient.strip()
                
                # Filtros para ignorar palabras comunes y cortas que no son INCI
                if len(ingredient) > 3 and not ingredient.isnumeric():
                    # Normalizar a minúsculas para contar juntos "Aqua" y "aqua"
                    unique_ingredients_in_product.add(ingredient.lower())
            
            # Actualizar el contador
            ingredient_counts.update(unique_ingredients_in_product)
            
        # 3. Eliminar términos de poco valor que pueden haber pasado el filtro (Ej.: 'is', 'the', 'a')
        # Este paso es crucial, ya que algunos productos tienen texto descriptivo en la columna ingredientes.
        del ingredient_counts['water'] # El agua es el ingrediente más común, suele ser irrelevante
        del ingredient_counts['aqua']
        del ingredient_counts['acid']
        del ingredient_counts['oil']
        
        # 4. Devolver los N ingredientes más comunes
        return ingredient_counts.most_common(limit)

    except sqlite3.Error as e:
        print(f"❌ Error al consultar la base de datos: {e}")
        return Counter()
    except Exception as e:
        print(f"❌ Error durante el procesamiento de texto: {e}")
        return Counter()

if __name__ == '__main__':
    print("=========================================")
    print("     📊 ANALIZADOR DE INGREDIENTES TOP    ")
    print("=========================================")
    
    top_20 = get_top_ingredients(limit=100)
    
    if top_20:
        print(f"\n✅ Los 20 ingredientes más repetidos en {DB_NAME}:")
        print("-----------------------------------------")
        for rank, (ingredient, count) in enumerate(top_20):
            print(f"{rank + 1: <3}. {ingredient.capitalize():<30} (En {count} productos)")
        print("-----------------------------------------")
    else:
        print("\nNo se pudieron extraer los datos o la base de datos está vacía.")