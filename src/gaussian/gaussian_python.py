import numpy as np

def gaussian_filter_pure_python(image_list):
    """
    Aplica un filtro Gaussiano 3x3 a una imagen representada como una lista de listas.
    Implementación en Pure Python (sin optimizaciones externas).
    """
    # Obtenemos las dimensiones de la imagen
    rows = len(image_list)
    cols = len(image_list[0])
    
    # Creamos una matriz vacía para la imagen de salida (lista de listas llena de ceros)
    output_image = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Definimos el kernel Gaussiano 3x3
    kernel = [
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]
    
    # Recorremos cada píxel ignorando los bordes
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            
            pixel_value = 0.0
            
            # Aplicamos el kernel 3x3 al vecindario
            for ki in range(3):
                for kj in range(3):
                    row_offset = i + ki - 1
                    col_offset = j + kj - 1
                    
                    pixel_value += image_list[row_offset][col_offset] * kernel[ki][kj]
            
            output_image[i][j] = int(pixel_value)
            
    return output_image


def gaussian_filter_numpy(image_array):
    """
    Aplica un filtro Gaussiano 3x3 a una imagen utilizando operaciones 
    vectorizadas de NumPy (slicing) para máxima optimización.
    """
    # Asegurarnos de que la imagen sea float para evitar desbordamientos
    img_float = image_array.astype(np.float32)
    
    # Creamos una matriz de ceros para la salida
    output_image = np.zeros_like(img_float)
    
    # Operaciones vectorizadas (slicing) para aplicar el kernel en toda la imagen a la vez
    output_image[1:-1, 1:-1] = (
        (img_float[:-2, :-2] * 1.0) + (img_float[:-2, 1:-1] * 2.0) + (img_float[:-2, 2:] * 1.0) +
        (img_float[1:-1, :-2] * 2.0) + (img_float[1:-1, 1:-1] * 4.0) + (img_float[1:-1, 2:] * 2.0) +
        (img_float[2:, :-2] * 1.0) + (img_float[2:, 1:-1] * 2.0) + (img_float[2:, 2:] * 1.0)
    ) / 16.0
    
    # Volvemos a convertir a formato de imagen de 8 bits (enteros)
    return output_image.astype(np.uint8)