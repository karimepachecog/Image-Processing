# src/sobel/sobel_python.py
import math
import numpy as np

def sobel_filter_pure_python(image_list):
    """
    Aplica el filtro Sobel a una imagen representada como una lista de listas.
    Detecta bordes calculando el gradiente en dirección X e Y.
    Implementación en Pure Python (sin optimizaciones externas).

    Kernels utilizados:
        Sx = [[-1, 0, 1],      Sy = [[-1, -2, -1],
              [-2, 0, 2],             [ 0,  0,  0],
              [-1, 0, 1]]             [ 1,  2,  1]]

    Magnitud del gradiente: G = sqrt(Sx² + Sy²)
    """
    rows = len(image_list)
    cols = len(image_list[0])

    # Matriz de salida inicializada en ceros
    output_image = [[0 for _ in range(cols)] for _ in range(rows)]

    # Kernels Sobel como listas
    Sx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

    Sy = [[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]

    # Recorremos cada píxel ignorando los bordes
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            gx = 0.0
            gy = 0.0

            # Convolución 3×3 con cada kernel
            for ki in range(3):
                for kj in range(3):
                    pixel = image_list[i + ki - 1][j + kj - 1]
                    gx += Sx[ki][kj] * pixel
                    gy += Sy[ki][kj] * pixel

            # Magnitud del gradiente, acotada a [0, 255]
            magnitude = math.sqrt(gx ** 2 + gy ** 2)
            output_image[i][j] = int(min(255.0, max(0.0, magnitude)))

    return output_image


def sobel_filter_numpy(image_array):
    """
    Aplica el filtro Sobel usando operaciones vectorizadas de NumPy (slicing)
    para máxima optimización. Sin ningún loop de Python.

    Cada gradiente se calcula multiplicando slices desplazados de la imagen
    por los pesos del kernel y sumándolos en una sola operación.
    """
    img_float = image_array.astype(np.float32)

    # Gradiente X  (Sx)
    # Sx = [[-1, 0,  1],
    #       [-2, 0,  2],
    #       [-1, 0,  1]]
    gx = (
        -1.0 * img_float[:-2, :-2] + 0.0 * img_float[:-2, 1:-1] + 1.0 * img_float[:-2, 2:] +
        -2.0 * img_float[1:-1, :-2] + 0.0 * img_float[1:-1, 1:-1] + 2.0 * img_float[1:-1, 2:] +
        -1.0 * img_float[2:, :-2] + 0.0 * img_float[2:, 1:-1] + 1.0 * img_float[2:, 2:]
    )

    # Gradiente Y  (Sy)
    # Sy = [[-1, -2, -1],
    #       [ 0,  0,  0],
    #       [ 1,  2,  1]]
    gy = (
        -1.0 * img_float[:-2, :-2] + -2.0 * img_float[:-2, 1:-1] + -1.0 * img_float[:-2, 2:] +
         0.0 * img_float[1:-1, :-2] +  0.0 * img_float[1:-1, 1:-1] +  0.0 * img_float[1:-1, 2:] +
         1.0 * img_float[2:, :-2] +  2.0 * img_float[2:, 1:-1] +  1.0 * img_float[2:, 2:]
    )

    # Magnitud del gradiente
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Colocamos el resultado en un array del tamaño original (bordes quedan en 0)
    output_image = np.zeros_like(img_float)
    output_image[1:-1, 1:-1] = np.clip(magnitude, 0, 255)

    # Devolvemos como imagen de 8 bits (igual que gaussian_filter_numpy)
    return output_image.astype(np.uint8)
