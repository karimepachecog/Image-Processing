# src/sobel/sobel_cython.pyx
# ---------------------------
# Filtro Sobel — Implementación con Cython.
#
# Cython compila este archivo a C, logrando velocidad casi nativa.
# Optimizaciones aplicadas:
#   • Declaraciones de tipo C estático (cdef) eliminan el overhead de objetos Python.
#   • Typed MemoryViews reemplazan el indexado de NumPy con aritmética directa de punteros C.
#   • `sqrt` se llama desde libc en lugar del módulo math de Python.
#   • Las directivas `boundscheck` y `wraparound` se deshabilitan en los loops internos.
#
# CÓMO COMPILAR
# -------------
# Desde la raíz del proyecto ejecutar:
#   python setup.py build_ext --inplace

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as np

# sqrt de libc — evita el overhead de la llamada a función de Python
from libc.math cimport sqrt

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t


def sobel_filter_cython(np.ndarray[DTYPE_t, ndim=2] image_array):
    """
    Aplica el filtro Sobel usando código C compilado por Cython.

    Parámetros
    ----------
    image_array : np.ndarray, shape (H, W), dtype float32
        Imagen en escala de grises con valores en [0, 255].

    Retorna
    -------
    output_image : np.ndarray, shape (H, W), dtype uint8
        Imagen de magnitud de bordes, valores acotados a [0, 255].
    """
    # Declaramos todas las variables del loop como tipos C puros
    cdef int rows = image_array.shape[0]
    cdef int cols = image_array.shape[1]
    cdef int i, j

    cdef double gx, gy, magnitude

    # Array de salida inicializado en cero; los píxeles del borde permanecen en 0
    cdef np.ndarray[DTYPE_t, ndim=2] output_float = np.zeros((rows, cols), dtype=DTYPE)

    # Typed MemoryViews: acceso directo al buffer C sin boxing de objetos Python
    cdef DTYPE_t[:, :] img_view    = image_array.astype(DTYPE)
    cdef DTYPE_t[:, :] out_view    = output_float

    # ── Kernels Sobel (constantes hard-coded; sin overhead de listas Python) ──
    #
    #  Sx = [[-1, 0,  1],     Sy = [[-1, -2, -1],
    #        [-2, 0,  2],           [ 0,  0,  0],
    #        [-1, 0,  1]]           [ 1,  2,  1]]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):

            # Gradiente X
            gx = (
                -1.0 * img_view[i-1, j-1] + 0.0 * img_view[i-1, j] + 1.0 * img_view[i-1, j+1] +
                -2.0 * img_view[i,   j-1] + 0.0 * img_view[i,   j] + 2.0 * img_view[i,   j+1] +
                -1.0 * img_view[i+1, j-1] + 0.0 * img_view[i+1, j] + 1.0 * img_view[i+1, j+1]
            )

            # Gradiente Y
            gy = (
                -1.0 * img_view[i-1, j-1] + -2.0 * img_view[i-1, j] + -1.0 * img_view[i-1, j+1] +
                 0.0 * img_view[i,   j-1] +  0.0 * img_view[i,   j] +  0.0 * img_view[i,   j+1] +
                 1.0 * img_view[i+1, j-1] +  2.0 * img_view[i+1, j] +  1.0 * img_view[i+1, j+1]
            )

            # Magnitud del gradiente usando sqrt de libc (sin overhead de Python)
            magnitude = sqrt(gx * gx + gy * gy)

            # Acotar a [0, 255]
            if magnitude > 255.0:
                magnitude = 255.0
            elif magnitude < 0.0:
                magnitude = 0.0

            out_view[i, j] = magnitude

    return output_float.astype(np.uint8)
