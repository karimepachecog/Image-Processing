# src/gaussian/gaussian_cython.pyx

import numpy as np
cimport numpy as np
cimport cython

# Desactivamos comprobaciones de límites e índices negativos para máxima velocidad en C
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_filter_cython(unsigned char[:, :] image):
    """
    Aplica un filtro Gaussiano 3x3 usando Cython con vistas de memoria (memoryviews) de C.
    """
    cdef int rows = image.shape[0]
    cdef int cols = image.shape[1]
    
    # Creamos el arreglo de salida en NumPy, pero lo manipularemos a través de una vista de C
    cdef np.ndarray[np.uint8_t, ndim=2] output = np.zeros((rows, cols), dtype=np.uint8)
    cdef unsigned char[:, :] output_view = output
    
    # Declaramos explícitamente las variables de los bucles como tipos de C
    cdef int i, j, ki, kj
    cdef int row_offset, col_offset
    cdef double pixel_value
    
    # Definimos el kernel de 3x3 como un arreglo bidimensional estático de C
    cdef double[3][3] kernel = [
        [1.0/16.0, 2.0/16.0, 1.0/16.0],
        [2.0/16.0, 4.0/16.0, 2.0/16.0],
        [1.0/16.0, 2.0/16.0, 1.0/16.0]
    ]
    
    # Realizamos los bucles anidados procesando cada píxel
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            pixel_value = 0.0
            
            # Aplicamos el kernel 3x3 al vecindario
            for ki in range(3):
                for kj in range(3):
                    row_offset = i + ki - 1
                    col_offset = j + kj - 1
                    
                    # image[...] accede directamente a la memoria en C, lo cual es rapidísimo
                    pixel_value += image[row_offset, col_offset] * kernel[ki][kj]
            
            # Asignamos el valor calculado convirtiéndolo a entero sin signo (8 bits)
            output_view[i, j] = <unsigned char>pixel_value
            
    return output