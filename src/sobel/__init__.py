# src/sobel/__init__.py

# Importamos las implementaciones de Pure Python y NumPy desde sobel_python.py
from .sobel_python import sobel_filter_pure_python, sobel_filter_numpy

# Importamos la implementación de Cython
try:
    from .sobel_cython import sobel_filter_cython
except ImportError:
    print("Advertencia: El módulo Cython 'sobel_cython' no ha sido compilado o no se encuentra.")
    print("Por favor, ejecuta 'python setup.py build_ext --inplace' en la raíz del proyecto.")
    sobel_filter_cython = None

# Definimos qué funciones se exportan cuando alguien usa "from sobel import *"
__all__ = [
    "sobel_filter_pure_python",
    "sobel_filter_numpy",
    "sobel_filter_cython"
]