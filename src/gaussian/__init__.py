# src/gaussian/__init__.py

# Importamos las implementaciones de Pure Python y NumPy desde gaussian_python.py
from .gaussian_python import gaussian_filter_pure_python, gaussian_filter_numpy

# Importamos la implementación de Cython
try:
    from .gaussian_cython import gaussian_filter_cython
except ImportError:
    print("Advertencia: El módulo Cython 'gaussian_cython' no ha sido compilado o no se encuentra.")
    print("Por favor, ejecuta 'python setup.py build_ext --inplace' en la raíz del proyecto.")
    gaussian_filter_cython = None

# Definimos qué funciones se exportan cuando alguien usa "from gaussian import *"
__all__ = [
    "gaussian_filter_pure_python",
    "gaussian_filter_numpy",
    "gaussian_filter_cython"
]