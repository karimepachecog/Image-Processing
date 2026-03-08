# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import glob

# Usamos glob para encontrar automáticamente todos los archivos .pyx 
# dentro de cualquier subcarpeta de 'src' (gaussian, sobel, median)
archivos_cython = glob.glob("src/*/*.pyx")

setup(
    name="Filtros de Procesamiento de Imagenes",
    ext_modules=cythonize(
        archivos_cython,
        compiler_directives={'language_level': "3"} # Le decimos que use la sintaxis de Python 3
    ),
    include_dirs=[np.get_include()] 
)