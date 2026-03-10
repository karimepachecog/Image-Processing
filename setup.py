from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Definimos las extensiones incluyendo 'src.' en el nombre del módulo
# para que se guarden en la carpeta correcta de tus compañeros.
extensions = [
    Extension(
        "src.gaussian.gaussian_cython", 
        ["src/gaussian/gaussian_cython.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "src.median.median_filter_cython", 
        ["src/median/median_filter_cython.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "src.sobel.sobel_cython", 
        ["src/sobel/sobel_cython.pyx"],
        include_dirs=[np.get_include()]
    ),
]

setup(
    ext_modules=cythonize(extensions),
)