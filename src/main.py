# src/main.py
import time
import os
from utils import load_image, save_image, plot_comparison

# Importamos las funciones de los 3 filtros (asumiendo que las crearemos con estos nombres)
from gaussian import gaussian_filter_pure_python, gaussian_filter_numpy, gaussian_filter_cython
from sobel import sobel_filter_pure_python, sobel_filter_numpy, sobel_filter_cython
from median import median_filter_pure_python, median_filter_numpy, median_filter_cython

def medir_tiempo(funcion, imagen, nombre_filtro):
    """Función auxiliar para no repetir tanto código de medición de tiempo."""
    inicio = time.time()
    resultado = funcion(imagen)
    fin = time.time()
    tiempo = fin - inicio
    print(f"{nombre_filtro}: {tiempo:.4f} segundos")
    return resultado

def main():
    nombre_imagen = "test_image.jpg" 
    input_path = os.path.join("input_images", nombre_imagen)
    
    try:
        img_gris = load_image(input_path)
        print(f"--- Procesando imagen: {nombre_imagen} ({img_gris.shape}) ---\n")
    except Exception as e:
        print(e)
        return

    img_list = img_gris.tolist() # Para las versiones Pure Python

    # Gaussian
    print(">>> FILTRO GAUSSIANO <<<")
    res_gauss_py = medir_tiempo(gaussian_filter_pure_python, img_list, "Pure Python")
    res_gauss_np = medir_tiempo(gaussian_filter_numpy, img_gris, "NumPy")
    if gaussian_filter_cython:
        res_gauss_cy = medir_tiempo(gaussian_filter_cython, img_gris, "Cython")
    print("-" * 30)

    # Sobel
    print(">>> FILTRO SOBEL (BORDES) <<<")
    res_sobel_py = medir_tiempo(sobel_filter_pure_python, img_list, "Pure Python")
    res_sobel_np = medir_tiempo(sobel_filter_numpy, img_gris, "NumPy")
    if sobel_filter_cython:
        res_sobel_cy = medir_tiempo(sobel_filter_cython, img_gris, "Cython")
    print("-" * 30)

    # Median
    print(">>> FILTRO MEDIAN <<<")
    res_median_py = medir_tiempo(median_filter_pure_python, img_list, "Pure Python")
    res_median_np = medir_tiempo(median_filter_numpy, img_gris, "NumPy")
    if median_filter_cython:
        res_median_cy = medir_tiempo(median_filter_cython, img_gris, "Cython")
    print("-" * 30)

    # --- GUARDAR Y MOSTRAR RESULTADOS (Usaremos los de NumPy/Cython como ejemplo) ---
    save_image(res_gauss_np, os.path.join("output_images", "gaussian_result.jpg"))
    save_image(res_sobel_np, os.path.join("output_images", "sobel_result.jpg"))
    save_image(res_median_np, os.path.join("output_images", "median_result.jpg"))

    # Mostrar visualmente (puedes comentar esto si solo quieres ver los tiempos en consola)
    plot_comparison(img_gris, res_gauss_np, "Filtro Gaussiano")
    plot_comparison(img_gris, res_sobel_np, "Filtro Sobel")
    plot_comparison(img_gris, res_median_np, "Filtro Mediana")

if __name__ == "__main__":
    main()