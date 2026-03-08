# src/utils.py
import cv2
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """Lee la imagen y la convierte a escala de grises."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen en: {image_path}")
    
    # Cargar en escala de grises (0-255)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return img

def save_image(image_array, output_path):
    """Guarda la imagen procesada en la carpeta de salida."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image_array)

def plot_comparison(original, processed, title="Resultado"):
    """Muestra una comparativa visual."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()