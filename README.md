# Image Processing Filters: Performance Analysis

## Overview
This project implements three fundamental image processing filters (Gaussian, Sobel, and Median) from scratch to analyze their computational performance. The goal is to compare execution times across three different implementation approaches:
1. **Pure Python:** Nested loops using native lists (no external optimizations).
2. **NumPy:** Vectorized matrix operations (slicing) for high-speed computation.
3. **NumPy + Cython:** C-compiled code with static typing to achieve maximum performance.

## Filters Implemented

### 1. Gaussian Filter (Blurring/Noise Reduction)
Smoothes the image by applying a weighted average to neighboring pixels. It uses the standard $3\times3$ Gaussian kernel:

$$G=\frac{1}{16}\left[\begin{matrix}1&2&1\\ 2&4&2\\ 1&2&1\end{matrix}\right]$$

### 2. Sobel Filter (Edge Detection)
Detects edges by computing the image gradient in both horizontal and vertical directions. The standard Sobel kernels are:

**X-Direction:**
$$S_{x}=\left[\begin{matrix}-1&0&1\\ -2&0&2\\ -1&0&1\end{matrix}\right]$$

**Y-Direction:**
$$S_{y}=\left[\begin{matrix}-1&-2&-1\\ 0&0&0\\ 1&2&1\end{matrix}\right]$$

The final edge magnitude is calculated combining both gradients:
$$G=\sqrt{S_{x}^{2}+S_{y}^{2}}$$

### 3. Median Filter (Noise Reduction)
Effective at removing salt-and-pepper noise while preserving edges. It replaces the central pixel's value with the median value of its $3\times3$ neighborhood.

## Project Structure

* `input_images/`: Contains the original source images to be processed.
* `output_images/`: Stores the resulting images after applying the filters.
* `src/`: Contains all the source code.
  * `main.py`: Main execution script that orchestrates the workflow and measures execution time.
  * `utils.py`: Helper functions for loading/saving images and plotting results.
  * `gaussian/`, `sobel/`, `median/`: Modules containing the three implementations (.py and .pyx) for each filter.
* `setup.py`: Configuration file to compile all Cython scripts.
* `requirements.txt`: Python dependencies.
* `Reporte_Filtros_U2.pdf`: Final performance analysis report.

## Setup Instructions

1. **Clone the repository and navigate to the project folder:**
```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
```

2. **Install dependencies:** Make sure you have Python 3 installed, then run:
```bash
   pip install -r requirements.txt
```

3. **Compile Cython code:** Before running the main script, you must compile the `.pyx` files into C extensions. Run the following command from the root directory:
```bash
   python setup.py build_ext --inplace
```

## Execution
Once compiled, run the main script to process the image and display the performance comparison:
```bash
python src/main.py
```

Check the console output for the execution time analysis and the `output_images/` folder for the processed results.