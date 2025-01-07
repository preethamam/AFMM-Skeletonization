# AFMM Project

This project implements the AFMM (Adaptive Fast Marching Method) algorithm using Cython for performance optimization. The repository includes C/C++ and Python code to perform image processing tasks, specifically skeletonization.

## Repository Structure

```
afmm.cp312-win_amd64.pyd
afmm.cpp
afmm.pyx
build/
    lib.win-amd64-cpython-312/
        afmm.cp312-win_amd64.pyd
    temp.win-amd64-cpython-312/
        Release/
            afmm.cp312-win_amd64.exp
            afmm.cp312-win_amd64.lib
            afmm.obj
main.py
Readme.md
setup.py
skeleton.log
```

- `afmm.cpp`: C++ source file for the AFMM algorithm.
- `afmm.pyx`: Cython file that interfaces the C++ code with Python.
- `build/`: Directory containing build artifacts.
- `main.py`: Main Python script to execute the AFMM algorithm and visualize results.
- `Readme.md`: This file.
- `setup.py`: Setup script for building the Cython extension.
- `skeleton.log`: Log file for skeletonization process.

## Prerequisites

- Python 3.12
- Cython
- Matplotlib
- Pillow (PIL)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/preethamam/AFMM-Skeletonization.git
    cd <repository_directory>
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Compile the Cython extension:
    ```sh
    python setup.py build_ext --inplace
    ```

## Usage

1. Run the main script:
    ```sh
    python main.py
    ```

2. The script will execute the AFMM algorithm and generate visualizations of the results. The output images will be saved as `fmm_dt.png`, `afmm_dt.png`, and `skeleton.png`.

## Example

The `main.py` script demonstrates how to use the AFMM algorithm to process an image and visualize the results. It creates a figure with two subplots showing the distance transform (DT) and the delta U images. The skeleton is also displayed and saved as an image.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The AFMM algorithm is based on the work by Telea, Alexandru et. al. and `Go` implementation of João Ramos.
- The project uses Cython to interface C++ code with Python for performance optimization.
- Visualization is done using Matplotlib and Pillow.
