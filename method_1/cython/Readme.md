# Skeleton Gradient Computation

This repository contains a Python project for computing skeleton gradients using Cython for performance optimization. The project includes a Cython extension module and a Python script to demonstrate its usage.

## Project Structure

```
build/
	lib.win-amd64-cpython-312/
		skelgrad.cp312-win_amd64.pyd
	temp.win-amd64-cpython-312/
		Release/
			skelgrad.cp312-win_amd64.exp
			skelgrad.cp312-win_amd64.lib
			skelgrad.obj
main.py
Readme.md
scratchpad.py
setup.py
skelgrad_pythonic_slow.pyx
skelgrad.cp312-win_amd64.pyd
skelgrad.cpp
skelgrad.pyx
```

- `main.py`: Main script to execute the skeleton gradient computation.
- `Readme.md`: This file, providing an overview and instructions.
- `scratchpad.py`: A scratchpad for testing and development.
- `setup.py`: Setup script for building the Cython extension.
- `skelgrad_pythonic_slow.pyx`: A slower, more Pythonic version of the Cython module.
- `skelgrad.cp312-win_amd64.pyd`: Compiled Cython extension module.
- `skelgrad.cpp`: C++ source file for the Cython extension.
- `skelgrad.pyx`: Cython source file for the skeleton gradient computation.

## Requirements

- Python 3.12
- Cython
- NumPy
- Matplotlib
- Pillow
- scikit-image

You can install the required packages using pip:

```sh
pip install cython numpy matplotlib pillow scikit-image
```

## Cython Compilation

To compile the Cython extension module, run the following command:

```sh
python setup.py build_ext --inplace
```

This will generate the `skelgrad.cp312-win_amd64.pyd` file in the current directory.

## Usage

To execute the main script and compute the skeleton gradient, run:

```sh
python main.py
```

The script will print the execution time for the complete skeletonization process.

## Example

Here is an example of how to use the `compute_skeleton_gradient` function from the Cython extension module:

```py
import numpy as np
from skimage.morphology import skeletonize
from skelgrad import compute_skeleton_gradient

# Example binary image
image = np.array([[0, 1, 1, 0],
                  [1, 1, 1, 1],
                  [0, 1, 1, 0],
                  [0, 0, 1, 0]], dtype=bool)

# Compute the skeleton
skeleton = skeletonize(image)

# Compute the skeleton gradient
gradient = compute_skeleton_gradient(skeleton)

print("Skeleton:")
print(skeleton)
print("Skeleton Gradient:")
print(gradient)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the following libraries:

- [Cython](https://cython.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow](https://python-pillow.org/)
- [scikit-image](https://scikit-image.org/)
