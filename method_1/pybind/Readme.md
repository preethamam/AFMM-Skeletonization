## Skeletonization with AFMM

This repository contains a Python implementation of a method for skeletonizing binary images using the Augmented Fast Marching Method (AFMM). The implementation includes Python bindings for the AFMM algorithm, which is written in C and exposed to Python using PyBind11.

### Repository Structure

```
build/
	lib.win-amd64-cpython-312/
		pyskelgrad.cp312-win_amd64.pyd
	temp.win-amd64-cpython-312/
		Release/
			pybind_skelgrad.obj
			pyskelgrad.cp312-win_amd64.exp
			...
main.py
pybind_skelgrad.cpp
pybind_skelgrad.obj
pyskelgrad.cp312-win_amd64.pyd
pyskelgrad.egg-info/
	dependency_links.txt
	not-zip-safe
	PKG-INFO
	requires.txt
	SOURCES.txt
	top_level.txt
Readme.md
setup_command.txt
setup.py
skeletongrad.c
skeletongrad.h
skeletongrad.obj
stb_image_write.h
stb_image.h
```

### Prerequisites

- Python 3.6 or higher
- C++ compiler
- [PyBind11](https://pybind11.readthedocs.io/en/stable/)
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
- [Matplotlib](https://matplotlib.org/)

### Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Build the C++ extension:
    ```sh
    python setup.py build_ext --inplace
    ```

3. Install the package:
    ```sh
    pip install -e .
    ```

### Usage

The main script `main.py` demonstrates how to use the AFMM skeletonization method. It processes an input image, computes the skeleton gradient, performs skeletonization, and saves the result.

#### Example

1. Place your input image in the appropriate directory (e.g., `../../images/crack2.png`).

2. Run the main script:
    ```sh
    python main.py
    ```

3. The script will process the image, compute the skeleton gradient, perform skeletonization, and save the result as `crack2_m1_pybind.png`.

### Code Overview

#### `main.py`

This script contains the main logic for processing the image and performing skeletonization.

```python
def main():
    // ...existing code...
    # Image path
    image_path = "../../images/crack2.png"
    
    # Threshold for binary conversion
    threshold = 25
    
    # Process image
    skg, rad, img = process_image(image_path)
    
    # Perform skeletonization
    skeleton = skeletonize(skg > threshold)

    # Save the image
    skeleton_thinned_pil = Image.fromarray(skeleton)
    skeleton_thinned_pil.save("crack2_m1_pybind.png") 
    
    # Visualize results        
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    // ...existing code...
```

#### `skeletongrad.c`

This file contains the C implementation of the AFMM skeletonization algorithm.

#### `pybind_skelgrad.cpp`

This file contains the PyBind11 bindings to expose the C implementation to Python.

#### `setup.py`

This file contains the setup configuration for building the C++ extension and installing the package.

### License

This software is provided "as-is," without any express or implied warranty. In no event shall the authors be held liable for any damages arising from the use of this software.

### References

- Telea, Alexandru, and Jarke J. Van Wijk. "An augmented fast marching method for computing skeletons and centerlines." In EPRINTS-BOOK-TITLE. University of Groningen, Johann Bernoulli Institute for Mathematics and Computer Science, 2002.
- Original MATLAB/MEX C code by Nicholas Howe: [Better Skeletonization](https://www.mathworks.com/matlabcentral/fileexchange/11123-better-skeletonization)

For more details, refer to the [AFMM Skeletonization](https://webspace.science.uu.nl/~telea001/uploads/Software/AFMM/) page.
