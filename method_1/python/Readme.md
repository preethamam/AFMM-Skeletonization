# Python Implementation of Skeletonization

## Description

This repository contains the Python implementation of a method for skeletonizing binary images. The implementation is based on the work by Telea and Van Wijk, and it is a translation and optimization of the original C++ code written by Nicholas Howe.

### Article Reference
Telea, Alexandru, and Jarke J. Van Wijk. "An augmented fast marching method for computing skeletons and centerlines." In EPRINTS-BOOK-TITLE. University of Groningen, Johann Bernoulli Institute for Mathematics and Computer Science, 2002.

### Original Implementation
The original MATLAB/MEX C++ code can be found at:
- [MathWorks File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/11123-better-skeletonization)

### Faster C/C++ Implementation
A faster C/C++ implementation by Alex (article author) can be found at:
- [AFMM Software](https://webspace.science.uu.nl/~telea001/uploads/Software/AFMM/)
- [Software](https://webspace.science.uu.nl/~telea001/Software/Software)

## Installation

To use this repository, you need to have Python installed along with the following packages:
- `numpy`
- `matplotlib`
- `Pillow`
- `scikit-image`
- `numba`
- `joblib`

You can install the required packages using pip:
```sh
pip install numpy matplotlib Pillow scikit-image numba joblib
```

or

Install the required Python packages:

```sh
pip install -r requirements.txt
```

## Building the Project

**Clone the repository:**

```sh
git clone https://github.com/preethamam/AFMM-Skeletonization.git
cd <repository-directory>
```

## Usage

### Running the Skeletonization
To run the skeletonization process, execute the `main.py` script:
```sh
python main.py
```

### Example
The `main.py` script loads a binary image, computes the skeleton gradient transform and radius, performs skeletonization, and displays the results. The script also saves the thinned skeleton image as `bagel_m1_python.png`.

### Code Structure
- `main.py`: Main script to execute the skeletonization process.
- `skeleton.py`: Contains the implementation of the skeletonization method.
- `Readme.md`: This file.

### Execution Steps
1. Load the binary image from the specified path.
2. Convert the image to grayscale if it is not already.
3. Compute the skeleton gradient transform and radius using the `compute_skeleton_gradient` function from `skeleton.py`.
4. Perform skeletonization on the thresholded skeleton gradient.
5. Thin the skeleton.
6. Save the thinned skeleton image.
7. Display the original binary image, skeleton gradient, skeleton, and thinned skeleton using matplotlib.

## License

This software comes with no warranty, expressed or implied, including but not limited to merchantability or fitness for any particular purpose.

All files are copyright Dr. Preetham Manjunatha. Permission is granted to use the material for noncommercial and research purposes.

## Author

Written by Dr. Preetham Manjunatha
Packaged December 2024
