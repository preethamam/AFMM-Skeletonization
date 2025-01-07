# AFMM - Augmented Fast Marching Method

This repository contains the C implementation of the Augmented Fast Marching Method (AFMM) for computing skeletons and centerlines of binary images. The implementation is based on the work of Dr. Preetham Manjunatha and is a translation and optimization of the original Go code by João Rafael Diniz Ramos.

## Description

The AFMM method is used for skeletonizing binary images. It is based on the following articles:
1. Telea, Alexandru, and Jarke J. Van Wijk. "An augmented fast marching method for computing skeletons and centerlines." In EPRINTS-BOOK-TITLE. University of Groningen, Johann Bernoulli Institute for Mathematics and Computer Science, 2002.
2. Reniers, Dennie & Telea, Alexandru. (2007). Tolerance-Based Feature Transforms. 10.1007/978-3-540-75274-5_12.

## Repository Structure

```
.vscode/
    launch.json
    tasks.json
afmm.exe
CMakeLists.txt
Readme.md
src/
    afmm.c
    afmm.h
    main.c
stb_image_write.h
stb_image.h
```

## Prerequisites

- CMake (version 3.10 or higher)
- A C compiler (GCC, Clang, MSVC, etc.)
- Python (for Cython compilation, if needed)

## Building the Project

1. Clone the repository:
    ```sh
    git clone https://github.com/preethamam/AFMM-Skeletonization.git
    cd <repository-directory>
    ```

2. Create a build directory and navigate to it:
    ```sh
    mkdir build
    cd build
    ```

3. Run CMake to configure the project:
    ```sh
    cmake ..
    ```

4. Build the project:
    ```sh
    cmake --build .
    ```

## Running the Executable

The executable `afmm.exe` can be run with the following command:
```sh
./afmm.exe <input_image> <threshold> <is_rgb>
```

- `<input_image>`: Path to the input image file.
- `<threshold>`: Threshold value for skeletonization.
- `<is_rgb>`: Set to `1` if the input image is RGB channels (binary image with three channels), otherwise set to `0`.

### Example
```sh
./afmm.exe example.png 100 1
```

## Output

The program generates the following output files:
- `<input_image>_dt.png`: Distance Transform image.
- `<input_image>_deltaU.png`: DeltaU Map image.
- `<input_image>_skeleton.png`: Skeleton image.

## Cython Compilation

If you need to compile Cython extensions, run the following command:
```sh
python setup.py build_ext --inplace
```

## License

This software comes with no warranty, expressed or implied, including but not limited to merchantability or fitness for any particular purpose. All files are copyright Dr. Preetham Manjunatha. Permission is granted to use the material for noncommercial and research purposes.

## References

1. Telea, Alexandru, and Jarke J. Van Wijk. "An augmented fast marching method for computing skeletons and centerlines." In EPRINTS-BOOK-TITLE. University of Groningen, Johann Bernoulli Institute for Mathematics and Computer Science, 2002.
2. Reniers, Dennie & Telea, Alexandru. (2007). Tolerance-Based Feature Transforms. 10.1007/978-3-540-75274-5_12.

For more information, visit the original Go code repository by João Rafael Diniz Ramos: [https://github.com/Joao-R/afmm](https://github.com/Joao-R/afmm)
