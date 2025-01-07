# Skeletonization Project

This project implements a skeletonization algorithm using C. The project includes reading an image, processing it to extract skeletons, and writing the output image.

## Project Structure

```
.vscode/
    launch.json
    settings.json
    tasks.json
build/
    .cmake/
        api/
            v1/
    cmake_install.cmake
    CMakeCache.txt
    CMakeFiles/
        3.29.5-msvc4/
            ...
        cmake.check_cache
        CMakeConfigureLog.yaml
        CMakeDirectoryInformation.cmake
        Makefile.cmake
        Makefile2
        pkgRedirects/
        progress.marks
        skelgrad.dir/
            build.make
            ...
        TargetDirectories.txt
    compile_commands.json
    Makefile
CMakeLists.txt
Readme.md
skelgrad.exe
src/
    main.c
    skeletongrad.c
    skeletongrad.h
stb_image_write.h
stb_image.h
```

## Prerequisites

- CMake 3.10 or higher
- GCC (MinGW for Windows)
- MSYS2 (for Windows)
- Visual Studio Build Tools (for Windows)

## Building the Project

1. **Clone the repository:**

    ```sh
    git clone https://github.com/preethamam/AFMM-Skeletonization.git
    cd <repository-directory>
    ```

2. **Configure the project using CMake:**

    ```sh
    cmake -S . -B build
    ```

3. **Build the project:**

    ```sh
    cmake --build build
    ```

## Running the Project

1. **Navigate to the build directory:**

    ```sh
    cd build
    ```

2. **Run the executable:**

    ```sh
    ./skelgrad <input_image> <output_image> <threshold>
    ```

    - `<input_image>`: Path to the input image file.
    - `<output_image>`: Path to the output image file.
    - `<threshold>`: Threshold value for skeletonization.

## Code Overview

### `main.c`

The `main.c` file contains the main function which handles command-line arguments, reads the input image, allocates memory for processing, and calls the skeletonization functions.

### `skeletongrad.c`

The `skeletongrad.c` file contains the implementation of the skeletonization algorithm and image processing functions.

### `skeletongrad.h`

The `skeletongrad.h` file contains the declarations of the functions used in `skeletongrad.c`.

### `stb_image.h` and `stb_image_write.h`

These files are used for reading and writing images. They are part of the [stb single-file public domain libraries](https://github.com/nothings/stb).

## Example

```sh
./skelgrad input.png output.png 25
```

This command will read `input.png`, process it with a threshold of `25`, and save the result to `output.png`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [stb_image](https://github.com/nothings/stb) for image reading and writing functions.

## Troubleshooting

If you encounter any issues, please check the following:

- Ensure all prerequisites are installed.
- Verify the paths to input and output images.
- Check the threshold value is a valid number.

For further assistance, please open an issue in the repository.

## Cython compilation
python setup.py build_ext --inplace
