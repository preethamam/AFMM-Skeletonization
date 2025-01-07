# Skeletonization using Fast Marching Method (FMM) and Augmented Fast Marching Method (AFMM)

This repository provides an implementation of skeletonization using the Fast Marching Method (FMM) and Augmented Fast Marching Method (AFMM). The code processes images to extract skeletons, which are useful in various image processing and computer vision applications.

## Repository Structure

```
afmm.py
datastructure.py
logging_config.py
main.py
Readme.md
scratchpad.py
skeleton.log
utils.py
```

- `afmm.py`: Contains the implementation of the AFMM and FMM algorithms.
- `datastructure.py`: Defines the data structures used in the algorithms.
- `logging_config.py`: Configures the logging settings.
- `main.py`: The main script to run the skeletonization process.
- `scratchpad.py`: A scratchpad for testing and experimenting with code snippets.
- `skeleton.log`: Log file generated during execution.
- `utils.py`: Utility functions used in the algorithms.
- `Readme.md`: This file.

## Dependencies

The following Python packages are required to run the code:

- `numpy`
- `numba`
- `Pillow`
- `scikit-image`
- `matplotlib`

You can install the dependencies using pip:

```sh
pip install numpy numba Pillow scikit-image matplotlib
```

## Building the Project

**Clone the repository:**

```sh
git clone https://github.com/preethamam/AFMM-Skeletonization.git
cd <repository-directory>
```

## Usage

### Cython Compilation

If you have any Cython extensions, you can compile them using the following command:

```sh
python setup.py build_ext --inplace
```

### Running the Skeletonization

1. Place the image you want to process in the appropriate directory (e.g., `../../imgs/example.png`).
2. Open `main.py` and adjust the parameters as needed:
   - `threshold`: The threshold value for skeletonization.
   - `fmm_method`: Choose between `"fmm"` and `"afmm"` for the method.
   - `parse_image_type`: Choose between `"binary"` and `"rgb"` for the image type.
   - `show_afmm_deltaU_dt`: Set to `True` to display both Delta U and DT images, or `False` to display only the skeleton.

3. Run the `main.py` script:

```sh
python main.py
```

### Output

The script will generate the following outputs:
- `fmm_dt.png` or `afmm_dt.png`: The distance transform image.
- `afmm_dt_both.png`: (Optional) Both Delta U and DT images.
- `skeleton.png`: The skeletonized image.

### Logging

Execution logs are saved in `skeleton.log`.

## Example

Here is an example of how to run the skeletonization process:

```sh
python main.py
```

This will process the image located at `../../images/example.png` and generate the skeletonized image `skeleton.png`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the following libraries:
- [NumPy](https://numpy.org/)
- [Numba](https://numba.pydata.org/)
- [Pillow](https://python-pillow.org/)
- [scikit-image](https://scikit-image.org/)
- [Matplotlib](https://matplotlib.org/)

# Authors

1. Dr. Preetham Manjunatha, Ph.D in Civil Engineering, M.S in Computer Science, M.S in Electrical Engineering and M.S in Civil Engineering, University of Southern California.

2. Aniketh Manjunath ([vma1996](https://github.com/vma1996)), M.S in Computer Science, University of Southern California.

## Contact

For any questions or issues, please open an issue on the GitHub repository.
