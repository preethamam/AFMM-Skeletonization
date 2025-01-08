#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "skeletongrad.h"

namespace py = pybind11;

// Main wrapper function for the skeleton gradient computation
std::tuple<py::array_t<double>, py::array_t<double>> 
compute_skeleton_gradient_py(py::array_t<uint8_t> input) {
    // Get input buffer info
    py::buffer_info buf = input.request();
    
    // Check dimensions
    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D array");
    }

    // Get dimensions - numpy arrays are row-major by default
    int nrow = buf.shape[0];  // height
    int ncol = buf.shape[1];  // width
    
    // Create output arrays
    auto skg = py::array_t<double>(buf.size);
    auto rad = py::array_t<double>(buf.size);
    
    // Get pointers to data
    uint8_t* img_ptr = static_cast<uint8_t*>(buf.ptr);
    double* skg_ptr = static_cast<double*>(skg.request().ptr);
    double* rad_ptr = static_cast<double*>(rad.request().ptr);

    // Create a temporary array with correct memory layout
    unsigned char* img_col_major = new unsigned char[nrow * ncol];
    
    // Convert from row-major (numpy) to column-major (algorithm expects)
    for (int j = 0; j < ncol; j++) {
        for (int i = 0; i < nrow; i++) {
            // Convert numpy's row-major to column-major
            img_col_major[i + j*nrow] = img_ptr[i*ncol + j] > 0 ? 1 : 0;
        }
    }

    // Call the original function
    compute_skeleton_gradient(img_col_major, nrow, ncol, skg_ptr, rad_ptr);

    // Convert output back from column-major to row-major
    auto skg_rowmajor = py::array_t<double>({nrow, ncol});
    auto rad_rowmajor = py::array_t<double>({nrow, ncol});
    double* skg_row_ptr = static_cast<double*>(skg_rowmajor.request().ptr);
    double* rad_row_ptr = static_cast<double*>(rad_rowmajor.request().ptr);

    for (int j = 0; j < ncol; j++) {
        for (int i = 0; i < nrow; i++) {
            skg_row_ptr[i*ncol + j] = skg_ptr[i + j*nrow];
            rad_row_ptr[i*ncol + j] = rad_ptr[i + j*nrow];
        }
    }

    // Cleanup
    delete[] img_col_major;

    return std::make_tuple(skg_rowmajor, rad_rowmajor);
}

// Binding code
PYBIND11_MODULE(pyskelgrad, m) {
    m.doc() = "Skeletonization module using augmented fast marching method";
    
    m.def("compute_skeleton_gradient", &compute_skeleton_gradient_py,
          "Compute skeleton gradient of a binary image\n"
          "Args:\n"
          "    input_img: 2D numpy array of type uint8 (binary image)\n"
          "Returns:\n"
          "    Tuple of (skeleton_gradient, radius) as numpy arrays\n"
          "    - skeleton_gradient: 2D numpy array of gradients\n"
          "    - radius: 2D numpy array of distance transform values",
          py::arg("input_img"));
}