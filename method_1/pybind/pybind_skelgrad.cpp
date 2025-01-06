#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "skeletongrad.h"

namespace py = pybind11;

// Main wrapper function for the skeleton gradient computation
py::tuple compute_skeleton_gradient_py(py::array_t<unsigned char, py::array::c_style> input_img) {
    // Get input array info
    py::buffer_info buf_img = input_img.request();
    
    if (buf_img.ndim != 2) {
        throw std::runtime_error("Input image must be 2-dimensional");
    }
    
    // Get dimensions
    int nrow = buf_img.shape[0];
    int ncol = buf_img.shape[1];
    
    // Create output arrays for skeleton gradient and radius
    auto skg = py::array_t<double>(buf_img.size);
    auto rad = py::array_t<double>(buf_img.size);
    
    py::buffer_info buf_skg = skg.request();
    py::buffer_info buf_rad = rad.request();
    
    // Get pointers to data
    unsigned char* img_ptr = static_cast<unsigned char*>(buf_img.ptr);
    double* skg_ptr = static_cast<double*>(buf_skg.ptr);
    double* rad_ptr = static_cast<double*>(buf_rad.ptr);
    
    // Call C function
    compute_skeleton_gradient(img_ptr, nrow, ncol, skg_ptr, rad_ptr);
    
    // Reshape output arrays to match input dimensions
    skg.resize({nrow, ncol});
    rad.resize({nrow, ncol});
    
    // Return tuple of results
    return py::make_tuple(skg, rad);
}

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