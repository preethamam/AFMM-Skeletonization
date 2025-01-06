/*
C code translation and implementation of Skeletonization
------------------------------------------------------------

Written by Dr. Preetham Manjunatha
Packaged December 2024

This package comes with no warranty of any kind (see below).


Description
-----------

The files in this package comprise the Python implementation of a
method for skeletonizing binary images.  

Article reference: 
Telea, Alexandru, and Jarke J. Van Wijk. "An augmented fast marching 
method for computing skeletons and centerlines." In EPRINTS-BOOK-TITLE. 
University of Groningen, Johann Bernoulli Institute for Mathematics and
Computer Science, 2002.

This implementation of a skeletonization method is the MATLAB/MEX C code 
translation and optimization of the original MATLAB/MEX C code written by Nicholas Howe. 
Weblink: https://www.mathworks.com/matlabcentral/fileexchange/11123-better-skeletonization

The code is written in C and is intended to be compiled with a C compiler.

Alex (article author) has a faster C/C++ (non-Matlab) implementation. It can be found at:

1. https://webspace.science.uu.nl/~telea001/uploads/Software/AFMM/
2. https://webspace.science.uu.nl/~telea001/Software/Software.

Execute main.c | ./skelgrad.exe input.jpg output.png 5.0


Copyright Notice
----------------
This software comes with no warranty, expressed or implied, including
but not limited to merchantability or fitness for any particular
purpose.

All files are copyright Dr. Preetham Manjunatha.  
Permission is granted to use the material for noncommercial and 
research purposes.
*/

#include <stdio.h>
#include <stdlib.h>
#include "skeletongrad.h"  // This brings in all the function declarations

// Main function showing explicit use of skeletongrad.c functions
int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_image> <output_image> <threshold>\n", argv[0]);
        fprintf(stderr, "Supported formats: JPG, PNG\n");
        return 1;
    }

    // Parse command line arguments
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    double threshold = atof(argv[3]);

    int nrow, ncol;
    // Using read_image from skeletongrad.c
    unsigned char* input_image = read_image(input_file, &nrow, &ncol);
    if (!input_image) {
        return 1;
    }

    printf("Image loaded: %dx%d pixels\n", nrow, ncol);

    // Allocate memory for output arrays
    double* skg = (double*)calloc(nrow * ncol, sizeof(double));
    double* rad = (double*)calloc(nrow * ncol, sizeof(double));
    unsigned char* output_image = (unsigned char*)calloc(nrow * ncol, sizeof(unsigned char));

    if (!skg || !rad || !output_image) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(input_image);
        free(skg);
        free(rad);
        free(output_image); 
        return 1;
    }

    // Using get_time from skeletongrad.c
    printf("Computing skeleton gradient...\n");
    double start_time = get_time();
    
    // Using compute_skeleton_gradient from skeletongrad.c
    compute_skeleton_gradient(input_image, nrow, ncol, skg, rad);
    
    double end_time = get_time();
    double elapsed_time = end_time - start_time;
    
    printf("Computation time: %.3f seconds\n", elapsed_time);
    printf("Processing speed: %.2f megapixels/second\n", 
           (nrow * ncol) / (elapsed_time * 1e6));

    // Threshold the skeleton gradient
    for (int i = 0; i < nrow * ncol; i++) {
        output_image[i] = (skg[i] > threshold) ? 1 : 0;
    }

    // Using write_image from skeletongrad.c
    if (!write_image(output_file, output_image, nrow, ncol)) {
        fprintf(stderr, "Error: Failed to write output image\n");
    } else {
        printf("Successfully processed image:\n");
        printf("Input: %s\n", input_file);
        printf("Output: %s\n", output_file);
        printf("Threshold: %f\n", threshold);
        
        // Print some statistics
        int skeleton_pixels = 0;
        for (int i = 0; i < nrow * ncol; i++) {
            if (output_image[i]) skeleton_pixels++;
        }
        printf("Skeleton pixels: %d (%.2f%% of image)\n", 
               skeleton_pixels, 
               100.0 * skeleton_pixels / (nrow * ncol));
    }

    // Free memory
    free(input_image);
    free(skg);
    free(rad);
    free(output_image);

    return 0;
}