/*
C code translation and implementation of Skeletonization
------------------------------------------------------------

Written by Dr. Preetham Manjunatha
Packaged December 2024

This package comes with no warranty of any kind (see below).


Description
-----------

The files in this package comprise the C implementation of a
AFMM method for skeletonizing binary images.  

Article reference: 
1. Telea, Alexandru, and Jarke J. Van Wijk. "An augmented fast marching 
method for computing skeletons and centerlines." In EPRINTS-BOOK-TITLE. 
University of Groningen, Johann Bernoulli Institute for Mathematics and
Computer Science, 2002.
2. Reniers, Dennie & Telea, Alexandru. (2007). Tolerance-Based Feature 
Transforms. 10.1007/978-3-540-75274-5_12.


This implementation of a skeletonization method is the Go code translation and 
optimization to C code originally written by João Rafael Diniz Ramos's. 
Weblink: https://github.com/Joao-R/afmm

The code is written in C and is intended to be compiled with a C compiler.

Alex (article author) has a faster C/C++ implementation. It can be found at:

1. https://webspace.science.uu.nl/~telea001/uploads/Software/AFMM/
2. https://webspace.science.uu.nl/~telea001/Software/Software.

Execute main.c | ./afmm.exe input.jpg 100 1 <input_image> <threshold> <is_rgb>


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
#include "afmm.h"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_image> <threshold> <is_rgb>\n", argv[0]);
        return 1;
    }

    // Parse command line arguments
    int is_rgb = atoi(argv[3]);

    // Load input image
    printf("Loading image %s...\n", argv[1]);
    clock_t load_start = clock();
    Image* img = load_image(argv[1], is_rgb);
    clock_t load_end = clock();
    print_elapsed_time(load_start, load_end, "Image loading");
    
    if (!img) {
        return 1;
    }
    printf("Image size: %dx%d pixels\n", img->width, img->height);

    // Generate output filenames
    char dt_filename[256];
    char deltaU_filename[256];
    char skeleton_filename[256];
    sprintf(dt_filename, "%s_dt.png", argv[1]);
    sprintf(deltaU_filename, "%s_deltaU.png", argv[1]);
    sprintf(skeleton_filename, "%s_skeleton.png", argv[1]);

    // Compute and save FMM result
    printf("\nComputing Fast Marching Method...\n");
    clock_t fmm_start = clock();
    double* DT = FMM(img, is_rgb);
    clock_t fmm_end = clock();
    print_elapsed_time(fmm_start, fmm_end, "FMM computation");
    
    // Save distance transform
    printf("Saving distance transform...\n");
    clock_t dt_save_start = clock();
    save_distance_transform(DT, img->width, img->height, dt_filename);
    clock_t dt_save_end = clock();
    print_elapsed_time(dt_save_start, dt_save_end, "Distance transform saving");
    free(DT);

    // Compute and save AFMM results and skeleton
    double* deltaU;
    double* afmm_DT;
    uint8_t* skeleton;

    printf("\nComputing AFMM and skeleton...\n");
    double threshold = atof(argv[2]);  // You might want to make this configurable
    clock_t skel_start = clock();
    Skeletonize(img, threshold, is_rgb, &skeleton, &deltaU, &afmm_DT);
    clock_t skel_end = clock();
    print_elapsed_time(skel_start, skel_end, "Skeleton computation");
    
    // Save AFMM results
    clock_t afmm_save_start = clock();
    save_deltaU(deltaU, img->width, img->height, deltaU_filename);
    clock_t afmm_save_end = clock();
    print_elapsed_time(afmm_save_start, afmm_save_end, "AFMM results saving");
    free(deltaU);
    free(afmm_DT);

    // Save skeleton
    printf("Saving skeleton...\n");
    clock_t skel_save_start = clock();
    save_skeleton(skeleton, img->width, img->height, skeleton_filename);
    clock_t skel_save_end = clock();
    print_elapsed_time(skel_save_start, skel_save_end, "Skeleton saving");
    free(skeleton);

    // Cleanup
    cleanup_image(img);

    printf("\nGenerated output files:\n");
    printf("  Distance Transform: %s\n", dt_filename);
    printf("  DeltaU Map: %s\n", deltaU_filename);
    printf("  Skeleton: %s\n", skeleton_filename);

    return 0;
}