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

#ifndef SKELETONGRAD_H
#define SKELETONGRAD_H

#include <stdbool.h>

// Direction enumeration
typedef enum {
    North, 
    South, 
    East, 
    West, 
    None
} direction;

// Function declarations
void quicksort(int *arr, int n);
int joint_neighborhood(const unsigned char *arr, int i, int j, int nrow, int ncol);
void compute_skeleton_gradient(const unsigned char *img, int nrow, int ncol, 
                             double *skg, double *rad);
unsigned char* read_image(const char* filename, int* nrow, int* ncol);
int write_image(const char* filename, const unsigned char* img, int nrow, int ncol);
double get_time(void);

#endif // SKELETONGRAD_H