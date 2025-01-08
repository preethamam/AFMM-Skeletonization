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
translation and optimization of the original MATLAB/MEX C code written by Nicholas Howe. . 
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

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "stb_image.h"
#include "stb_image_write.h"
#include "skeletongrad.h"

#define SQR(x) (x)*(x)
#define MIN(x,y) (((x) < (y)) ? (x):(y))
#define MAX(x,y) (((x) > (y)) ? (x):(y))
#define ABS(x) (((x) < 0) ? (-(x)):(x))
#define MOD(x,n) (((x)%(n)<0) ? ((x)%(n)+(n)):((x)%(n)))

// Direction lookup table
static const direction dircode[16] = {
    None, West, North, West, East, None, North, West,
    South, South, None, South, East, East, North, None
};

// Quick sort implementation
void quicksort(int *arr, int n) {
    if (n > 8) {
        int pivot;
        int pivid;
        int lo = 1;
        int hi = n-1;
        int tmp;

        pivid = rand() % n;
        tmp = arr[0];
        arr[0] = arr[pivid];
        arr[pivid] = tmp;
        pivot = arr[0];

        while (hi != lo-1) {
            if (arr[lo] < pivot) {
                lo++;
            } else {
                tmp = arr[hi];
                arr[hi] = arr[lo];
                arr[lo] = tmp;
                hi--;
            }
        }
        tmp = arr[hi];
        arr[hi] = arr[0];
        arr[0] = tmp;
        quicksort(arr, hi);
        quicksort(arr+lo, n-lo);
    } else {
        int i, j;
        int tmp;

        for (i = 0; i < n; i++) {
            for (j = i; (j > 0) && (arr[j] < arr[j-1]); j--) {
                tmp = arr[j];
                arr[j] = arr[j-1];
                arr[j-1] = tmp;
            }
        }
    }
}

// Joint neighborhood function
int joint_neighborhood(const unsigned char *arr, int i, int j, int nrow, int ncol) {
    int p = i + j*nrow;
    int condition = 8*(i <= 0) + 4*(j <= 0) + 2*(i >= nrow) + (j >= ncol);

    switch (condition) {
        case 0:  // all points valid
            return (arr[p-nrow-1]?1:0) + (arr[p-1]?2:0) + 
                   (arr[p]?4:0) + (arr[p-nrow]?8:0);
        case 1:  // right side not valid
            return (arr[p-nrow-1]?1:0) + (arr[p-nrow]?8:0);
        case 2:  // bottom not valid
            return (arr[p-nrow-1]?1:0) + (arr[p-1]?2:0);
        case 3:  // bottom and right not valid
            return (arr[p-nrow-1]?1:0);
        case 4:  // left side not valid
            return (arr[p-1]?2:0) + (arr[p]?4:0);
        case 5:  // left and right sides not valid
            return 0;
        case 6:  // left and bottom sides not valid
            return (arr[p-1]?2:0);
        case 7:  // left, bottom, and right sides not valid
            return 0;
        case 8:  // top side not valid
            return (arr[p]?4:0) + (arr[p-nrow]?8:0);
        case 9:  // top and right not valid
            return (arr[p-nrow]?8:0);
        case 10:  // top and bottom not valid
        case 11:  // top, bottom and right not valid
            return 0;
        case 12:  // top and left not valid
            return (arr[p]?4:0);
        case 13:  // top, left and right sides not valid
        case 14:  // top, left and bottom sides not valid
        case 15:  // no sides valid
            return 0;
        default:
            fprintf(stderr, "Impossible condition.\n");
            return -1;
    }
}

// Skeleton gradient computation function
void compute_skeleton_gradient(const unsigned char *img, int nrow, int ncol,
                             double *skg, double *rad) {
    int i, j, ei, ej, inear;
    int ijunc, iedge, iseq, lastdir, mind, minjunc, pspan;
    int jnrow = nrow+1, jncol = ncol+1;
    int njunc = 0, jhood, nedge = 0, nnear;
    int mindNE, mindNW, mindSE, mindSW;
    int *jx, *jy, *edgej, *seqj, *edgelen = NULL;
    int *dNE, *dNW, *dSE, *dSW, *nearj;
    bool *seenj;

    // Count junctions
    for (j = 0; j < jncol; j++) {
        for (i = 0; i < jnrow; i++) {
            jhood = joint_neighborhood(img, i, j, nrow, ncol);
            if ((jhood != 0) && (jhood != 15)) {
                njunc++;
            }
        }
    }

    // Allocate memory
    jx = (int*)malloc(njunc * sizeof(int));
    jy = (int*)malloc(njunc * sizeof(int));
    seqj = (int*)malloc(njunc * sizeof(int));
    edgej = (int*)malloc(njunc * sizeof(int));
    seenj = (bool*)calloc(jnrow * jncol, sizeof(bool));
    dNE = (int*)malloc(njunc * sizeof(int));
    dNW = (int*)malloc(njunc * sizeof(int));
    dSE = (int*)malloc(njunc * sizeof(int));
    dSW = (int*)malloc(njunc * sizeof(int));
    nearj = (int*)malloc(njunc * sizeof(int));

    if (!jx || !jy || !seqj || !edgej || !seenj || !dNE || !dNW || 
        !dSE || !dSW || !nearj) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }

    for (i = 0; i < jnrow*jncol; i++) {
        seenj[i] = false;
    }

    // Register junctions
    ijunc = 0;
    for (j = 0; j < jncol; j++) {
        for (i = 0; i < jnrow; i++) {
            jhood = joint_neighborhood(img, i, j, nrow, ncol);
            if ((jhood != 0) && (jhood != 15) && (jhood != 5) && (jhood != 10)
                && !seenj[i + j*jnrow]) {
                iseq = 0;
                ei = i;
                ej = j;
                lastdir = North;

                while (!seenj[ei + ej*jnrow] || (jhood == 5) || (jhood == 10)) {
                    if (!seenj[ei + ej*jnrow]) {
                        jx[ijunc] = ej;
                        jy[ijunc] = ei;
                        edgej[ijunc] = nedge;
                        seqj[ijunc] = iseq;
                        iseq++;
                        ijunc++;
                        seenj[ei + ej*jnrow] = true;
                    }

                    switch (dircode[jhood]) {
                        case North:
                            ei--;
                            lastdir = North;
                            break;
                        case South:
                            ei++;
                            lastdir = South;
                            break;
                        case East:
                            ej++;
                            lastdir = East;
                            break;
                        case West:
                            ej--;
                            lastdir = West;
                            break;
                        case None:
                            switch (lastdir) {
                                case East:
                                    ei--;
                                    lastdir = North;
                                    break;
                                case West:
                                    ei++;
                                    lastdir = South;
                                    break;
                                case South:
                                    ej++;
                                    lastdir = East;
                                    break;
                                case North:
                                    ej--;
                                    lastdir = West;
                                    break;
                            }
                            break;
                    }
                    
                    if (ei < 0 || ej < 0 || ei >= jnrow || ej >= jncol) {
                        fprintf(stderr, "Traversed out of bounds\n");
                        goto cleanup;
                    }
                    
                    jhood = joint_neighborhood(img, ei, ej, nrow, ncol);
                }
                nedge++;
            }
        }
    }

    // Count perimeter along each edge
    edgelen = (int*)malloc(nedge * sizeof(int));
    if (!edgelen) {
        fprintf(stderr, "Memory allocation failed\n");
        goto cleanup;
    }

    // Explicit initialization as in original MEX code
    for (iedge = 0; iedge < nedge; iedge++) {
        edgelen[iedge] = 0;
    }

    // Count occurrences for each edge
    for (ijunc = 0; ijunc < njunc; ijunc++) {
        edgelen[edgej[ijunc]]++;
    }

    // Compute skeleton gradient
    for (j = 0; j < ncol; j++) {
        for (i = 0; i < nrow; i++) {
            if (img[i + j*nrow]) {
                mind = mindNE = mindNW = mindSE = mindSW = SQR(jnrow + jncol);
                minjunc = -1;

                for (ijunc = 0; ijunc < njunc; ijunc++) {
                    dNE[ijunc] = SQR(i-jy[ijunc]) + SQR(j-jx[ijunc]);
                    dNW[ijunc] = SQR(i-jy[ijunc]) + SQR(j+1-jx[ijunc]);
                    dSE[ijunc] = SQR(i+1-jy[ijunc]) + SQR(j-jx[ijunc]);
                    dSW[ijunc] = SQR(i+1-jy[ijunc]) + SQR(j+1-jx[ijunc]);

                    if (dNE[ijunc] < mindNE) {
                        mindNE = dNE[ijunc];
                        if (dNE[ijunc] < mind) {
                            mind = dNE[ijunc];
                            minjunc = ijunc;
                        }
                    }
                    if (dNW[ijunc] < mindNW) {
                        mindNW = dNW[ijunc];
                        if (dNW[ijunc] < mind) {
                            mind = dNW[ijunc];
                            minjunc = ijunc;
                        }
                    }
                    if (dSE[ijunc] < mindSE) {
                        mindSE = dSE[ijunc];
                        if (dSE[ijunc] < mind) {
                            mind = dSE[ijunc];
                            minjunc = ijunc;
                        }
                    }
                    if (dSW[ijunc] < mindSW) {
                        mindSW = dSW[ijunc];
                        if (dSW[ijunc] < mind) {
                            mind = dSW[ijunc];
                            minjunc = ijunc;
                        }
                    }
                }

                if (minjunc >= 0) {
                    if (rad != NULL) {
                        rad[i + j*nrow] = mind;
                    }

                    nnear = pspan = 0;
                    for (ijunc = 0; ijunc < njunc; ijunc++) {
                        if ((dNE[ijunc] <= MIN(mindNE, dNE[minjunc])) ||
                            (dNW[ijunc] <= MIN(mindNW, dNW[minjunc])) ||
                            (dSE[ijunc] <= MIN(mindSE, dSE[minjunc])) ||
                            (dSW[ijunc] <= MIN(mindSW, dSW[minjunc]))) {
                            if (edgej[ijunc] != edgej[minjunc]) {
                                pspan = -1;
                                break;
                            } else {
                                nearj[nnear] = seqj[ijunc];
                                nnear++;
                            }
                        }
                    }

                    if (pspan >= 0) {
                        quicksort(nearj, nnear);
                        pspan = nearj[0] - nearj[nnear-1] + edgelen[edgej[minjunc]];
                        for (inear = 1; inear < nnear; inear++) {
                            if (pspan < nearj[inear] - nearj[inear-1]) {
                                pspan = nearj[inear] - nearj[inear-1];
                            }
                        }
                        pspan = edgelen[edgej[minjunc]] - pspan;
                        skg[i + j*nrow] = pspan;
                    } else {
                        skg[i + j*nrow] = INFINITY;
                    }
                } else {
                    skg[i + j*nrow] = 0;
                    if (rad != NULL) {
                        rad[i + j*nrow] = 0;
                    }
                }
            } else {
                skg[i + j*nrow] = 0;
                if (rad != NULL) {
                    rad[i + j*nrow] = 0;
                }
            }
        }
    }

// Cleanup
cleanup:
    free(jx);
    free(jy);
    free(seqj);
    free(edgej);
    free(seenj);
    free(dNE);
    free(dNW);
    free(dSE);
    free(dSW);
    free(nearj);
    if (edgelen) {  // Only free if not NULL
        free(edgelen);
    }
}

// Read image function
unsigned char* read_image(const char* filename, int* nrow, int* ncol) {
    int channels;
    // stb_image uses width,height order (ncol,nrow)
    unsigned char* img_data = stbi_load(filename, ncol, nrow, &channels, 1);
    
    // Need to transpose the data to match MATLAB's column-major order
    unsigned char* binary_img = (unsigned char*)malloc((*nrow) * (*ncol));
    
    for (int j = 0; j < *ncol; j++) {
        for (int i = 0; i < *nrow; i++) {
            // Convert row-major to column-major during thresholding
            binary_img[i + j*(*nrow)] = (img_data[i*(*ncol) + j] > 128) ? 1 : 0;
        }
    }

    stbi_image_free(img_data);
    return binary_img;
}

// Write image to disk
int write_image(const char* filename, const unsigned char* img, int nrow, int ncol) {
    unsigned char* output_data = (unsigned char*)malloc(nrow * ncol);
    
    // Convert from column-major back to row-major
    for (int j = 0; j < ncol; j++) {
        for (int i = 0; i < nrow; i++) {
            output_data[i*ncol + j] = img[i + j*nrow] ? 255 : 0;
        }
    }

    int success = stbi_write_png(filename, ncol, nrow, 1, output_data, ncol);
    free(output_data);
    return success;
}

// Helper function to get current time
double get_time() {
    return (double)clock() / CLOCKS_PER_SEC;
}