# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, rand
from libcpp cimport bool
from cython.parallel cimport prange

# Constants and enums
cdef enum direction:
    North = 0
    South = 1
    East = 2
    West = 3
    None = 4

# Dircode lookup table
cdef int[16] dircode
dircode = [
    4, 3, 0, 3, 2, 4, 0, 3,
    1, 1, 4, 1, 2, 2, 0, 4
]

# Utility macros as inline functions
cdef inline int SQR(int x):
    return x * x

cdef inline int MIN(int x, int y):
    return x if x < y else y

cdef inline int MAX(int x, int y):
    return x if x > y else y

cdef inline int ABS(int x):
    return -x if x < 0 else x

cdef inline int MOD(int x, int n):
    return (x % n + n) if (x % n < 0) else (x % n)

# Use Python's sort functionality for C array
cdef void c_array_sort(int *arr, int n) nogil:
    # Create a Python list to hold the values
    with gil:
        temp = sorted([arr[i] for i in range(n)])
        for i in range(n):
            arr[i] = temp[i]

# Joint neighborhood function
cdef int joint_neighborhood(const unsigned char* arr, int i, int j, int nrow, int ncol) nogil:
    cdef int p = i + j * nrow
    cdef int condition = 8*(i <= 0) + 4*(j <= 0) + 2*(i >= nrow) + (j >= ncol)

    if condition == 0:  # all points valid
        return (arr[p-nrow-1]!=0)*1 + (arr[p-1]!=0)*2 + (arr[p]!=0)*4 + (arr[p-nrow]!=0)*8
    elif condition == 1:  # right side not valid
        return (arr[p-nrow-1]!=0)*1 + (arr[p-nrow]!=0)*8
    elif condition == 2:  # bottom not valid
        return (arr[p-nrow-1]!=0)*1 + (arr[p-1]!=0)*2
    elif condition == 3:  # bottom and right not valid
        return (arr[p-nrow-1]!=0)*1
    elif condition == 4:  # left side not valid
        return (arr[p-1]!=0)*2 + (arr[p]!=0)*4
    elif condition == 5:  # left and right sides not valid
        return 0
    elif condition == 6:  # left and bottom sides not valid
        return (arr[p-1]!=0)*2
    elif condition == 7:  # left, bottom, and right sides not valid
        return 0
    elif condition == 8:  # top side not valid
        return (arr[p]!=0)*4 + (arr[p-nrow]!=0)*8
    elif condition == 9:  # top and right not valid
        return (arr[p-nrow]!=0)*8
    elif condition == 10:  # top and bottom not valid
        return 0
    elif condition == 11:  # top, bottom and right not valid
        return 0
    elif condition == 12:  # top and left not valid
        return (arr[p]!=0)*4
    else:  # remaining cases
        return 0


# Main computation function
def compute_skeleton_gradient(np.ndarray[np.uint8_t, ndim=2] img not None):
    # Ensure input image is in Fortran order (column-major) to match MATLAB
    img = np.asfortranarray(img)
    cdef int nrow = img.shape[0]
    cdef int ncol = img.shape[1]
    cdef int jnrow = nrow + 1
    cdef int jncol = ncol + 1
    cdef int njunc = 0
    cdef int i, j, ei, ej, ijunc, iedge, iseq, lastdir, mind, minjunc
    cdef int mindNE, mindNW, mindSE, mindSW, nnear, pspan
    cdef int jhood, nedge
    cdef const unsigned char* img_ptr = <const unsigned char*>img.data
    
    # Count junctions
    for j in range(jncol):
        for i in range(jnrow):
            jhood = joint_neighborhood(img_ptr, i, j, nrow, ncol)
            if jhood != 0 and jhood != 15:
                njunc += 1

    # Allocate arrays
    cdef int *jx = <int*>malloc(njunc * sizeof(int))
    cdef int *jy = <int*>malloc(njunc * sizeof(int))
    cdef int *seqj = <int*>malloc(njunc * sizeof(int))
    cdef int *edgej = <int*>malloc(njunc * sizeof(int))
    cdef bool *seenj = <bool*>malloc(jnrow * jncol * sizeof(bool))
    cdef int *dNE = <int*>malloc(njunc * sizeof(int))
    cdef int *dNW = <int*>malloc(njunc * sizeof(int))
    cdef int *dSE = <int*>malloc(njunc * sizeof(int))
    cdef int *dSW = <int*>malloc(njunc * sizeof(int))
    cdef int *nearj = <int*>malloc(njunc * sizeof(int))

    if not (jx and jy and seqj and edgej and seenj and dNE and dNW and 
            dSE and dSW and nearj):
        raise MemoryError()

    for i in range(jnrow * jncol):
        seenj[i] = False

    # Register junctions
    ijunc = 0
    nedge = 0
    for j in range(jncol):
        for i in range(jnrow):
            jhood = joint_neighborhood(img_ptr, i, j, nrow, ncol)
            if (jhood != 0 and jhood != 15 and jhood != 5 and jhood != 10 
                and not seenj[i + j*jnrow]):
                # Found new edge; traverse it
                iseq = 0
                ei = i
                ej = j
                lastdir = North
                
                while not seenj[ei + ej*jnrow] or jhood == 5 or jhood == 10:
                    if not seenj[ei + ej*jnrow]:
                        jx[ijunc] = ej
                        jy[ijunc] = ei
                        edgej[ijunc] = nedge
                        seqj[ijunc] = iseq
                        iseq += 1
                        ijunc += 1
                        seenj[ei + ej*jnrow] = True

                    # Traverse clockwise based on direction code
                    if dircode[jhood] == North:
                        ei -= 1
                        lastdir = North
                    elif dircode[jhood] == South:
                        ei += 1
                        lastdir = South
                    elif dircode[jhood] == East:
                        ej += 1
                        lastdir = East
                    elif dircode[jhood] == West:
                        ej -= 1
                        lastdir = West
                    else:  # None
                        if lastdir == East:
                            ei -= 1
                            lastdir = North
                        elif lastdir == West:
                            ei += 1
                            lastdir = South
                        elif lastdir == South:
                            ej += 1
                            lastdir = East
                        elif lastdir == North:
                            ej -= 1
                            lastdir = West

                    if not (0 <= ei < jnrow and 0 <= ej < jncol):
                        break

                    jhood = joint_neighborhood(img_ptr, ei, ej, nrow, ncol)
                
                nedge += 1

    # Create output arrays - note we transpose the shape to match MATLAB's column-major order
    cdef np.ndarray[np.float64_t, ndim=2] skg = np.zeros((ncol, nrow), dtype=np.float64).T
    cdef np.ndarray[np.float64_t, ndim=2] rad = np.zeros((ncol, nrow), dtype=np.float64).T

    # Count perimeter along each edge
    cdef int *edgelen = <int*>malloc(nedge * sizeof(int))
    for iedge in range(nedge):
        edgelen[iedge] = 0
    for ijunc in range(njunc):
        edgelen[edgej[ijunc]] += 1

    # Main computation for each pixel
    for j in range(ncol):
        for i in range(nrow):
            if img[i, j] != 0:  # Match MATLAB's binary image handling
                mind = mindNE = mindNW = mindSE = mindSW = SQR(jnrow + jncol)
                minjunc = -1

                # Compute distances to all junction points
                for ijunc in range(njunc):
                    dNE[ijunc] = SQR(i - jy[ijunc]) + SQR(j - jx[ijunc])
                    dNW[ijunc] = SQR(i - jy[ijunc]) + SQR(j + 1 - jx[ijunc])
                    dSE[ijunc] = SQR(i + 1 - jy[ijunc]) + SQR(j - jx[ijunc])
                    dSW[ijunc] = SQR(i + 1 - jy[ijunc]) + SQR(j + 1 - jx[ijunc])

                    if dNE[ijunc] < mindNE:
                        mindNE = dNE[ijunc]
                        if dNE[ijunc] < mind:
                            mind = dNE[ijunc]
                            minjunc = ijunc

                    if dNW[ijunc] < mindNW:
                        mindNW = dNW[ijunc]
                        if dNW[ijunc] < mind:
                            mind = dNW[ijunc]
                            minjunc = ijunc

                    if dSE[ijunc] < mindSE:
                        mindSE = dSE[ijunc]
                        if dSE[ijunc] < mind:
                            mind = dSE[ijunc]
                            minjunc = ijunc

                    if dSW[ijunc] < mindSW:
                        mindSW = dSW[ijunc]
                        if dSW[ijunc] < mind:
                            mind = dSW[ijunc]
                            minjunc = ijunc

                rad[i, j] = mind

                # Find all other junction points at minimal distance
                nnear = pspan = 0
                for ijunc in range(njunc):
                    if ((dNE[ijunc] <= MIN(mindNE, dNE[minjunc])) or
                        (dNW[ijunc] <= MIN(mindNW, dNW[minjunc])) or
                        (dSE[ijunc] <= MIN(mindSE, dSE[minjunc])) or
                        (dSW[ijunc] <= MIN(mindSW, dSW[minjunc]))):
                        
                        if edgej[ijunc] != edgej[minjunc]:
                            pspan = -1
                            break
                        else:
                            nearj[nnear] = seqj[ijunc]
                            nnear += 1

                if pspan >= 0:
                    # Compute perimeter span
                    c_array_sort(nearj, nnear)
                    pspan = nearj[0] - nearj[nnear-1] + edgelen[edgej[minjunc]]
                    
                    for inear in range(1, nnear):
                        if pspan < nearj[inear] - nearj[inear-1]:
                            pspan = nearj[inear] - nearj[inear-1]
                    
                    pspan = edgelen[edgej[minjunc]] - pspan
                    skg[i, j] = pspan
                else:
                    skg[i, j] = np.inf

    # Free allocated memory
    free(jx)
    free(jy)
    free(seqj)
    free(edgej)
    free(seenj)
    free(dNE)
    free(dNW)
    free(dSE)
    free(dSW)
    free(nearj)
    free(edgelen)

    return skg, rad