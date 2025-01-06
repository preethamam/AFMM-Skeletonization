'''
Python Implementation of Skeletonization
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

This implementation of a skeletonization method is the MATLAB/MEX C++ code 
translation and optimization of the original C++ code written by Nicholas Howe. 
Weblink: https://www.mathworks.com/matlabcentral/fileexchange/11123-better-skeletonization

This implementation is JIT optimized and reliable. We believe it can be further 
improved by minimizing the for-loops. 

Alex (article author) has a faster C/C++ (non-Matlab) implementation. It can be found at:

1. https://webspace.science.uu.nl/~telea001/uploads/Software/AFMM/
2. https://webspace.science.uu.nl/~telea001/Software/Software.

Execute demo.py


Copyright Notice
----------------
This software comes with no warranty, expressed or implied, including
but not limited to merchantability or fitness for any particular
purpose.

All files are copyright Dr. Preetham Manjunatha.  
Permission is granted to use the material for noncommercial and 
research purposes.
'''
import sys

sys.dont_write_bytecode = True
from enum import Enum

from numba import jit, prange
import numpy as np
from joblib import Parallel, delayed
import time
import numba as nb

class Direction(Enum):
    North = 0
    South = 1
    East = 2
    West = 3
    None_ = 4

# Direction lookup table converted from C++ array
DIRCODE = [
    Direction.None_, Direction.West, Direction.North, Direction.West,
    Direction.East, Direction.None_, Direction.North, Direction.West,
    Direction.South, Direction.South, Direction.None_, Direction.South,
    Direction.East, Direction.East, Direction.North, Direction.None_
]

def joint_neighborhood(arr, i, j, nrow, ncol):
    """Calculate the joint neighborhood of a point."""
    # Convert to Fortran-style (column-major) indexing
    p = i + j * nrow
    condition = 8*(i <= 0) + 4*(j <= 0) + 2*(i >= nrow) + (j >= ncol)
    
    if condition == 0:  # all points valid
        return ((1 if arr[p-nrow-1] else 0) + 
                (2 if arr[p-1] else 0) + 
                (4 if arr[p] else 0) + 
                (8 if arr[p-nrow] else 0))
    elif condition == 1:  # right side not valid
        return ((1 if arr[p-nrow-1] else 0) + 
                (8 if arr[p-nrow] else 0))
    elif condition == 2:  # bottom not valid
        return ((1 if arr[p-nrow-1] else 0) + 
                (2 if arr[p-1] else 0))
    elif condition == 3:  # bottom and right not valid
        return (1 if arr[p-nrow-1] else 0)
    elif condition == 4:  # left side not valid
        return ((2 if arr[p-1] else 0) + 
                (4 if arr[p] else 0))
    elif condition in [5, 6, 7, 10, 11, 13, 14, 15]:  # various invalid combinations
        return 0
    elif condition == 8:  # top side not valid
        return ((4 if arr[p] else 0) + 
                (8 if arr[p-nrow] else 0))
    elif condition == 9:  # top and right not valid
        return (8 if arr[p-nrow] else 0)
    elif condition == 12:  # top and left not valid
        return (4 if arr[p] else 0)
    else:
        raise ValueError("Invalid condition in joint_neighborhood")

def compute_skeleton_gradient(img):
    """
    Compute the skeleton gradient transform and skeleton radius.
    Optimized version that computes distances once and minimizes batching.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=bool, order='F')
    elif not img.flags.f_contiguous:
        img = np.asfortranarray(img)
    
    nrow, ncol = img.shape
    jnrow, jncol = nrow + 1, ncol + 1
    flat_img = img.ravel('F')
    
    # Count junctions (keeping this loop to maintain exact junction counting)
    njunc = 0
    for j in range(jncol):
        for i in range(jnrow):
            jhood = joint_neighborhood(flat_img, i, j, nrow, ncol)
            if (jhood != 0) and (jhood != 15):
                njunc += 1
    
    # Initialize arrays
    jx = np.zeros(njunc, dtype=np.int32)
    jy = np.zeros(njunc, dtype=np.int32)
    seqj = np.zeros(njunc, dtype=np.int32)
    edgej = np.zeros(njunc, dtype=np.int32)
    seenj = np.zeros(jnrow * jncol, dtype=bool)
    
    # Initialize output arrays
    skg = np.zeros((nrow, ncol), order='F')
    rad = np.zeros((nrow, ncol), order='F')
    
    # Process edges (keeping this single loop for essential edge tracking)
    nedge = 0
    ijunc = 0
    
    first_loop = time.time()
    for j in range(jncol):
        for i in range(jnrow):
            jhood = joint_neighborhood(flat_img, i, j, nrow, ncol)
            if ((jhood != 0) and (jhood != 15) and (jhood != 5) and 
                (jhood != 10) and not seenj[i + j*jnrow]):
                # Edge traversal
                iseq = 0
                ei, ej = i, j
                lastdir = Direction.North
                
                while not seenj[ei + ej*jnrow] or (jhood == 5) or (jhood == 10):
                    if not seenj[ei + ej*jnrow]:
                        jx[ijunc], jy[ijunc] = ej, ei
                        edgej[ijunc] = nedge
                        seqj[ijunc] = iseq
                        iseq += 1
                        ijunc += 1
                        seenj[ei + ej*jnrow] = True
                    
                    dir_code = DIRCODE[jhood]
                    if dir_code in [Direction.North, Direction.South, Direction.East, Direction.West]:
                        ei += (dir_code == Direction.South) - (dir_code == Direction.North)
                        ej += (dir_code == Direction.East) - (dir_code == Direction.West)
                        lastdir = dir_code
                    else:
                        ei += (lastdir == Direction.West) - (lastdir == Direction.East)
                        ej += (lastdir == Direction.South) - (lastdir == Direction.North)
                        lastdir = {Direction.East: Direction.North,
                                 Direction.West: Direction.South,
                                 Direction.South: Direction.East,
                                 Direction.North: Direction.West}[lastdir]
                    
                    if not (0 <= ei < jnrow and 0 <= ej < jncol):
                        raise ValueError("Traversed out of bounds")
                    
                    jhood = joint_neighborhood(flat_img, ei, ej, nrow, ncol)
                
                nedge += 1
    print("First loop time: ", time.time() - first_loop)
    
    # Compute edge lengths (vectorized)
    edgelen = np.bincount(edgej)
    
    # Process points
    if njunc > 0:
        true_points = np.argwhere(img)
        if len(true_points) > 0:
            BATCH_SIZE = 20000  # Increased batch size for better vectorization
            
            second_loop = time.time()
            
            for batch_start in range(0, len(true_points), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(true_points))
                batch_points = true_points[batch_start:batch_end]
                
                # Compute all distances for this batch at once
                i_coords = batch_points[:, 0][:, None]
                j_coords = batch_points[:, 1][:, None]
                
                # Compute all four distance matrices at once
                dNE = (i_coords - jy)**2 + (j_coords - jx)**2
                dNW = (i_coords - jy)**2 + (j_coords + 1 - jx)**2
                dSE = (i_coords + 1 - jy)**2 + (j_coords - jx)**2
                dSW = (i_coords + 1 - jy)**2 + (j_coords + 1 - jx)**2
                
                # Find minimum distances efficiently
                min_dists = np.minimum.reduce([
                    np.min(dNE, axis=1),
                    np.min(dNW, axis=1),
                    np.min(dSE, axis=1),
                    np.min(dSW, axis=1)
                ])
                
                min_juncs = np.argmin(dNE, axis=1)  # Using NE for consistency
                
                # Store radius information
                rad[batch_points[:, 0], batch_points[:, 1]] = min_dists
                
                # Process skeleton gradient for the batch
                third_loop = time.time()
                for idx in range(len(batch_points)):
                    i, j = batch_points[idx]
                    minjunc = min_juncs[idx]
                    
                    mindNE = np.min(dNE[idx])
                    mindNW = np.min(dNW[idx])
                    mindSE = np.min(dSE[idx])
                    mindSW = np.min(dSW[idx])
                    
                    # Use pre-computed distances for near points detection
                    near_mask = ((dNE[idx] <= min(mindNE, dNE[idx, minjunc])) |
                               (dNW[idx] <= min(mindNW, dNW[idx, minjunc])) |
                               (dSE[idx] <= min(mindSE, dSE[idx, minjunc])) |
                               (dSW[idx] <= min(mindSW, dSW[idx, minjunc])))
                    
                    near_points = np.where(near_mask)[0]
                    
                    if len(near_points) > 0:
                        if not np.all(edgej[near_points] == edgej[minjunc]):
                            skg[i, j] = np.inf
                        else:
                            seq_points = np.sort(seqj[near_points])
                            if len(seq_points) > 1:
                                pspan = seq_points[0] - seq_points[-1] + edgelen[edgej[minjunc]]
                                diffs = np.diff(seq_points)
                                if len(diffs) > 0:
                                    max_diff = np.max(diffs)
                                    pspan = max(pspan, max_diff)
                                skg[i, j] = edgelen[edgej[minjunc]] - pspan
                            else:
                                skg[i, j] = 0
                    else:
                        skg[i, j] = np.inf
                    
                print("Third loop time: ", time.time() - third_loop)
            print("Second loop time: ", time.time() - second_loop)
    
    return skg, rad