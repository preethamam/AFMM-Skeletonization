import numpy as np
cimport numpy as np
from libc.math cimport INFINITY
cimport cython
from cython.parallel import parallel, prange
import time
from numba import jit

# Direction enum (matching original implementation)
cdef enum Direction:
    North = 0
    South = 1
    East = 2
    West = 3
    None_ = 4

# Direction lookup table (matching original implementation)
cdef unsigned char[16] DIRCODE = [
    Direction.None_, Direction.West, Direction.North, Direction.West,
    Direction.East, Direction.None_, Direction.North, Direction.West,
    Direction.South, Direction.South, Direction.None_, Direction.South,
    Direction.East, Direction.East, Direction.North, Direction.None_
]

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline int joint_neighborhood(const unsigned char[:] arr, 
                          Py_ssize_t i, 
                          Py_ssize_t j, 
                          Py_ssize_t nrow, 
                          Py_ssize_t ncol) nogil:
    """Calculate the joint neighborhood of a point."""
    cdef Py_ssize_t p = i + j * nrow
    cdef int condition = (8 * (i <= 0) + 
                         4 * (j <= 0) + 
                         2 * (i >= nrow) + 
                         (j >= ncol))
    
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
    elif condition == 8:  # top side not valid
        return ((4 if arr[p] else 0) + 
                (8 if arr[p-nrow] else 0))
    elif condition == 9:  # top and right not valid
        return (8 if arr[p-nrow] else 0)
    elif condition == 12:  # top and left not valid
        return (4 if arr[p] else 0)
    else:
        return 0

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def compute_skeleton_gradient(object img):
    """
    Compute the skeleton gradient transform and skeleton radius.
    """
    # Ensure input is a Fortran-ordered boolean array
    cdef np.ndarray[np.uint8_t, ndim=2, mode='fortran'] img_array
    if not isinstance(img, np.ndarray):
        img_array = np.asfortranarray(np.array(img, dtype=bool))
    else:
        img_array = np.asfortranarray(img.astype(bool))
    
    cdef Py_ssize_t nrow = img_array.shape[0]
    cdef Py_ssize_t ncol = img_array.shape[1]
    cdef Py_ssize_t jnrow = nrow + 1
    cdef Py_ssize_t jncol = ncol + 1
    
    # Flat view for efficient indexing
    cdef unsigned char[:] flat_img = img_array.ravel('F')
    
    # Count junctions
    cdef Py_ssize_t njunc = 0, i, j, nedge = 0, ijunc = 0, iseq, ei, ej
    cdef int jhood
    cdef unsigned char lastdir, dir_code
    cdef Py_ssize_t temp_njunc
    cdef Py_ssize_t batch_size = 2000
    cdef Py_ssize_t batch_start, batch_end, point_idx
    
    # Count junctions
    if img.shape[0]*img.shape[1] <= 1E4:
        for j in range(jncol):
            for i in range(jnrow):
                jhood = joint_neighborhood(flat_img, i, j, nrow, ncol)
                if (jhood != 0) and (jhood != 15):
                    njunc += 1
    else:
        for j in prange(jncol, nogil=True):
            for i in range(jnrow):
                jhood = joint_neighborhood(flat_img, i, j, nrow, ncol)
                if (jhood != 0) and (jhood != 15):
                    temp_njunc = 1
                    njunc += temp_njunc
    
    # Initialize arrays (keeping same array types)
    cdef np.ndarray[np.int64_t, ndim=1] jx = np.zeros(njunc, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] jy = np.zeros(njunc, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] seqj = np.zeros(njunc, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] edgej = np.zeros(njunc, dtype=np.int64)
    cdef np.ndarray[np.uint8_t, ndim=1] seenj = np.zeros(jnrow * jncol, dtype=bool)
    
    # Initialize output arrays
    cdef np.ndarray[double, ndim=2, mode='fortran'] skg = np.zeros((nrow, ncol), dtype=float, order='F')
    cdef np.ndarray[double, ndim=2, mode='fortran'] rad = np.zeros((nrow, ncol), dtype=float, order='F')
    
    # Process edges
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
                        jx[ijunc] = ej
                        jy[ijunc] = ei
                        edgej[ijunc] = nedge
                        seqj[ijunc] = iseq
                        iseq += 1
                        ijunc += 1
                        seenj[ei + ej*jnrow] = True
                    
                    dir_code = DIRCODE[jhood]
                    if dir_code < Direction.None_:
                        ei += (dir_code == Direction.South) - (dir_code == Direction.North)
                        ej += (dir_code == Direction.East) - (dir_code == Direction.West)
                        lastdir = dir_code
                    else:
                        ei += (lastdir == Direction.West) - (lastdir == Direction.East)
                        ej += (lastdir == Direction.South) - (lastdir == Direction.North)
                        if lastdir == Direction.East:
                            lastdir = Direction.North
                        elif lastdir == Direction.West:
                            lastdir = Direction.South
                        elif lastdir == Direction.South:
                            lastdir = Direction.East
                        else:
                            lastdir = Direction.West
                    
                    if not (0 <= ei < jnrow and 0 <= ej < jncol):
                        raise ValueError("Traversed out of bounds")
                    
                    jhood = joint_neighborhood(flat_img, ei, ej, nrow, ncol)
                
                nedge += 1
    print("First loop time: ", time.time() - first_loop)

    # Process points
    if njunc > 0:
        # Find true points
        true_points = np.argwhere(img_array)
        
        if true_points.shape[0] > 0:

            second_loop = time.time()
            for batch_start in range(0, true_points.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, true_points.shape[0])
                batch_points = true_points[batch_start:batch_end]
                
                # Compute all distances for this batch
                dNE = ((batch_points[:, 0][:, None] - jy)**2 + 
                       (batch_points[:, 1][:, None] - jx)**2)
                dNW = ((batch_points[:, 0][:, None] - jy)**2 + 
                       (batch_points[:, 1][:, None] + 1 - jx)**2)
                dSE = ((batch_points[:, 0][:, None] + 1 - jy)**2 + 
                       (batch_points[:, 1][:, None] - jx)**2)
                dSW = ((batch_points[:, 0][:, None] + 1 - jy)**2 + 
                       (batch_points[:, 1][:, None] + 1 - jx)**2)
                
                # Find minimum distances
                min_dists = np.minimum.reduce([
                    np.min(dNE, axis=1),
                    np.min(dNW, axis=1),
                    np.min(dSE, axis=1),
                    np.min(dSW, axis=1)
                ])
                
                # Get indices of minimum distances
                min_juncs = np.argmin(dNE, axis=1)
                
                # Store radius information
                third_loop = time.time()
                for point_idx in range(batch_points.shape[0]):
                    i, j = batch_points[point_idx]
                    rad[i, j] = min_dists[point_idx]
                
                print("Third loop time: ", time.time() - third_loop)

                # Process skeleton gradient for the batch
                fourth_loop = time.time()
                
                for point_idx in range(batch_points.shape[0]):
                    i, j = batch_points[point_idx]
                    minjunc = min_juncs[point_idx]
                    
                    mindNE = np.min(dNE[point_idx])
                    mindNW = np.min(dNW[point_idx])
                    mindSE = np.min(dSE[point_idx])
                    mindSW = np.min(dSW[point_idx])
                    
                    # Detect near points
                    near_mask = (
                        (dNE[point_idx] <= min(mindNE, dNE[point_idx, minjunc])) |
                        (dNW[point_idx] <= min(mindNW, dNW[point_idx, minjunc])) |
                        (dSE[point_idx] <= min(mindSE, dSE[point_idx, minjunc])) |
                        (dSW[point_idx] <= min(mindSW, dSW[point_idx, minjunc]))
                    )
                    
                    near_points = np.where(near_mask)[0]
                    
                    if near_points.shape[0] > 0:
                        if not np.all(edgej[near_points] == edgej[minjunc]):
                            skg[i, j] = np.inf
                        else:
                            seq_points = np.sort(seqj[near_points])
                            edgelen = np.bincount(edgej).astype(np.int64)
                            
                            if seq_points.shape[0] > 1:
                                pspan = seq_points[0] - seq_points[-1] + edgelen[edgej[minjunc]]
                                diffs = np.diff(seq_points)
                                
                                if diffs.shape[0] > 0:
                                    max_diff = np.max(diffs)
                                    pspan = max(pspan, max_diff)
                                
                                skg[i, j] = edgelen[edgej[minjunc]] - pspan
                            else:
                                skg[i, j] = 0
                    else:
                        skg[i, j] = np.inf
                                            
                print("Fourth loop time: ", time.time() - fourth_loop)

            print("Second loop time: ", time.time() - second_loop)
    
    return skg, rad