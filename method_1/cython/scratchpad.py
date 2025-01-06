import numpy as np
from enum import Enum

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
    Compute the skeleton gradient transform and optionally the skeleton radius.
    
    Parameters:
    img : numpy.ndarray
        Binary silhouette image
        
    Returns:
    tuple: (skeleton gradient transform, skeleton radius)
    """
    # Convert input to column-major (Fortran-style) ordering
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=bool, order='F')
    elif not img.flags.f_contiguous:
        img = np.asfortranarray(img)
    
    nrow, ncol = img.shape
    jnrow, jncol = nrow + 1, ncol + 1
    
    # Assuming img is a 2D array with shape (jnrow, jncol)
    img_flat = img.ravel('F')
    
    # Count junctions
    njunc = 0
    for j in range(jncol):
        for i in range(jnrow):
            jhood = joint_neighborhood(img_flat, i, j, nrow, ncol)
            if (jhood != 0) and (jhood != 15):
                njunc += 1
    
    # Initialize arrays
    jx = np.zeros(njunc, dtype=np.int32)
    jy = np.zeros(njunc, dtype=np.int32)
    seqj = np.zeros(njunc, dtype=np.int32)
    edgej = np.zeros(njunc, dtype=np.int32)
    seenj = np.zeros(jnrow * jncol, dtype=bool)
    
    # Distance arrays
    dNE = np.zeros(njunc, dtype=np.int32)
    dNW = np.zeros(njunc, dtype=np.int32)
    dSE = np.zeros(njunc, dtype=np.int32)
    dSW = np.zeros(njunc, dtype=np.int32)
    nearj = np.zeros(njunc, dtype=np.int32)
    
    # Register junctions
    ijunc = 0
    nedge = 0
    
    for j in range(jncol):
        for i in range(jnrow):
            jhood = joint_neighborhood(img_flat, i, j, nrow, ncol)
            if ((jhood != 0) and (jhood != 15) and (jhood != 5) and 
                (jhood != 10) and not seenj[i + j*jnrow]):
                # Found new edge; traverse it
                iseq = 0
                ei, ej = i, j
                lastdir = Direction.North
                
                while not seenj[ei + ej*jnrow] or (jhood == 5) or (jhood == 10):
                    if not seenj[ei + ej*jnrow]:
                        # Register junction
                        jx[ijunc] = ej
                        jy[ijunc] = ei
                        edgej[ijunc] = nedge
                        seqj[ijunc] = iseq
                        iseq += 1
                        ijunc += 1
                        seenj[ei + ej*jnrow] = True
                    
                    # Traverse clockwise
                    dir_code = DIRCODE[jhood]
                    if dir_code == Direction.North:
                        ei -= 1
                        lastdir = Direction.North
                    elif dir_code == Direction.South:
                        ei += 1
                        lastdir = Direction.South
                    elif dir_code == Direction.East:
                        ej += 1
                        lastdir = Direction.East
                    elif dir_code == Direction.West:
                        ej -= 1
                        lastdir = Direction.West
                    elif dir_code == Direction.None_:
                        if lastdir == Direction.East:
                            ei -= 1
                            lastdir = Direction.North
                        elif lastdir == Direction.West:
                            ei += 1
                            lastdir = Direction.South
                        elif lastdir == Direction.South:
                            ej += 1
                            lastdir = Direction.East
                        elif lastdir == Direction.North:
                            ej -= 1
                            lastdir = Direction.West
                    
                    if not (0 <= ei < jnrow and 0 <= ej < jncol):
                        raise ValueError("Traversed out of bounds")
                    
                    jhood = joint_neighborhood(img.ravel('F'), ei, ej, nrow, ncol)
                
                nedge += 1
    
    # Initialize output arrays in Fortran order
    skg = np.zeros((nrow, ncol), order='F')
    rad = np.zeros((nrow, ncol), order='F')
    
    # Count edge lengths
    edgelen = np.zeros(nedge, dtype=np.int32)
    for ijunc in range(njunc):
        edgelen[edgej[ijunc]] += 1
    
    # Compute skeleton gradient and radius
    # Note: Using column-major iteration order to match MATLAB
    for j in range(ncol):
        for i in range(nrow):
            if img[i, j]:  # Access img in the same order as we iterate
                # Compute distances to all junction points
                mind = mindNE = mindNW = mindSE = mindSW = (jnrow + jncol)**2
                minjunc = -1
                
                for ijunc in range(njunc):
                    dNE[ijunc] = (i - jy[ijunc])**2 + (j - jx[ijunc])**2
                    dNW[ijunc] = (i - jy[ijunc])**2 + (j + 1 - jx[ijunc])**2
                    dSE[ijunc] = (i + 1 - jy[ijunc])**2 + (j - jx[ijunc])**2
                    dSW[ijunc] = (i + 1 - jy[ijunc])**2 + (j + 1 - jx[ijunc])**2
                    
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
                
                rad[i, j] = mind  # Store in Fortran order
                
                # Find all junction points at minimal distance
                nnear = 0
                pspan = 0
                near_points = []
                
                for ijunc in range(njunc):
                    if ((dNE[ijunc] <= min(mindNE, dNE[minjunc])) or
                        (dNW[ijunc] <= min(mindNW, dNW[minjunc])) or
                        (dSE[ijunc] <= min(mindSE, dSE[minjunc])) or
                        (dSW[ijunc] <= min(mindSW, dSW[minjunc]))):
                        
                        if edgej[ijunc] != edgej[minjunc]:
                            pspan = -1
                            break
                        else:
                            near_points.append(seqj[ijunc])
                            nnear += 1
                
                if pspan >= 0:
                    # Compute perimeter span
                    near_points.sort()
                    pspan = near_points[0] - near_points[-1] + edgelen[edgej[minjunc]]
                    
                    for k in range(1, len(near_points)):
                        diff = near_points[k] - near_points[k-1]
                        if diff > pspan:
                            pspan = diff
                    
                    pspan = edgelen[edgej[minjunc]] - pspan
                    skg[i, j] = pspan  # Store in Fortran order
                else:
                    skg[i, j] = np.inf  # Store in Fortran order
    
    return skg, rad

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image
    import time
    
    # Load image
    img_path = "../../imgs/mushroom.png"
    # "../imgs/mushroom.png"
    # "../imgs/crack 2.png"
    # "../imgs/crack.bmp"
    img = Image.open(img_path)

    # Get skeleton
    threshold = 20  # Adjust as needed

    # Convert to binary image
    if img.mode != "L":
        print("Converting to grayscale image.")
        img = img.convert("L")

    # Load your binary image (should be boolean or 0/1 values)
    img = np.array(img)  # Your binary image here

    # Compute the skeleton gradient transform and radius
    # Get the start time
    st1 = time.time()
    skg, rad = compute_skeleton_gradient(img)
    # Get the end time
    et1 = time.time()
    
    # Print the execution time
    elapsed_time = et1 - st1
    print(f"Algorithm Execution Time (Skeletonize): {elapsed_time:.4f} seconds.")

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Skeleton Gradient")
    plt.imshow(skg, cmap='jet')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Skeleton Radius")
    plt.imshow(rad, cmap='jet')
    plt.colorbar()

    plt.show()