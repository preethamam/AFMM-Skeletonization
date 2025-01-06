from collections import deque
import cython
from libcpp.vector cimport vector
from libc.math cimport sqrt, fabs, INFINITY
from libc.stdint cimport uint8_t
from PIL import Image
import numpy as np
cimport numpy as np
from threading import Thread
from skimage.morphology import thin
import time

# Structures
cdef struct DataGrid:
    # '''
    # A structure to hold grid data for the algorithm.

    # Attributes:
    #     U (vector[int]): Vector of integers representing some state.
    #     T (vector[double]): Vector of doubles representing some state.
    #     f (vector[uint8_t]): Vector of unsigned 8-bit integers representing some state.
    #     set (vector[int]): Vector of integers representing some state.
    #     x (vector[double]): Vector of doubles representing x-coordinates.
    #     y (vector[double]): Vector of doubles representing y-coordinates.
    #     colNum (int): Number of columns in the grid.
    #     rowNum (int): Number of rows in the grid.
    # '''

    vector[int] U
    vector[double] T
    vector[uint8_t] f
    vector[int] set
    vector[double] x
    vector[double] y
    int colNum
    int rowNum

cdef struct pixelHeap:
    # """
    # A structure to represent a heap of pixels.

    # Attributes:
    #     data (DataGrid*): Pointer to the DataGrid structure.
    #     heapIndex (vector[int]): Vector of integers representing heap indices.
    #     pixelIds (vector[int]): Vector of integers representing pixel IDs.
    # """
    DataGrid* data
    vector[int] heapIndex
    vector[int] pixelIds

# Helper functions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int parent(int i) nogil:
    """
    Get the parent index of a given index in a heap.

    Args:
        i (int): The index of the current element.

    Returns:
        int: The index of the parent element.
    """
    return (i - 1) >> 1  # Fast division by 2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int left_child(int i) nogil:
    """
    Get the left child index of a given index in a heap.

    Args:
        i (int): The index of the current element.

    Returns:
        int: The index of the left child element.
    """
    return (i << 1) + 1  # Fast multiplication by 2

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int right_child(int i) nogil:
    """
    Get the right child index of a given index in a heap.

    Args:
        i (int): The index of the current element.

    Returns:
        int: The index of the right child element.
    """
    return (i << 1) + 2  # Fast multiplication by 2

# Heap operations
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void heap_swap(pixelHeap* h, int i, int j) nogil:
    """
    Swap two elements in the heap.

    Args:
        h (pixelHeap*): Pointer to the pixelHeap structure.
        i (int): Index of the first element.
        j (int): Index of the second element.
    """
    cdef int temp = h.pixelIds[i]
    h.pixelIds[i] = h.pixelIds[j]
    h.pixelIds[j] = temp
    h.heapIndex[h.pixelIds[i]] = i
    h.heapIndex[h.pixelIds[j]] = j

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sift_up(pixelHeap* h, int i) nogil:
    """
    Maintain the heap property by moving the element at index i up.

    Args:
        h (pixelHeap*): Pointer to the pixelHeap structure.
        i (int): Index of the element to move up.
    """
    cdef int parent_idx
    while i > 0:
        parent_idx = parent(i)
        if h.data.T[h.pixelIds[i]] < h.data.T[h.pixelIds[parent_idx]]:
            heap_swap(h, i, parent_idx)
            i = parent_idx
        else:
            break

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sift_down(pixelHeap* h, int i) nogil:
    """
    Maintain the heap property by moving the element at index i down.

    Args:
        h (pixelHeap*): Pointer to the pixelHeap structure.
        i (int): Index of the element to move down.
    """
    cdef int min_idx, left, right, size
    size = h.pixelIds.size()
    
    while True:
        min_idx = i
        left = left_child(i)
        right = right_child(i)
        
        if (left < size and 
            h.data.T[h.pixelIds[left]] < h.data.T[h.pixelIds[min_idx]]):
            min_idx = left
        
        if (right < size and 
            h.data.T[h.pixelIds[right]] < h.data.T[h.pixelIds[min_idx]]):
            min_idx = right
        
        if min_idx != i:
            heap_swap(h, i, min_idx)
            i = min_idx
        else:
            break

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int heap_pop(pixelHeap* h) nogil:
    """
    Remove and return the root element of the heap.

    Args:
        h (pixelHeap*): Pointer to the pixelHeap structure.

    Returns:
        int: The root element of the heap.
    """
    if h.pixelIds.empty():
        return -1
    
    cdef int root = h.pixelIds[0]
    cdef int last = h.pixelIds.back()
    h.pixelIds.pop_back()
    
    if not h.pixelIds.empty():
        h.pixelIds[0] = last
        h.heapIndex[last] = 0
        sift_down(h, 0)
    
    return root

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void heap_push(pixelHeap* h, int item) nogil:
    """
    Add an element to the heap.

    Args:
        h (pixelHeap*): Pointer to the pixelHeap structure.
        item (int): The element to add.
    """
    if h.data.f[item] == 1:
        h.heapIndex[item] = h.pixelIds.size()
        h.pixelIds.push_back(item)
        sift_up(h, h.pixelIds.size() - 1)
    else:
        sift_down(h, h.heapIndex[item])
        sift_up(h, h.heapIndex[item])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint heap_empty(pixelHeap* h) nogil:
    """
    Check if the heap is empty.

    Args:
        h (pixelHeap*): Pointer to the pixelHeap structure.

    Returns:
        bint: True if the heap is empty, False otherwise.
    """
    return h.pixelIds.empty()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void heap_init(pixelHeap* h) nogil:
    """
    Initialize the heap.

    Args:
        h (pixelHeap*): Pointer to the pixelHeap structure.
    """
    cdef int i
    for i in range(h.pixelIds.size() // 2 - 1, -1, -1):
        sift_down(h, i)

# Neighborhood functions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[int] vonNeumannNeighborhood(int idx, int colNum) nogil:
    """
    Get the von Neumann neighborhood of a given index.

    Args:
        idx (int): The index of the current element.
        colNum (int): Number of columns in the grid.

    Returns:
        vector[int]: The von Neumann neighborhood indices.
    """
    cdef:
        int x = idx % colNum
        int base = (idx // colNum) * colNum
        vector[int] result
        
    result.reserve(4)
    result.push_back(base + x - 1)
    result.push_back(base - colNum + x)
    result.push_back(base + x + 1)
    result.push_back(base + colNum + x)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[int] mooreNeighborhood(int idx, int colNum) nogil:
    """
    Get the Moore neighborhood of a given index.

    Args:
        idx (int): The index of the current element.
        colNum (int): Number of columns in the grid.

    Returns:
        vector[int]: The Moore neighborhood indices.
    """
    cdef:
        int x = idx % colNum
        int base = (idx // colNum) * colNum
        int up = base - colNum
        int down = base + colNum
        vector[int] result
    
    result.reserve(8)
    result.push_back(up + x - 1)
    result.push_back(up + x)
    result.push_back(up + x + 1)
    result.push_back(base + x + 1)
    result.push_back(down + x + 1)
    result.push_back(down + x)
    result.push_back(down + x - 1)
    result.push_back(base + x - 1)
    return result

# Continue with rest of the code...
@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[int] safeMooreNeighborhood(DataGrid* d, int idx) nogil:
    """
    Get the Moore neighborhood of a given index, ensuring all neighbors are within bounds.

    Args:
        d (DataGrid*): Pointer to the DataGrid structure.
        idx (int): The index of the current element.

    Returns:
        vector[int]: The Moore neighborhood indices.
    """
    cdef:
        vector[int] neighbors = mooreNeighborhood(idx, d.colNum)
        int offset = 0
        int i
    
    for i in range(8):
        if d.f[neighbors[i]] == 0:
            offset = i
            break
    
    cdef vector[int] neighborsStartingOutside
    neighborsStartingOutside.reserve(8)
    for i in range(8):
        neighborsStartingOutside.push_back(neighbors[(i + offset) % 8])
    
    return neighborsStartingOutside

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void solve(int idx1, int idx2, vector[double]& T, vector[uint8_t]& f, double* solution) nogil:
    """
    Solve the Eikonal equation for two neighboring points.

    Args:
        idx1 (int): Index of the first neighbor.
        idx2 (int): Index of the second neighbor.
        T (vector[double]): Vector of distances.
        f (vector[uint8_t]): Vector of flags.
        solution (double*): Pointer to the solution.
    """
    cdef:
        double r, s
        double t1, t2
    
    if f[idx1] == 0:
        if f[idx2] == 0:
            t1 = T[idx1]
            t2 = T[idx2]
            r = sqrt(2 - ((t1 - t2) * (t1 - t2)))
            s = (t1 + t2 - r) * 0.5
            if s >= t1 and s >= t2:
                if s < solution[0]:
                    solution[0] = s
                else:
                    s += r
                    if s >= t1 and s >= t2 and s < solution[0]:
                        solution[0] = s
        else:
            if 1 + T[idx1] < solution[0]:
                solution[0] = 1 + T[idx1]
    elif f[idx2] == 0:
        if 1 + T[idx2] < solution[0]:
            solution[0] = 1 + T[idx2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void parse_rgb_image(DataGrid* imgData, image):
    """
    Parse an RGB image and populate the DataGrid structure.

    Args:
        imgData (DataGrid*): Pointer to the DataGrid structure.
        image: The input image.
    """
    cdef:
        tuple bounds = image.size
        int width = bounds[0]
        int height = bounds[1]
        int padded_width = width + 2
        np.ndarray[np.float32_t, ndim=1] flat_array    
    
    imgData.colNum = padded_width
    imgData.rowNum = height + 2
    
    imgData.f.reserve(padded_width * (height + 2))
    imgData.T.reserve(padded_width * (height + 2))
    imgData.f.resize(padded_width * (height + 2))
    imgData.T.resize(padded_width * (height + 2))
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Handle both grayscale and RGB cases
    if img_array.ndim == 2:
        lum_array = img_array.astype(np.float32)
    else:
        # RGB weights for luminance calculation
        rgb_weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        lum_array = np.dot(img_array[..., :3], rgb_weights)
    
    # Create padded output arrays
    f_array = np.zeros((height + 2, padded_width), dtype=np.float32)
    T_array = np.zeros((height + 2, padded_width), dtype=np.float32)
    
    # Fill the inner region
    f_array[1:height+1, 1:width+1] = (lum_array > 128).astype(np.float32)
    T_array[1:height+1, 1:width+1] = np.where(lum_array > 128, INFINITY, 0)
    
    # Flatten arrays and copy directly to vectors
    flat_f = f_array.ravel()
    flat_T = T_array.ravel()
    
    # Use memoryview for direct memory access
    cdef float[:] f_view = flat_f
    cdef float[:] T_view = flat_T
    
    # Copy entire arrays at once using memoryview
    imgData.f.assign(&f_view[0], &f_view[0] + f_view.shape[0])
    imgData.T.assign(&T_view[0], &T_view[0] + T_view.shape[0])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void parse_binary_image(DataGrid* imgData, image):
    """
    Parse a binary image and populate the DataGrid structure.

    Args:
        imgData (DataGrid*): Pointer to the DataGrid structure.
        image: The input image.
    """
    cdef:
        tuple bounds = image.size
        int width = bounds[0]
        int height = bounds[1]
        int padded_width = width + 2
        np.ndarray[np.float32_t, ndim=1] flat_array    
    
    imgData.colNum = padded_width
    imgData.rowNum = height + 2
    
    imgData.f.reserve(padded_width * (height + 2))
    imgData.T.reserve(padded_width * (height + 2))
    imgData.f.resize(padded_width * (height + 2))
    imgData.T.resize(padded_width * (height + 2))
    
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Create padded output arrays
    f_array = np.zeros((height + 2, padded_width), dtype=np.float32)
    T_array = np.zeros((height + 2, padded_width), dtype=np.float32)
    
    # Fill the inner region
    f_array[1:height+1, 1:width+1] = (img_array > 0).astype(np.float32)
    T_array[1:height+1, 1:width+1] = np.where(img_array > 0, INFINITY, 0)
    
    # Flatten arrays
    flat_f = f_array.ravel()
    flat_T = T_array.ravel()
    
    # Use memoryview for direct memory access
    cdef float[:] f_view = flat_f
    cdef float[:] T_view = flat_T
    
    # Copy entire arrays at once using memoryview
    imgData.f.assign(&f_view[0], &f_view[0] + f_view.shape[0])
    imgData.T.assign(&T_view[0], &T_view[0] + T_view.shape[0])


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void guessU(DataGrid* d, int idx, vector[int]& neighbors) nogil:
    """
    Guess the U value for a given index based on its neighbors.

    Args:
        d (DataGrid*): Pointer to the DataGrid structure.
        idx (int): The index of the current element.
        neighbors (vector[int]&): Vector of neighbor indices.
    """
    cdef:
        double D = INFINITY
        double distance
        double dx, dy
        int neighbor
        double cur_x = idx % d.colNum
        double cur_y = idx // d.colNum
    
    for i in range(neighbors.size()):
        neighbor = neighbors[i]
        if d.f[neighbor] != 1:
            dx = cur_x - d.x[d.U[neighbor]]
            dy = cur_y - d.y[d.U[neighbor]]
            distance = sqrt(dx * dx + dy * dy)
            
            if distance < D:
                D = distance
                d.U[idx] = d.U[neighbor]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void stepAFMM(DataGrid* d, pixelHeap* h) nogil:
    """
    Perform one step of the AFMM algorithm.

    Args:
        d (DataGrid*): Pointer to the DataGrid structure.
        h (pixelHeap*): Pointer to the pixelHeap structure.
    """
    cdef:
        double solution
        int current = heap_pop(h)
        vector[int] neighbors
        vector[int] otherNeighbors
        vector[int] neighborhood
        int neighbor
    
    if current < 0:
        return
    
    d.f[current] = 0
    neighbors = vonNeumannNeighborhood(current, d.colNum)
    
    for i in range(neighbors.size()):
        neighbor = neighbors[i]
        if d.f[neighbor] != 0:
            otherNeighbors = vonNeumannNeighborhood(neighbor, d.colNum)
            solution = d.T[neighbor]
            
            solve(otherNeighbors[0], otherNeighbors[1], d.T, d.f, &solution)
            solve(otherNeighbors[2], otherNeighbors[1], d.T, d.f, &solution)
            solve(otherNeighbors[0], otherNeighbors[3], d.T, d.f, &solution)
            solve(otherNeighbors[2], otherNeighbors[3], d.T, d.f, &solution)
            
            d.T[neighbor] = solution
            heap_push(h, neighbor)
            
            if d.f[neighbor] == 1:
                d.f[neighbor] = 2
                neighborhood = mooreNeighborhood(neighbor, d.colNum)
                guessU(d, neighbor, neighborhood)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void initFMM(DataGrid* state, pixelHeap* h):
    """
    Initialize the Fast Marching Method (FMM) algorithm.

    Args:
        state (DataGrid*): Pointer to the DataGrid structure.
        h (pixelHeap*): Pointer to the pixelHeap structure.
    """
    cdef:
        int idx, j
        vector[int] neighbors
        np.ndarray[np.int_t, ndim=2] grid_idx
        np.ndarray[np.int_t, ndim=1] inner_indices
        np.ndarray[np.int_t, ndim=1] solid_cells
        np.ndarray[np.int_t, ndim=1] neighbor_states
    
    h.data = state
    h.heapIndex.resize(state.rowNum * state.colNum)
    
    grid_idx = np.arange(state.rowNum * state.colNum).reshape(state.rowNum, state.colNum)
    inner_indices = grid_idx[1:-1, 1:-1].ravel()
    solid_cells = inner_indices[np.asarray([state.f[i] for i in inner_indices]) == 1]

    if solid_cells.size == 0:
        return
        
    # Get neighbors using vonNeumannNeighborhood
    neighbors = vonNeumannNeighborhood(solid_cells[0], state.colNum)
    neighbor_states = np.asarray([state.f[neighbors[j]] for j in range(neighbors.size())])
    
    # Check if any neighbor is liquid (f == 0)
    if np.any(neighbor_states == 0):
        idx = solid_cells[0]
        h.heapIndex[idx] = h.pixelIds.size()
        h.pixelIds.push_back(idx)
        state.T[idx] = 0
        state.f[idx] = 2  # band

    heap_init(h)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void initAFMM(DataGrid* state, pixelHeap* h, bint startInFront):
    """
    Initialize the Adaptive Fast Marching Method (AFMM) algorithm.

    Args:
        state (DataGrid*): Pointer to the DataGrid structure.
        h (pixelHeap*): Pointer to the pixelHeap structure.
        startInFront (bint): Boolean indicating whether to start in front.
    """
    cdef:
        int idx = 0
        int count = 0
        int setID = 0
        int current = 0
        int j
        bint found = False
        vector[int] neighbors
    
    band_list = deque()
    idx_to_list = {}
    
    h.data = state
    h.heapIndex.resize(state.rowNum * state.colNum)
        
    for idx in range(state.colNum, (state.rowNum - 1) * state.colNum):
        if state.f[idx] == 1:
            neighbors = vonNeumannNeighborhood(idx, state.colNum)
            for j in range(neighbors.size()):
                if state.f[neighbors[j]] == 0:
                    if startInFront:
                        band_list.append(idx)
                    else:
                        band_list.appendleft(idx)
                    idx_to_list[idx] = idx
                    h.pixelIds.push_back(idx)
                    state.T[idx] = 0
                    state.f[idx] = 3  # band uninitialized
                    break

    state.x.resize(h.pixelIds.size())
    state.y.resize(h.pixelIds.size())
    state.set.resize(h.pixelIds.size())
    
    while len(band_list) > 0:
        current = band_list.popleft()
        
        state.U[current] = count
        state.f[current] = 2
        state.set[count] = setID
        state.x[count] = current % state.colNum
        state.y[count] = current // state.colNum
        count += 1
        
        found = True
        while found:
            found = False
            neighbors = safeMooreNeighborhood(state, current)
            for j in range(neighbors.size()):
                if state.f[neighbors[j]] == 3:
                    current = neighbors[j]
                    state.U[current] = count
                    state.f[current] = 2
                    state.set[count] = setID
                    state.x[count] = current % state.colNum
                    state.y[count] = current // state.colNum
                    count += 1
                    band_list.remove(idx_to_list[current])
                    found = True
                    break
        
        setID += 1
    
    heap_init(h)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void stepFMM(DataGrid* d, pixelHeap* h) nogil:
    """
    Perform one step of the Fast Marching Method (FMM) algorithm.

    Args:
        d (DataGrid*): Pointer to the DataGrid structure.
        h (pixelHeap*): Pointer to the pixelHeap structure.
    """
    cdef:
        double solution
        int current = heap_pop(h)
        vector[int] neighbors
        vector[int] otherNeighbors
        int neighbor
    
    if current < 0:
        return
    
    d.f[current] = 0
    neighbors = vonNeumannNeighborhood(current, d.colNum)
    
    for i in range(neighbors.size()):
        neighbor = neighbors[i]
        if d.f[neighbor] != 0:
            otherNeighbors = vonNeumannNeighborhood(neighbor, d.colNum)
            solution = d.T[neighbor]
            
            solve(otherNeighbors[0], otherNeighbors[1], d.T, d.f, &solution)
            solve(otherNeighbors[2], otherNeighbors[1], d.T, d.f, &solution)
            solve(otherNeighbors[0], otherNeighbors[3], d.T, d.f, &solution)
            solve(otherNeighbors[2], otherNeighbors[3], d.T, d.f, &solution)
            
            d.T[neighbor] = solution
            heap_push(h, neighbor)
            
            if d.f[neighbor] == 1:
                d.f[neighbor] = 2

def fmm(image, parse_image_type_rgb):
    """
    Compute the distance transform of the binary mask provided as argument.

    Args:
        image: The input image.
        parse_image_type_rgb (str): The type of image parsing ('binary' or 'rgb').

    Returns:
        np.ndarray: The distance transform of the binary mask.
    """
    cdef:
        DataGrid state
        int x, y, oldIdx, newIdx
        pixelHeap heap
        np.ndarray[np.float64_t, ndim=1] DT

    if parse_image_type_rgb == 'rgb':
        parse_rgb_image(&state, image)
    else:
        parse_binary_image(&state, image)

    state.U.resize(state.colNum * state.rowNum)
    initFMM(&state, &heap)

    while not heap_empty(&heap):
        stepFMM(&state, &heap)

    DT = np.zeros((state.colNum - 2) * (state.rowNum - 2), dtype=np.float64)

    for y in range(1, state.rowNum - 1):
        for x in range(1, state.colNum - 1):
            oldIdx = y * state.colNum + x
            newIdx = (y - 1) * (state.colNum - 2) + (x - 1)
            DT[newIdx] = state.T[oldIdx]

    return DT

def afmm(image, parse_image_type_rgb):
    """
    Compute the discontinuity magnitude and distance transform using the Adaptive Fast Marching Method (AFMM).

    Args:
        image: The input image.
        parse_image_type_rgb (str): The type of image parsing ('binary' or 'rgb').

    Returns:
        tuple: A tuple containing the discontinuity magnitude and distance transform.
    """
    cdef:
        DataGrid stateFirst, stateLast
        pixelHeap heap_first, heap_last
        vector[uint8_t] mask
        int x, y, oldIdx, newIdx, neighbor, i
        vector[int] neighbors
        double deltaUFirst, deltaULast, difference
        np.ndarray[np.float64_t, ndim=1] deltaU, DT
    
    if parse_image_type_rgb == 'rgb':
        parse_rgb_image(&stateFirst, image)
    else:
        parse_binary_image(&stateFirst, image)

    stateFirst.U.resize(stateFirst.colNum * stateFirst.rowNum)
    mask = stateFirst.f
    
    stateLast.colNum = stateFirst.colNum
    stateLast.rowNum = stateFirst.rowNum
    stateLast.f = stateFirst.f
    stateLast.T = stateFirst.T
    stateLast.U = stateFirst.U
    
    def run_afmm_first():
        nonlocal stateFirst
        initAFMM(&stateFirst, &heap_first, True)
        while not heap_empty(&heap_first):
            stepAFMM(&stateFirst, &heap_first)
    
    def run_afmm_last():
        nonlocal stateLast
        initAFMM(&stateLast, &heap_last, False)
        while not heap_empty(&heap_last):
            stepAFMM(&stateLast, &heap_last)
    
    # Run both AFMMs in parallel
    t1 = Thread(target=run_afmm_first)
    t2 = Thread(target=run_afmm_last)
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    deltaU = np.zeros((stateFirst.colNum - 2) * (stateFirst.rowNum - 2), dtype=np.float64)
    DT = np.zeros_like(deltaU)      
    
    # Process results
    for y in range(1, stateFirst.rowNum - 1):
        for x in range(1, stateFirst.colNum - 1):
            oldIdx = y * stateFirst.colNum + x
            if mask[oldIdx] == 0:
                continue
            
            newIdx = (y - 1) * (stateFirst.colNum - 2) + (x - 1)
            DT[newIdx] = stateFirst.T[oldIdx]
            
            deltaUFirst = 0
            deltaULast = 0
            
            neighbors = mooreNeighborhood(oldIdx, stateFirst.colNum)
            for i in range(neighbors.size()):
                neighbor = neighbors[i]
                if mask[neighbor] == 0:
                    continue
                
                if stateFirst.set[stateFirst.U[neighbor]] != stateFirst.set[stateFirst.U[oldIdx]]:
                    difference = INFINITY
                else:
                    difference = fabs(stateFirst.U[neighbor] - stateFirst.U[oldIdx])
                
                if deltaUFirst < difference:
                    deltaUFirst = difference
                
                if stateLast.set[stateLast.U[neighbor]] != stateLast.set[stateLast.U[oldIdx]]:
                    difference = INFINITY
                else:
                    difference = fabs(stateLast.U[neighbor] - stateLast.U[oldIdx])
                
                if deltaULast < difference:
                    deltaULast = difference
            
            if deltaUFirst < 3:
                deltaUFirst = 0
            if deltaULast < 3:
                deltaULast = 0
            
            deltaU[newIdx] = min(deltaUFirst, deltaULast)    
    return deltaU, DT

def get_skeleton(img, double t, str fmm_method, str parse_image_type_rgb, bint thinning):
    """
    Compute the skeleton of an image.

    Args:
        img: The input image.
        t (double): The threshold value.
        fmm_method (str): The method to use ('fmm' or 'afmm').
        parse_image_type_rgb (str): The type of image parsing ('binary' or 'rgb').
        thinning (bint): Boolean indicating whether to apply thinning.

    Returns:
        tuple: A tuple containing the distance transform image, deltaU image, and skeleton image.
    """
    cdef:
        int width = img.width
        int height = img.height
        np.ndarray dt_img, deltaU_img
        object skeleton = None
        np.ndarray[np.int_t, ndim=2] X, Y
        np.ndarray[np.int_t, ndim=1] linear_indices
        np.ndarray[np.uint8_t, ndim=1] output
    
    if fmm_method == 'fmm':
        dt = fmm(img, parse_image_type_rgb)
        dt_img = dt.reshape(height, width)
        return dt_img, None, None
    else:
        deltaU, dt = afmm(img, parse_image_type_rgb)
        deltaU_img = deltaU.reshape(height, width)
        dt_img = dt.reshape(height, width)
        
        X, Y = np.meshgrid(np.arange(width), np.arange(height))
        linear_indices = X.ravel() + width * Y.ravel()
        output = np.where(deltaU[linear_indices] > t, 255, 0).astype(np.uint8)
        
        if thinning:
            skeleton = Image.fromarray(thin(output.reshape(height, width)))
        else:
            skeleton = Image.fromarray(output.reshape(height, width))
        
        return dt_img, deltaU_img, skeleton