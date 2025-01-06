from scipy.ndimage import label, generate_binary_structure
s = generate_binary_structure(2,2)

d=0


# from PIL import Image
# from skimage.morphology import thin
# img = Image.open("imgs/100.png")

# skeleton = Image.fromarray(thin(img))
# skeleton.save("skeleton_100.png")

# from PIL import Image
# import numpy as np

# class DataGrid:
#     def __init__(self):
#         self.U = []
#         self.T = []
#         self.f = []
#         self.set = []
#         self.x = []
#         self.y = []
#         self.colNum = 0
#         self.rowNum = 0
        
# def parse_binary_image(img, imgData):
#     bounds = (0, 0, img.size[0], img.size[1])
#     imgData.colNum = bounds[2] - bounds[0] + 2
#     imgData.rowNum = bounds[3] - bounds[1] + 2
    
#     # Initialize arrays
#     imgData.f = np.zeros(imgData.colNum * imgData.rowNum, dtype=np.uint8)
#     imgData.T = np.zeros(imgData.colNum * imgData.rowNum, dtype=np.float64)
    
#     # Convert image region to numpy array
#     img_array = np.array(img.crop(bounds))
        
#     # Create indices for the target arrays
#     y_indices = np.arange(1, bounds[3] - bounds[1] + 1)
#     x_indices = np.arange(1, bounds[2] - bounds[0] + 1)
#     xx, yy = np.meshgrid(x_indices, y_indices)
#     indices = xx + yy * imgData.colNum
    
#     # Set values based on luminance threshold
#     mask = img_array == True
#     imgData.f[indices] = mask
#     imgData.T[indices] = np.where(mask, np.inf, 0)


#  # Load image
# img = Image.open("imgs/crack.bmp") #
# imgData = DataGrid()
# parse_binary_image(img, imgData)
    
# for y in range(1, stateFirst.rowNum - 1):
#     for x in range(1, stateFirst.colNum - 1):
#         oldIdx = y * stateFirst.colNum + x
#         if mask[oldIdx] == 0:
#             continue
#         newIdx = (y - 1) * (stateFirst.colNum - 2) + (x - 1)
#         DT[newIdx] = stateFirst.T[oldIdx]

#         deltaUFirst = 0
#         deltaULast = 0

#         for neighbor in ImageUtils().moore_neighborhood(oldIdx, stateFirst.colNum):
#             if mask[neighbor] == 0:
#                 continue

#             if (
#                 stateFirst.set[stateFirst.U[neighbor]]
#                 != stateFirst.set[stateFirst.U[oldIdx]]
#             ):
#                 difference = float("inf")
#             else:
#                 difference = abs(stateFirst.U[neighbor] - stateFirst.U[oldIdx])

#             if deltaUFirst < difference:
#                 deltaUFirst = difference

#             if (
#                 stateLast.set[stateLast.U[neighbor]]
#                 != stateLast.set[stateLast.U[oldIdx]]
#             ):
#                 difference = float("inf")
#             else:
#                 difference = abs(stateLast.U[neighbor] - stateLast.U[oldIdx])

#             if deltaULast < difference:
#                 deltaULast = difference

#         if deltaUFirst < 3:
#             deltaUFirst = 0

#         if deltaULast < 3:
#             deltaULast = 0

#         deltaU[newIdx] = deltaUFirst
#         if deltaULast < deltaUFirst:
#             deltaU[newIdx] = deltaULast



''' outer boundary 
# Create arrays for valid positions
y_indices, x_indices = np.arange(1, stateFirst.rowNum - 1), np.arange(1, stateFirst.colNum - 1)
Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
old_indices = Y.ravel() * stateFirst.colNum + X.ravel()
new_indices = (Y.ravel() - 1) * (stateFirst.colNum - 2) + (X.ravel() - 1)

valid_mask = mask[old_indices] != 0
old_indices = old_indices[valid_mask]
new_indices = new_indices[valid_mask]

DT[new_indices] = stateFirst.T[old_indices]

# Calculate all neighbor offsets for each position
offsets = np.array([-stateFirst.colNum-1, -stateFirst.colNum, -stateFirst.colNum+1,
                    -1, 1,
                    stateFirst.colNum-1, stateFirst.colNum, stateFirst.colNum+1])

def calc_deltas_vectorized(state, positions, offsets, mask):
    # Generate all neighbor indices (N x 8)
    neighbors = positions[:, np.newaxis] + offsets[np.newaxis, :]
    
    # Get valid neighbor mask (N x 8)
    valid_mask = mask[neighbors]
    
    # Get U values using advanced indexing
    position_u = np.take(state.U, positions)  # (N,)
    neighbor_u = np.take(state.U, neighbors)  # (N, 8)
    
    # Get set values
    position_set = np.take(state.set, np.take(state.U, positions))  # (N,)
    neighbor_set = np.take(state.set, np.take(state.U, neighbors))  # (N, 8)
    
    # Calculate differences where sets match and neighbors are valid
    same_set = neighbor_set == position_set[:, np.newaxis]  # (N, 8)
    valid_and_same = same_set & valid_mask
    
    diffs = np.where(
        valid_and_same,
        np.abs(neighbor_u - position_u[:, np.newaxis]),
        np.inf
    )
    
    # Get maximum difference per position
    max_diffs = np.max(diffs, axis=1)  # (N,)
    return np.where(max_diffs < 3, 0, max_diffs)

delta_first = calc_deltas_vectorized(stateFirst, old_indices, offsets, mask)
delta_last = calc_deltas_vectorized(stateLast, old_indices, offsets, mask)
deltaU[new_indices] = np.minimum(delta_first, delta_last)
'''

'''
# Create arrays for valid positions
y_indices = np.arange(1, stateFirst.rowNum - 1)
x_indices = np.arange(1, stateFirst.colNum - 1)
Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
old_indices = Y.ravel() * stateFirst.colNum + X.ravel()
new_indices = (Y.ravel() - 1) * (stateFirst.colNum - 2) + (X.ravel() - 1)

valid_positions = mask[old_indices] != 0
old_indices = old_indices[valid_positions]
new_indices = new_indices[valid_positions]

# Copy temperatures for valid positions
DT[new_indices] = stateFirst.T[old_indices]

# Process each valid position
for i, (oldIdx, newIdx) in enumerate(zip(old_indices, new_indices)):
    deltaUFirst = 0
    deltaULast = 0
    
    for neighbor in ImageUtils().moore_neighborhood(oldIdx, stateFirst.colNum):
        if mask[neighbor] == 0:
            continue
            
        if stateFirst.set[stateFirst.U[neighbor]] != stateFirst.set[stateFirst.U[oldIdx]]:
            difference = float("inf")
        else:
            difference = abs(stateFirst.U[neighbor] - stateFirst.U[oldIdx])
        deltaUFirst = max(deltaUFirst, difference)
        
        if stateLast.set[stateLast.U[neighbor]] != stateLast.set[stateLast.U[oldIdx]]:
            difference = float("inf")
        else:
            difference = abs(stateLast.U[neighbor] - stateLast.U[oldIdx])
        deltaULast = max(deltaULast, difference)
    
    deltaUFirst = 0 if deltaUFirst < 3 else deltaUFirst
    deltaULast = 0 if deltaULast < 3 else deltaULast
    deltaU[newIdx] = min(deltaUFirst, deltaULast)
'''
        


                    
# def init_afmm(self, state, band, start_in_front):
#         idx_to_list = {}
#         band_list = deque()

#         # Define add_to_list function based on start_in_front
#         def add_to_list(item):
#             if start_in_front:
#                 band_list.append(item)
#             else:
#                 band_list.appendleft(item)

#         band.data = state
#         band.heapIndex = [0] * (state.rowNum * state.colNum)
        # for y in range(1, state.rowNum - 1):
        #     for x in range(1, state.colNum - 1):
        #         idx = y * state.colNum + x
        #         if state.f[idx] == 1:
        #             for j in ImageUtils().von_neumann_neighborhood(idx, state.colNum):
        #                 if state.f[j] == 0:
        #                     idx_to_list[idx] = add_to_list(idx)
        #                     idx_to_list[idx] = idx
        #                     band.pixelIds.append(idx)
        #                     state.T[idx] = 0
        #                     state.f[idx] = 3  # 3: band uninitialized
        #                     break

        # # Create grid coordinates
        # y, x = np.meshgrid(np.arange(1, state.rowNum - 1), np.arange(1, state.colNum - 1), indexing='ij')
        # indices = y * state.colNum + x
        # active_mask = state.f[indices] == 1
        # active_indices = indices[active_mask]

        # # Vectorize neighbor checks using the original function
        # neighborhoods = np.array([ImageUtils().von_neumann_neighborhood(idx, state.colNum) for idx in active_indices])
        # has_unvisited = np.any(state.f[neighborhoods] == 0, axis=1)
        # band_indices = active_indices[has_unvisited]

        # # Vectorized updates
        # if len(band_indices) > 0:
        #     state.T[band_indices] = 0
        #     state.f[band_indices] = 3
            
        #     # Update tracking arrays (this could be vectorized further depending on add_to_list implementation)
        #     for idx in band_indices:
        #         idx_to_list[idx] = add_to_list(idx)
        #         idx_to_list[idx] = idx
        #         band.pixelIds.append(idx)

# def init_fmm(self, state, band):
    #     band.data = state
    
    #     # Initialize heap index array
    #     band.heapIndex = np.zeros(state.rowNum * state.colNum, dtype=np.int32)
        
    #     # Convert state.f to numpy array
    #     f_array = np.array(state.f)
        
    #     # Create grid of indices
    #     y = np.arange(1, state.rowNum - 1)
    #     x = np.arange(1, state.colNum - 1)
    #     xx, yy = np.meshgrid(x, y)
    #     indices = yy * state.colNum + xx
        
    #     # Find visited pixels (f == 1)
    #     visited_mask = f_array[indices] == 1
    #     visited_indices = indices[visited_mask]
        
    #     # Create all neighbor indices at once
    #     offsets = np.array([-state.colNum, 1, state.colNum, -1])
    #     neighbor_indices = visited_indices[:, np.newaxis] + offsets
        
    #     # Check which visited pixels have unvisited neighbors
    #     has_unvisited = np.any(f_array[neighbor_indices] == 0, axis=1)
    #     band_indices = visited_indices[has_unvisited]
        
    #     # Update band information
    #     band.pixelIds = band_indices.tolist()
    #     band.heapIndex[band_indices] = np.arange(len(band_indices))
        
    #     # Update state arrays
    #     state.T[band_indices] = 0
    #     state.f[band_indices] = 2  # band
        
    #     # Initialize state arrays                        
    #     band.init

# def init_fmm(self, state, band):
#         band.data = state
#         band.heapIndex = np.zeros(state.rowNum * state.colNum, dtype=np.int32)
        
#         # Create grid coordinates
#         y, x = np.meshgrid(np.arange(1, state.rowNum - 1), np.arange(1, state.colNum - 1), indexing='ij')
#         indices = y * state.colNum + x
#         active_mask = state.f[indices] == 1
#         active_indices = indices[active_mask]
        
#         # Vectorize neighbor checks using the original function
#         neighborhoods = np.array([ImageUtils().von_neumann_neighborhood(idx, state.colNum) for idx in active_indices])
#         has_unvisited = np.any(state.f[neighborhoods] == 0, axis=1)
#         band_indices = active_indices[has_unvisited]
        
#         # Vectorized updates
#         if len(band_indices) > 0:
#             # Update heapIndex with sequential indices
#             band.heapIndex[band_indices] = np.arange(len(band_indices))
            
#             # Update band information
#             band.pixelIds.extend(band_indices)
            
#             # Update state arrays
#             state.T[band_indices] = 0
#             state.f[band_indices] = 2  # band
            
#         band.init
        
    # def init_fmm(self, state, band):
    #     band.data = state
    #     band.heapIndex = [0] * (state.rowNum * state.colNum)
    #     for y in range(1, state.rowNum - 1):
    #         for x in range(1, state.colNum - 1):
    #             idx = y * state.colNum + x
    #             if state.f[idx] == 1:
    #                 for j in ImageUtils().von_neumann_neighborhood(idx, state.colNum):
    #                     if state.f[j] == 0:
    #                         band.heapIndex[idx] = len(band.pixelIds)
    #                         band.pixelIds.append(idx)
    #                         state.T[idx] = 0
    #                         state.f[idx] = 2  # band
    #                         break
    #     band.init                
                
    #     # Add to the list and store reference in map
    #     band_list.append(idx)  # or appendleft, depending on add_to_list
    #     return len(band_list) - 1  # store index instead of Element pointer  
    # Define a function to add to the list
    # if not start_in_front:
    #     add_to_list = band_list.append  # PushBack equivalent
    # else:
    #     add_to_list = band_list.appendleft  # PushFront equivalent          

    # add_to_list = band_list.append
    # if not start_in_front:
    #     add_to_list = band_list.appendleft
    
    # Define add_to_list function based on start_in_front
    # def add_to_list(item):
    #     if start_in_front:
    #         band_list.append(item)
    #     else:
    #         band_list.appendleft(item)
        # Return a reference to track the item's position
        # return len(band_list) - 1 if start_in_front else 0

    # # Define the add_to_list function based on startInFront
    # if not start_in_front:
    #     add_to_list = band_list.appendleft  # equivalent to PushFront
    # else:
    #     add_to_list = band_list.append  # equivalent to PushBack

    # Add to the list and store reference in map
    # band_list.append(idx)  # or appendleft, depending on add_to_list
    
    # def init_afmm(self, state, band, start_in_front):
    #     idx_to_list = {}
    #     band_list = deque()

    #     # Define add_to_list function based on start_in_front
    #     def add_to_list(item):
    #         if start_in_front:
    #             band_list.append(item)
    #         else:
    #             band_list.appendleft(item)
        
    #     # Initialize state and band
    #     band.data = state
    #     band.heapIndex = [0] * (state.rowNum * state.colNum)
                
    #     # Create grid coordinates
    #     y, x = np.meshgrid(np.arange(1, state.rowNum - 1), np.arange(1, state.colNum - 1), indexing='ij')
    #     indices = y * state.colNum + x
    #     active_mask = state.f[indices] == 1
    #     active_indices = indices[active_mask]

    #     # Vectorize neighbor checks using the original function
    #     neighborhoods = np.array([ImageUtils().von_neumann_neighborhood(idx, state.colNum) for idx in active_indices])
    #     has_unvisited = np.any(state.f[neighborhoods] == 0, axis=1)
    #     band_indices = active_indices[has_unvisited]

    #     # Vectorized updates
    #     if len(band_indices) > 0:
    #         state.T[band_indices] = 0
    #         state.f[band_indices] = 3
            
    #         # Vectorized tracking array updates
    #         list_indices = [add_to_list(i) for i in band_indices]  # Can't fully vectorize due to stateful queue
    #         idx_to_list.update(dict(zip(band_indices, list_indices)))
    #         idx_to_list.update(dict(zip(band_indices, band_indices)))
    #         band.pixelIds.extend(band_indices)        
    
    #     state.x = np.zeros(len(band.pixelIds), dtype=np.float64)
    #     state.y = np.zeros(len(band.pixelIds), dtype=np.float64)
    #     state.set = np.zeros(len(band.pixelIds), dtype=np.int32)