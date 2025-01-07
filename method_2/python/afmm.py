import sys

sys.dont_write_bytecode = True

import logging

from logging_config import *

logger = logging.getLogger(__name__)
logger.info('This is a log message from afmm.py')

from collections import deque
from threading import Thread

import numba
import numpy as np
from numba import jit
from PIL import Image
from skimage.morphology import thin

from datastructure import DataGrid, PixelHeap
from utils import (guess_U, moore_neighborhood, parse_binary_image,
                   parse_rgb_image, safe_moore_neighborhood, solve,
                   von_neumann_neighborhood)


class AFMMInitStep:
    
    @staticmethod                                   
    def init_fmm(state, band):
        # Initialize state and band
        band.data = state
        
        # Initialize heap index array
        band.heapIndex = np.zeros(state.rowNum * state.colNum, dtype=np.int32)
        
        # Convert state.f to numpy array
        f_array = np.array(state.f)
        
        # Create grid of indices
        y = np.arange(1, state.rowNum - 1)
        x = np.arange(1, state.colNum - 1)
        xx, yy = np.meshgrid(x, y)
        indices = yy * state.colNum + xx
        
        # Find visited pixels (f == 1)
        visited_mask = f_array[indices] == 1
        visited_indices = indices[visited_mask]
        
        # Create all neighbor indices at once
        offsets = np.array([-state.colNum, 1, state.colNum, -1])
        neighbor_indices = visited_indices[:, np.newaxis] + offsets
        
        # Check which visited pixels have unvisited neighbors
        has_unvisited = np.any(f_array[neighbor_indices] == 0, axis=1)
        band_indices = visited_indices[has_unvisited]
        
        # Update band information
        band.pixelIds = np.array(band_indices, np.int64)
        band.heapIndex[band_indices] = np.arange(len(band_indices), dtype=np.int32)
        
        # Update state arrays
        state.T[band_indices] = 0
        state.f[band_indices] = 2  # band
        
        # Initialize state arrays            
        band.init
    
    @staticmethod        
    def init_afmm(state, band, start_in_front):
        idx_to_list = {}
        band_list = deque()

        # Define add_to_list function based on start_in_front
        def add_to_list(item):
            if start_in_front:
                band_list.append(item)
            else:
                band_list.appendleft(item)
        
        # Initialize state and band
        band.data = state
        band.heapIndex = np.zeros(state.rowNum * state.colNum, dtype=np.int32)
        
        # Convert state.f to numpy array if it isn't already
        f_array = np.array(state.f)
        
        # Create grid of indices
        y = np.arange(1, state.rowNum - 1)
        x = np.arange(1, state.colNum - 1)
        xx, yy = np.meshgrid(x, y)
        indices = yy * state.colNum + xx
        
        # Find pixels where f == 1
        visited_mask = f_array[indices] == 1
        visited_indices = indices[visited_mask]
        
        # Create von Neumann neighborhood indices for all visited pixels at once
        offsets = np.array([-state.colNum, 1, state.colNum, -1])  # up, right, down, left
        neighbor_indices = visited_indices[:, np.newaxis] + offsets
    
        # Find pixels that have at least one unvisited neighbor (f == 0)
        has_unvisited = np.any(f_array[neighbor_indices] == 0, axis=1)
        band_indices = visited_indices[has_unvisited]
        
        # Create idx_to_list mapping for the band pixels
        idx_to_list = {}
        for idx in band_indices:
            idx_to_list[idx] = add_to_list(idx)
            idx_to_list[idx] = idx
        
        # Update band information
        band.pixelIds = np.array(band_indices, np.int64)        
        
        # Update state arrays
        state.T[band_indices] = 0
        state.f[band_indices] = 3  # 3: band uninitialized    
        
        # Initialize state arrays
        state.x = np.zeros(len(band.pixelIds), dtype=np.float64)
        state.y = np.zeros(len(band.pixelIds), dtype=np.float64)
        state.set = np.zeros(len(band.pixelIds), dtype=np.int32)

        # Initialize count and setID
        count = 0
        setID = 0
        
        while len(band_list) > 0:
            current = band_list.popleft()

            state.U[current] = count
            state.f[current] = 2
            state.set[count] = setID
            state.x[count] = float(current % state.colNum)
            state.y[count] = float(current // state.colNum)
            count += 1

            found = True
            while found:
                found = False
                neighbors = safe_moore_neighborhood(state, current)
                for j in neighbors:
                    if state.f[j] == 3:
                        current = j
                        state.U[current] = count
                        state.f[current] = 2
                        state.set[count] = setID
                        state.x[count] = float(current % state.colNum)
                        state.y[count] = float(current // state.colNum)
                        count += 1

                        # Remove from band_list
                        band_list.remove(idx_to_list[current])
                        found = True
                        break

            setID += 1

        band.init

    @staticmethod
    @jit(nopython=True)
    def step_fmm(d, band):
        solution = [float("inf")]

        while len(band):
            current = band.pop()
            d.f[current] = 0
            for neighbor in von_neumann_neighborhood(current, d.colNum):
                if d.f[neighbor] != 0:
                    other_neighbors = von_neumann_neighborhood(neighbor, d.colNum)
                    solution = d.T[neighbor]
                    solution = solve(other_neighbors[0], other_neighbors[1], d.T, d.f, solution)
                    solution = solve(other_neighbors[2], other_neighbors[1], d.T, d.f, solution)
                    solution = solve(other_neighbors[0], other_neighbors[3], d.T, d.f, solution)
                    solution = solve(other_neighbors[2], other_neighbors[3], d.T, d.f, solution)

                    d.T[neighbor] = solution
                    band.push(neighbor)                

                    if d.f[neighbor] == 1:
                        d.f[neighbor] = 2
        
        return d

    @staticmethod
    @jit(nopython=True)
    def step_afmm(d, band):
        solution = [float("inf")]
        
        while len(band) > 0:
            current = band.pop()
            d.f[current] = 0        
            for neighbor in von_neumann_neighborhood(current, d.colNum):
                if d.f[neighbor] != 0:
                    other_neighbors = von_neumann_neighborhood(neighbor, d.colNum)
                    solution = d.T[neighbor]
                    solution = solve(other_neighbors[0], other_neighbors[1], d.T, d.f, solution)
                    solution = solve(other_neighbors[2], other_neighbors[1], d.T, d.f, solution)
                    solution = solve(other_neighbors[0], other_neighbors[3], d.T, d.f, solution)
                    solution = solve(other_neighbors[2], other_neighbors[3], d.T, d.f, solution)

                    d.T[neighbor] = solution
                    band.push(neighbor)

                    if d.f[neighbor] == 1:
                        d.f[neighbor] = 2
                        neighborhood = moore_neighborhood(neighbor, d.colNum)
                        guess_U(d, neighbor, neighborhood)
                        
        return d
                    
    @staticmethod
    def afmm_init_step(state, start_in_front):
        band = PixelHeap(state)
        
        # Initialize state and band
        logging.info("AFMM init start")
        AFMMInitStep.init_afmm(state, band, start_in_front)
        logging.info("AFMM init end")

        logging.info("AFMM start")
        AFMMInitStep.step_afmm(state, band)
        logging.info("AFMM start")

class FastMarchingMethod:    
    
    # Vectorized function to calculate deltas
    @staticmethod
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
    
    @staticmethod
    def fast_marching_method(img, parse_image_type):
        # Initialize state and band
        state = DataGrid()
        band = PixelHeap(state)
        
        # Parse image
        import time
        time1 = time.time()
        if parse_image_type == "rgb":
            parse_rgb_image(img, state)
        else:
            parse_binary_image(img, state)
        print(f"Image parsing time: {time.time() - time1:.4f} seconds.")
           
        # Initialize state and band
        logging.info("FMM init start")
        AFMMInitStep.init_fmm(state, band)
        logging.info("FMM init end")

        # Fast marching method
        logging.info("FMM start")
        state = AFMMInitStep.step_fmm(state, band)
        logging.info("FMM end")

        # Extract DT
        DT = np.zeros((state.colNum - 2) * (state.rowNum - 2), dtype=np.float64)
        
        # Extract Distance Transform from state
        xx, yy = np.meshgrid(np.arange(1, state.colNum - 1), np.arange(1, state.rowNum - 1))
        DT[(yy - 1) * (state.colNum - 2) + (xx - 1)] = state.T[yy * state.colNum + xx]

        return DT

    @staticmethod
    def augmented_fast_marching_method(img, parse_image_type):
        stateFirst = DataGrid()
        
        if parse_image_type == "rgb":
            parse_rgb_image(img, stateFirst)
        else:
            parse_binary_image(img, stateFirst)
        
        # Initialize stateLast and stateFirst
        stateFirst.U = np.zeros(stateFirst.colNum * stateFirst.rowNum, dtype=np.int64)

        mask = np.copy(stateFirst.f)
        stateLast = DataGrid()
        stateLast.colNum, stateLast.rowNum = stateFirst.colNum, stateFirst.rowNum
        stateLast.f = np.zeros(stateFirst.colNum * stateFirst.rowNum, dtype=np.uint8)
        stateLast.T = np.zeros(stateFirst.colNum * stateFirst.rowNum, dtype=np.float64)

        stateLast.U = np.zeros(stateFirst.colNum * stateFirst.rowNum, dtype=np.int64)
    
        stateLast.f[:] = stateFirst.f
        stateLast.T[:] = stateFirst.T
        stateLast.U[:] = stateFirst.U
        
        # Initialize stateLast and band
        logging.info("Thread 1 start")
        thread1 = Thread(target=lambda: AFMMInitStep.afmm_init_step(stateFirst, True))
        
        logging.info("Thread 2 start")
        thread2 = Thread(target=lambda: AFMMInitStep.afmm_init_step(stateLast, False))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        logging.info("Thread end")

        # Calculate deltaU and DT
        import time
        st1 = time.time()
        
        # Initialize deltaU and DT
        deltaU = np.zeros(
            (stateFirst.colNum - 2) * (stateFirst.rowNum - 2), dtype=np.float64
        )        
        DT = np.zeros_like(deltaU)
                
        # Calculate all neighbor offsets for each position
        y_indices, x_indices = np.arange(1, stateFirst.rowNum - 1), np.arange(1, stateFirst.colNum - 1)
        Y, X = np.meshgrid(y_indices, x_indices, indexing='ij')
        old_indices = Y.ravel() * stateFirst.colNum + X.ravel()
        new_indices = (Y.ravel() - 1) * (stateFirst.colNum - 2) + (X.ravel() - 1)

        # Additional boundary check
        rows = old_indices // stateFirst.colNum
        cols = old_indices % stateFirst.colNum

        # Check if the current pixel and all its neighbors are non-zero in the mask
        valid_mask = mask[old_indices] != 0
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            neighbor_indices = (rows + dy) * stateFirst.colNum + (cols + dx)
            valid_mask &= mask[neighbor_indices] != 0

        # Filter out invalid indices
        old_indices = old_indices[valid_mask]
        new_indices = new_indices[valid_mask]

        # Copy DT values
        DT[new_indices] = stateFirst.T[old_indices]

        # Calculate all neighbor offsets for each position
        offsets = np.array([-stateFirst.colNum-1, -stateFirst.colNum, -stateFirst.colNum+1,
                            -1, 1,
                            stateFirst.colNum-1, stateFirst.colNum, stateFirst.colNum+1])        

        delta_first = FastMarchingMethod.calc_deltas_vectorized(stateFirst, old_indices, offsets, mask)
        delta_last = FastMarchingMethod.calc_deltas_vectorized(stateLast, old_indices, offsets, mask)
        deltaU[new_indices] = np.minimum(delta_first, delta_last)

        # Get the end time
        et1 = time.time()

        # Print the execution time
        elapsed_time = et1 - st1
        print(
            f"Algorithm Execution Time (AFMM TFT Skeleton Delta_U processing): {elapsed_time:.4f} seconds."
        )
    
        logging.info("DeltaU, DT array complete.")
        return deltaU, DT


class Skeletonize():
    @staticmethod
    def get_skeleton(img, t, fmm_method, parse_image_type, thinning=True):
        width = img.width
        height = img.height

        # Initialize
        dt_img = []
        deltaU_img = []
        skeleton = []

        # Fast Marching Method
        if fmm_method == "fmm":
            dt = FastMarchingMethod.fast_marching_method(img, parse_image_type)
            dt_img = dt.reshape(height, width)
        else:
            # Augmented Fast Marching Method
            deltaU, dt = FastMarchingMethod.augmented_fast_marching_method(img, parse_image_type)
            deltaU_img = deltaU.reshape(height, width)
            dt_img = deltaU.reshape(height, width)

            # Skeletonization
            X, Y = np.meshgrid(np.arange(width), np.arange(height))
            linear_indices = X + width * Y
            output = np.where(deltaU[linear_indices] > t, 255, 0).astype(np.uint8)
            
            if thinning:
                # Perform thinning
                skeleton = Image.fromarray(thin(output))
            else:
                skeleton = Image.fromarray(output)

        return dt_img, deltaU_img, skeleton