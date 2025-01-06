import sys

sys.dont_write_bytecode = True

import logging

from logging_config import *

logger = logging.getLogger(__name__)
logger.info('This is a log message from utils.py')
import math

import numpy as np
from numba import jit

def parse_rgb_image(img, imgData):
    bounds = img.getbbox()
    imgData.colNum = bounds[2] - bounds[0] + 2
    imgData.rowNum = bounds[3] - bounds[1] + 2
    
    # Initialize arrays
    imgData.f = np.zeros(imgData.colNum * imgData.rowNum, dtype=np.uint8)
    imgData.T = np.zeros(imgData.colNum * imgData.rowNum, dtype=np.float64)
    
    # Convert image region to numpy array
    img_array = np.array(img.crop(bounds))
    
    # Calculate luminance for all pixels at once
    # RGB weights: R=0.299, G=0.587, B=0.114
    luminance = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
    
    # Create indices for the target arrays
    y_indices = np.arange(1, bounds[3] - bounds[1] + 1)
    x_indices = np.arange(1, bounds[2] - bounds[0] + 1)
    xx, yy = np.meshgrid(x_indices, y_indices)
    indices = xx + yy * imgData.colNum
    
    # Set values based on luminance threshold
    mask = luminance > 128
    imgData.f[indices] = mask
    imgData.T[indices] = np.where(mask, np.inf, 0)
    
def parse_binary_image(img, imgData):
    bounds = (0, 0, img.size[0], img.size[1])
    imgData.colNum = bounds[2] - bounds[0] + 2
    imgData.rowNum = bounds[3] - bounds[1] + 2
    
    # Initialize arrays
    imgData.f = np.zeros(imgData.colNum * imgData.rowNum, dtype=np.uint8)
    imgData.T = np.zeros(imgData.colNum * imgData.rowNum, dtype=np.float64)
    
    # Convert image region to numpy array
    img_array = np.array(img.crop(bounds))
        
    # Create indices for the target arrays
    y_indices = np.arange(1, bounds[3] - bounds[1] + 1)
    x_indices = np.arange(1, bounds[2] - bounds[0] + 1)
    xx, yy = np.meshgrid(x_indices, y_indices)
    indices = xx + yy * imgData.colNum
    
    # Set values based on luminance threshold
    mask = img_array == True
    imgData.f[indices] = mask
    imgData.T[indices] = np.where(mask, np.inf, 0)

@jit(nopython=True)
def von_neumann_neighborhood(idx, colNum):
    x = idx % colNum
    y = idx // colNum
    y = y * colNum

    return [y + x - 1, y - colNum + x, y + x + 1, y + colNum + x]

@jit(nopython=True)
def moore_neighborhood(idx, colNum):
    x = idx % colNum
    y = idx // colNum
    y = y * colNum
    ym1 = y - colNum
    yp1 = y + colNum
    return [
        ym1 + (x - 1),
        ym1 + x,
        ym1 + (x + 1),
        y + (x + 1),
        yp1 + (x + 1),
        yp1 + x,
        yp1 + (x - 1),
        y + (x - 1),
    ]

@jit(nopython=True)
def safe_moore_neighborhood(d, idx):
    neighbors = moore_neighborhood(idx, d.colNum)
    for i, neighbor in enumerate(neighbors):
        if d.f[neighbor] == 0:
            offset = i
            break
    return [neighbors[(i + offset) % 8] for i in range(8)]    

@jit(nopython=True)
def solve(idx1, idx2, T, f, solution):
    if f[idx1] == 0:
        if f[idx2] == 0:
            r = math.sqrt(2 - ((T[idx1] - T[idx2]) ** 2))
            s = (T[idx1] + T[idx2] - r) * 0.5
            if s >= T[idx1] and s >= T[idx2]:
                if s < solution:
                    solution = s
                else:
                    s += r
                    if s >= T[idx1] and s >= T[idx2]:
                        if s < solution:
                            solution = s
        else:
            if 1 + T[idx1] < solution:
                solution = 1 + T[idx1]
    else:
        if f[idx2] == 0:
            if 1 + T[idx2] < solution:
                solution = 1 + T[idx2]

    return solution

@jit(nopython=True)
def guess_U(d, idx, neighbors):
    D = float("inf")
    for neighbor in neighbors:
        if d.f[neighbor] != 1:
            dx = float(idx % d.colNum) - d.x[d.U[neighbor]]
            dy = float(idx // d.colNum) - d.y[d.U[neighbor]]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < D:
                D = distance
                d.U[idx] = d.U[neighbor]
