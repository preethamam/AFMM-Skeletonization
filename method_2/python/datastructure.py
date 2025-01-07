import sys

sys.dont_write_bytecode = True
import logging
import numpy as np
from logging_config import *
from numba import uint8, int32, float64, int64
from numba.experimental import jitclass

logger = logging.getLogger(__name__)
logger.info('This is a log message from datastructure.py')

spec_data_grid = [
    ('U', int64[:]),
    ('T', float64[:]),
    ('f', uint8[:]),
    ('set', int32[:]),
    ('x', float64[:]),
    ('y', float64[:]),
    ('colNum', int32),
    ('rowNum', int32),
]

@jitclass(spec_data_grid)
class DataGrid:
    def __init__(self):
        self.U =  int64[:]
        self.T = float64[:]
        self.f = uint8[:]
        self.set = int32[:]
        self.x = float64[:]
        self.y = float64[:]
        self.colNum = 0
        self.rowNum = 0

spec_pixel_heap = [
    ('data', DataGrid.class_type.instance_type),
    ('heapIndex', int32[:]),
    ('pixelIds', int64[:])
]

@jitclass(spec_pixel_heap)
class PixelHeap:
    def __init__(self, data):
        self.data = data
        self.heapIndex = int32[:]
        self.pixelIds = int64[:]
        
    def init(self) -> None:
        """
        Init establishes the heap invariants required by the other routines in this package.
        Init is idempotent with respect to the heap invariants
        and may be called whenever the heap invariants may have been invalidated.
        The complexity is O(n) where n = h.len().
        """
        n = len(self.pixelIds)
        for i in range(n // 2 - 1, -1, -1):
            self.__down(self.pixelIds, i, n)

    def __len__(self):
        return len(self.pixelIds)

    def __less(self, i, j):
        return self.data.T[self.pixelIds[i]] < self.data.T[self.pixelIds[j]]

    def __swap(self, i, j):
        self.pixelIds[i], self.pixelIds[j] = self.pixelIds[j], self.pixelIds[i]
        self.heapIndex[self.pixelIds[i]] = i
        self.heapIndex[self.pixelIds[j]] = j

    def __fix(self, i: int) -> None:
        """
        Fix re-establishes the heap ordering after the element at index i has changed its value.
        Changing the value of the element at index i and then calling fix is equivalent to,
        but less expensive than, calling remove(h, i) followed by a push of the new value.
        The complexity is O(log n) where n = h.len().
        """
        if not self.__down(i, len(self.pixelIds)):
            self.__up(i)

    def __up(self, j: int) -> None:
        """Internal function to move an element up the heap."""
        while True:
            i = int((j - 1) / 2)  # i = (j - 1) // 2  --> Killed me for 3 days
            if i == j or not self.__less(j, i):
                break
            self.__swap(i, j)
            j = i

    def __down(self, i0: int, n: int) -> bool:
        """Internal function to move an element down the heap."""
        i = i0
        while True:
            j1 = 2 * i + 1
            if j1 >= n or j1 < 0:  # j1 < 0 after int overflow
                break
            j = j1  # left child
            j2 = j1 + 1
            if j2 < n and self.__less(j2, j1):
                j = j2  # = 2*i + 2  # right child
            if not self.__less(j, i):
                break
            self.__swap(i, j)
            i = j
        return i > i0

    def push(self, x):
        """
        Push pushes the element x onto the heap.
        The complexity is O(log n) where n = h.len().
        """
        if self.data.f[x] == 1:
            self.heapIndex[x] = len(self.pixelIds)
            self.pixelIds = np.concatenate((self.pixelIds, np.array([x])))
            return
        self.__fix(self.heapIndex[x])

    def pop(self):
        """
        Pop removes and returns the minimum element (according to less) from the heap.
        The complexity is O(log n) where n = h.len().
        Pop is equivalent to remove(h, 0).
        """
        n = len(self.pixelIds) - 1
        self.__swap(0, n)
        self.__down(0, n)
        x = self.pixelIds[len(self.pixelIds) - 1]
        self.pixelIds = self.pixelIds[: len(self.pixelIds) - 1]
        return x