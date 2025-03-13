# distutils: language = c++

import cython
from libc.float cimport FLT_MAX, DBL_MAX

import numpy as np

cimport numpy as cnp

cnp.import_array()
DTYPE = np.int64

ctypedef cnp.int64_t DTYPE_t


cdef extern from "stdint.h":
    ctypedef unsigned long uint64_t


# largest 64 bit prime
cdef long PRIME = 0xffffffffffffffc5


cdef class CountMin:

    def __cinit__(self, height, width):
        self.sketch = np.zeros(shape=(height, width), dtype=np.float64)
        info = np.iinfo(np.uint64)
        self.offset_a = np.random.randint(low=0, high=PRIME - 1, size=height, dtype=np.uint64)
        self.offset_b = np.random.randint(low=0, high=PRIME - 1, size=height, dtype=np.uint64)
        self.width = width
        self.height = height


    def __init__(self, width, height):
        pass


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef void add(self, long idx):
        cdef double[:, :] arr = self.sketch
        cdef uint64_t h = self.height
        cdef uint64_t w = self.width
        cdef uint64_t hashval = int_hash(idx)
        cdef uint64_t _idx

        for h_i in range(h):
            _idx = ((self.offset_a[h_i] * hashval + self.offset_b[h_i]) % PRIME) % w
            self.sketch[h_i, _idx] += 1


    def add_batch(self, inp):
        self._add_batch(np.array(inp).astype(np.int64))


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef void _add_batch(self, cnp.ndarray[DTYPE_t, ndim=1] inp):
        cdef double[:, :] arr = self.sketch
        cdef uint64_t h = self.height
        cdef uint64_t w = self.width
        cdef uint64_t l = inp.shape[0]
        cdef uint64_t h_i, l_i
        cdef uint64_t hashval
        cdef uint64_t _idx
        for l_i in range(l):
            hashval = int_hash(inp[l_i])
            for h_i in range(self.offset_a.shape[0]):
                _idx = (self.offset_a[h_i] * hashval + self.offset_b[h_i]) % w
                self.sketch[h_i, _idx] += 1


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef double getcount(self, int idx):
        cdef double[:, :] arr = self.sketch
        cdef uint64_t h = self.height
        cdef uint64_t w = self.width
        cdef uint64_t hashval = int_hash(idx)
        cdef uint64_t _idx
        cdef double minval = DBL_MAX

        for h_i in range(h):
            _idx = (self.offset_a[h_i] * hashval + self.offset_b[h_i]) % w
            if self.sketch[h_i, _idx] < minval:
                minval = self.sketch[h_i, _idx]
        return minval


    cpdef void clear(self):
        self.sketch[:,:] = 0


cdef class HCountMin:

    def __cinit__(self, height, width):
        self.sketch = np.zeros(shape=(height, width, width), dtype=np.float64)
        info = np.iinfo(np.uint64)
        self.offset_a = np.random.randint(low=1, high=0xffffffffffffffc5 - 1, size=height, dtype=np.uint64)
        self.offset_b = np.random.randint(low=0, high=0xffffffffffffffc5 - 1, size=height, dtype=np.uint64)
        self.width = width
        self.height = height


    def __init__(self, width, height):
        pass


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef void add(self, long source, long target):
        cdef double[:, :, :] arr = self.sketch
        cdef uint64_t h = self.height
        cdef uint64_t w = self.width
        cdef uint64_t source_hashval = int_hash(source)
        cdef uint64_t target_hashval = int_hash(target)
        cdef uint64_t _idx

        for h_i in range(h):
            _sidx = ((self.offset_a[h_i] * source_hashval + self.offset_b[h_i]) % PRIME) % w
            _tidx = ((self.offset_a[h_i] * target_hashval + self.offset_b[h_i]) % PRIME) % w
            self.sketch[h_i, _sidx, _tidx] += 1


    def add_batch(self, inp):
        self._add_batch(np.array(inp).astype(np.int64))


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef void _add_batch(self, cnp.ndarray[DTYPE_t, ndim=2] inp):
        cdef double[:, :, :] arr = self.sketch
        cdef uint64_t h = self.height
        cdef uint64_t w = self.width
        cdef uint64_t l = inp.shape[0]
        cdef uint64_t h_i, l_i
        cdef uint64_t source_hashval
        cdef uint64_t target_hashval
        cdef uint64_t _idx

        for l_i in range(l):
            source_hashval = int_hash(inp[l_i][0])
            target_hashval = int_hash(inp[l_i][1])
            for h_i in range(h):
                _sidx = ((self.offset_a[h_i] * source_hashval + self.offset_b[h_i]) % PRIME) % w
                _tidx = ((self.offset_a[h_i] * target_hashval + self.offset_b[h_i]) % PRIME) % w
                self.sketch[h_i, _sidx, _tidx] += 1


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cpdef double getcount(self, long source, long target):
        cdef double[:, :, :] arr = self.sketch
        cdef uint64_t h = self.height
        cdef uint64_t w = self.width
        cdef uint64_t source_hashval = int_hash(source)
        cdef uint64_t target_hashval = int_hash(target)
        cdef uint64_t _idx
        cdef double minval = DBL_MAX

        for h_i in range(h):
            _sidx = ((self.offset_a[h_i] * source_hashval + self.offset_b[h_i]) % PRIME) % w
            _tidx = ((self.offset_a[h_i] * target_hashval + self.offset_b[h_i]) % PRIME) % w
            if self.sketch[h_i, _sidx, _tidx] < minval:
                minval = self.sketch[h_i, _sidx, _tidx]
        return minval


    cpdef void clear(self):
        self.sketch[:,:, :] = 0


cdef uint64_t int_hash(long key):
    return <uint64_t>key
    # key = (~key) + (key << 21)  # key = (key << 21) - key - 1
    # key = key ^ (key >> 24)
    # key = (key + (key << 3)) + (key << 8)  # key * 265
    # key = key ^ (key >> 14)
    # key = (key + (key << 2)) + (key << 4)  # key * 21
    # key = key ^ (key >> 28)
    # key = key + (key << 31)
    # return <uint64_t>key

