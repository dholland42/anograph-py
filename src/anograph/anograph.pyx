# distutils: language = c++

import cython
from libc.float cimport FLT_MAX, DBL_MAX
from libc.math cimport sqrt
from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from libcpp cimport bool

# import numpy and all it's C apis
import numpy as np
cimport numpy as np
np.import_array()

####################################################################################
# Custom Type Definitions (these are just cosmetic, and are personal preference)
####################################################################################

ctypedef np.float64_t float64_t
ctypedef np.uint64_t uint64_t
ctypedef bint boolean

####################################################################################
# Main AnoGraph Implementation
####################################################################################

cdef class AnoGraph:
    cdef HCM hcm
    cdef bool[:] rmask
    cdef bool[:] cmask

    def __cinit__(self, num_buckets: int = 2, bucket_size: int = 32):
        self.hcm = HCM(num_buckets, bucket_size)
        self.rmask = np.ones(bucket_size, dtype=np.bool)
        self.cmask = np.ones(bucket_size, dtype=np.bool)

    def score(self):
        return self._score()

    def add(self, source, target):
        self.hcm.add(source, target)

    def add_batch(self, source, target):
        self.hcm.add_batch(source, target)

    def clear(self):
        self.hcm.clear()

    cdef double _score(self):
        cdef double[:,:,:] arr = self.hcm.sketch
        cdef double score = -DBL_MAX
        cdef int i
        cdef double s
        for i in range(arr.shape[0]):
            s = anographscore(arr[i], self.rmask, self.cmask)
            if s > score:
                score = s
        return score

####################################################################################
# Higher-Dimension Count-Min Sketch
####################################################################################

cdef class HCM:
    cdef int a
    cdef int b
    cdef int p
    cdef int num_buckets
    cdef int bucket_size
    cdef double[:,:,:] sketch

    def __cinit__(self, num_buckets: int = 2, bucket_size: int = 32):
        self.sketch = np.zeros(shape=(num_buckets, bucket_size, bucket_size))
        self.num_buckets = num_buckets
        self.bucket_size = bucket_size
        self.a = 5
        self.b = 3
        self.p = 7

    def add(self, source, target):
        self._add(u64_hash(source), u64_hash(target))

    def add_batch(self, source, target):
        self._add_batch(vhash(source), vhash(target))

    cpdef void clear(self):
        self.sketch[:,:,:] = 0

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef void _add(self, uint64_t source, uint64_t target):
        cdef int i  # for looping
        cdef double[:,:,:] sketch = self.sketch
        cdef uint64_t source_idx
        cdef uint64_t target_idx
        cdef uint64_t num_buckets = self.num_buckets
        cdef uint64_t bucket_size = self.bucket_size
        cdef uint64_t a = self.a
        cdef uint64_t b = self.b
        cdef uint64_t p = self.p
    
        # insert the original items
        source_idx = (source + ((a * source + b) % p)) % bucket_size
        target_idx = (target + ((a * target + b) % p)) % bucket_size

        sketch[0, source_idx, target_idx] += 1

        for i in range(1, num_buckets):
            source_idx = (source_idx + ((a * source_idx + b) % p)) % bucket_size
            target_idx = (target_idx + ((a * target_idx + b) % p)) % bucket_size
            sketch[i, source_idx, target_idx] += 1


    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    cdef void _add_batch(self, np.ndarray[np.uint64_t, ndim=1] source, np.ndarray[np.uint64_t, ndim=1] target):
        cdef uint64_t l = source.shape[0]
        cdef uint64_t[:] source_map = source
        cdef uint64_t[:] target_map = target
        for l_i in range(l):
            self._add(source_map[l_i], target_map[l_i])

    @property
    def sketch(self):
        return np.array(self.sketch)

####################################################################################
# HELPER UTILITIES 
####################################################################################

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double anographscore(double[:,:] arr, boolean[:] rmask, boolean[:] cmask):
    rmask[:] = 1
    cmask[:] = 1
    cdef double s
    cdef int i
    cdef int j
    cdef (unsigned long, double) mr
    cdef (unsigned long, double) mc
    cdef double score = density(arr, rmask, cmask)
    cdef int ctr = rmask.shape[0] + cmask.shape[0]

    for i in range(ctr - 1):
        mr = minrow(arr, rmask, cmask)
        mc = mincol(arr, rmask, cmask)
        if mr[1] < mc[1]:
            rmask[mr[0]] = 0
            s = density(arr, rmask, cmask)
            if s > score:
                score = s
        else:
            cmask[mc[0]] = 0
            s = density(arr, rmask, cmask)
            if s > score:
                score = s

    return score


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (unsigned long, double) minrow(double[:, :] sketch, cbool[:] rmask, cbool[:] cmask) nogil:
    cdef double minval = DBL_MAX
    cdef unsigned long minidx = 0
    cdef double rval = DBL_MAX

    # for iterating over rows, columns
    cdef unsigned long r_idx
    cdef unsigned long c_idx

    # loop over all the rows
    for r_idx in range(rmask.shape[0]):
        # skip the rows that are already masked
        if rmask[r_idx]:
            # sum the row
            rval = 0
            for c_idx in range(cmask.shape[0]):
                # only include values in columns that are not masked
                if cmask[c_idx]:
                    rval += sketch[r_idx, c_idx]
            if rval < minval:
                minval = rval
                minidx = r_idx
    return (minidx, minval)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef (unsigned long, double) mincol(double[:, :] sketch, cbool[:] rmask, cbool[:] cmask) nogil:
    cdef double minval = DBL_MAX
    cdef unsigned long minidx = 0
    cdef double cval = DBL_MAX

    # for iterating over rows, columns
    cdef unsigned long r_idx
    cdef unsigned long c_idx

    # loop over all the columns
    for c_idx in range(cmask.shape[0]):
        # skip the columns that are already masked
        if cmask[c_idx]:
            # sum the column
            cval = 0
            for r_idx in range(rmask.shape[0]):
                # only include values in columns that are not masked
                if rmask[r_idx]:
                    cval += sketch[r_idx, c_idx]
            if cval < minval:
                minval = cval
                minidx = c_idx
    return (minidx, minval)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef double density(double[:,:] mat, cbool[:] rmask, cbool[:] cmask) nogil:
    cdef double out = 0
    cdef unsigned long r_idx
    cdef unsigned long c_idx

    cdef double numrows = 0
    cdef double numcols = 0

    cdef double denom = 0

    # loop over rows and columns, skipping masked ones
    for r_idx in range(rmask.shape[0]):
        if rmask[r_idx]:
            for c_idx in range(cmask.shape[0]):
                if cmask[c_idx]:
                    out += mat[r_idx, c_idx]
                    denom += 1
    if denom == 0:
        return 0
    return out / sqrt(denom)

# siphash, but bounded to uint64
#
# NOTE: The output of this function will change from run to run. To make
# it deterministic, set the PYTHONHASHSEED environment variable.
def u64_hash(x):
    return (hash(x) + (1<<64)) & 0xffffffffffffffff

# vectorized hash function for use on numpy arrays
vhash = np.vectorize(u64_hash, otypes=[np.uint64])

