from countmin cimport HCountMin

import cython
from libc.float cimport FLT_MAX, DBL_MAX
from libcpp cimport bool as cbool

import numpy as np

cimport numpy as cnp

cnp.import_array()
DTYPE = np.int64

ctypedef cnp.float64_t float64_t
ctypedef cnp.uint64_t uint64_t


cdef class AnoGraph:
    cdef HCountMin hcm

    def __cinit__(self, height, width):
        self.hcm = HCountMin(height, width)

    cpdef add(self, source, target):
        self.hcm.add(source, target)

    cpdef add_batch(self, data):
        self.hcm.add_batch(data)

    cpdef getcount(self, source, target):
        return self.hcm.getcount(source, target)

    cpdef ((unsigned long, double), (unsigned long, double)) minrow(self):
        rmask = np.ones(self.hcm.width).astype(bool)
        cmask = np.ones(self.hcm.width).astype(bool)
        return minrow(self.hcm.sketch[0], rmask, cmask), mincol(self.hcm.sketch[0], rmask, cmask)
    
    @property
    def offsets(self):
        return (np.array(self.hcm.offset_a), np.array(self.hcm.offset_b))

    @property
    def sketch(self):
        return np.array(self.hcm.sketch)


cpdef pyminrow(arr, rmask, cmask):
    return minrow(arr, rmask, cmask)


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

