# distutils: language = c++

import numpy as np

cimport numpy as cnp

cnp.import_array()


cdef class CountMin:
    cdef unsigned long width
    cdef unsigned long height
    cdef unsigned long[:] offset_a
    cdef unsigned long[:] offset_b
    cdef double[:, :] sketch

    cpdef void add(self, long idx)
    cdef void _add_batch(self, cnp.ndarray[cnp.int64_t, ndim=1] inp)
    cpdef double getcount(self, int idx)
    cpdef void clear(self)


cdef class HCountMin:
    cdef unsigned long width
    cdef unsigned long height
    cdef unsigned long[:] offset_a
    cdef unsigned long[:] offset_b
    cdef double[:, :, :] sketch

    cpdef void add(self, long source, long target)
    cdef void _add_batch(self, cnp.ndarray[cnp.int64_t, ndim=2] inp)
    cpdef double getcount(self, long source, long target)
    cpdef void clear(self)

