"""This custom setup is needed in order for uv to build countmin automatically."""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(name="countmin", sources=["src/anograph/countmin.pyx"]),
    Extension(name="anograph", sources=["src/anograph/anograph.pyx"]),
]

setup(
    ext_modules=cythonize(module_list=extensions),
    include_dirs=[numpy.get_include()]
)
