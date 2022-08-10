from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(Extension(
        "ion_trap",
        sources=["ion_trap.pyx", "ion_trap_lib.cpp"],
        language="c++",
        include_dirs=["."]
        ))
)