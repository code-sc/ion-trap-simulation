from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension(
        "ion_trap_1d",
        sources = [
            "./ion_trap_1d.pyx",
            "./ion_trap_1d_lib.cpp"
        ],
        language="c++",
        include_dirs=["."],
        extra_compile_args=["-w"]
    ),
    Extension(
        "ion_trap_3d",
        sources = [
            "./ion_trap_3d.pyx",
            "./ion_trap_3d_lib.cpp"
        ],
        language="c++",
        include_dirs=["."],
        extra_compile_args=["-w"]
    )
]

setup(
    ext_modules=cythonize(extensions)
)