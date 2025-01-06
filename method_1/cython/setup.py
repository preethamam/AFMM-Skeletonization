from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

if sys.platform.startswith("win"):
    openmp_arg = '/openmp'
else:
    openmp_arg = '-fopenmp'

ext_modules = [
    Extension(
        "skelgrad",
        ["skelgrad.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    )
]

setup(
    name='skelgrad',
    ext_modules=cythonize(ext_modules, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
    }),
    zip_safe=False,
)