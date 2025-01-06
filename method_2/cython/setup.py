from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import platform

# Determine the OpenMP compiler and linker flags based on the platform
if platform.system() == "Darwin":  # macOS
    extra_compile_args = ['-Xpreprocessor', '-fopenmp']
    extra_link_args = ['-lomp']
else:  # Linux and Windows
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

extensions = [
    Extension(
        "afmm",
        ["afmm.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[numpy.get_include()],
        language="c++"
    )
]

setup(
    name='afmm',
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'initializedcheck': False,
        'nonecheck': False,
        'cdivision': True
    }),
    requires=['numpy', 'Cython']
)