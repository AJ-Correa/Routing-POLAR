from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "heuristics",
        ["heuristics.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],  # Maximum optimization
    )
]

setup(
    name="heuristics",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)