 #!/usr/bin/env python3

from distutils.core import setup, Extension
from Cython.Build import build_ext


ext = [
    Extension(
        'fpe',['fpe.pyx'],
        extra_compile_args=["-Ofast", "-v", "-march=native", "-Wall"]
        ),
    Extension(
        'utilities',['utilities.pyx'],
        extra_compile_args=["-Ofast", "-v", "-march=native", "-Wall"]
        )
    ]

ext_parallel = [
    Extension(
        'fpe', ['fpe.pyx'],
        extra_compile_args=["-Ofast", "-march=native", "-Wall", "-fopenmp"],
        extra_link_args=['-fopenmp', '-lm']
        ),
    Extension(
        'utilities',['utilities.pyx'],
        extra_compile_args=["-Ofast", "-v", "-march=native", "-Wall"]
        )
    ]

setup(
    name="FPE_PARALLEL",
    version="1.0",
    ext_modules=ext,
    cmdclass={'build_ext': build_ext}
    )
