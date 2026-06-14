from setuptools import setup, Extension
import pybind11
import sys

extra_compile_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]

ext_modules = [
    Extension(
        "bloom",
        [
            "src/bindings.cpp",
            "src/BloomFilter.cpp",
            "src/Vectorizer.cpp",
            "utils/xxhash.c",
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            "include",
            "utils",
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="bloom",
    ext_modules=ext_modules,
)

# python setup.py build_ext --inplace