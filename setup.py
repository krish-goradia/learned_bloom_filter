from setuptools import setup,Extension
import pybind11

ext_modules = [
    Extension(
        "bloom",
        ["src/bindings.cpp","src/BloomFilter.cpp","utils/xxhash.c"],
        include_dirs=[pybind11.get_include(),pybind11.get_include(user=True)],
        language="c++",
        extra_compile_args=["/std:c++17"]
    ),
]

setup(
    name="bloom",
    ext_modules=ext_modules,
)

#python setup.py build_ext --inplace