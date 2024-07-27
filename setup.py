#!/usr/bin/env python3
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Get the python version as a pair of numbers (i.e. 3.8 becomes "38")
import sys

ext_modules = [
    Pybind11Extension("pyenskog",
                      ["pyenskog.cpp"],
                      ),
]

setup(
    name='pyenskog',
    version='0.2',
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
