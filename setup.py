# -*- coding: utf-8 -*-

import numpy
from setuptools import setup, find_packages, Extension


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

correlation_module = Extension('sga.toolbox._correlation', 
        sources=['sga/toolbox/src/correlation.i', 'sga/toolbox/src/correlation.c'],
        include_dirs = [numpy_include],
        swig_opts=['-modern', '-outdir', 'sga/toolbox/'],
        extra_compile_args = ["-O3"],
    )

setup(
    name='sga',
    version='0.1.0',
    description='SGA Utilities',
    install_requires=['nose'],
    long_description=readme,
    author='Matej Ušaj',
    author_email='usaj.m@utoronto.ca',
    url='https://github.com/usajusaj/sga_utils',
    license=license,
    ext_modules = [correlation_module],
    packages=find_packages(exclude=('tests', 'docs')),
)

