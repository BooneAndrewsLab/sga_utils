# -*- coding: utf-8 -*-

import numpy
from setuptools import setup, find_packages, Extension


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    lic = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

correlation_module = Extension('sga.toolbox._c_impl', 
        sources=['sga/toolbox/src/c_impl.i', 'sga/toolbox/src/correlation.c', 'sga/toolbox/src/table_norm.c', 'sga/toolbox/src/safe.c'],
        include_dirs = [numpy_include],
        swig_opts=['-threads', '-modern', '-outdir', 'sga/toolbox/'],
        libraries = ['gsl', 'gslcblas','m'],
        extra_compile_args = ["-O3"],
    )

console_scripts = [
    'sga-similarity=sga.similarity:main',
    'sga-safe=sga.safe:main'
]

setup(
    name='sga',
    version='0.1.0',
    description='SGA Utilities',
    install_requires=required,
    long_description=readme,
    author='Matej Usaj',
    author_email='usaj.m@utoronto.ca',
    url='https://github.com/usajusaj/sga_utils',
    license=lic,
    ext_modules = [correlation_module],
    packages=find_packages(exclude=('tests', 'docs')),
    entry_points = {
        'console_scripts': console_scripts,
    }
)

