SGA Utilities
========================

Swiss army knife (in the making) for working with SGA (Synthetic genetic array) data.

Install
------------------------
You will need python and gsl (GNU scientific library) headers (something like python-dev and gsl-dev on debian based systems) and SWIG to compile a few C functions.

All python dependencies are stated in requirements.txt and referenced from setup.py so pip can install them directly:

.. code:: bash

  pip install -r requirements.txt
  python setup.py install

or

.. code:: bash

  pip install sga_utils # run this one level above the cloned project
