Python utilities for MagAO-X interface
======================================

This is the home for Python components of the system that depend on specific
details of how MagAO-X is implemented and thus should evolve in lockstep
with the rest of the system.

The top-level Makefile target ``python_install`` installs a copy of the
``magaox`` package in this directory into the Python install at
``/opt/conda/bin/python`` (i.e. the environment used on MagAO-X). This
can be overridden by supplying the ``PYTHON=`` Makefile variable.

For, example ``make python_install PYTHON=/home/user/bin/python`` would
install the MagAO-X software for a hypothetical user's Python in their
home directory.