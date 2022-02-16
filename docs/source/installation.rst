Installation
============

Code
----

Clone the repo to your preferred location

.. code-block:: console

    $ git clone https://github.com/hheenen/SimSoliqTools.git
    $ pip install -e SimSoliqTools

Ensure proper funtionality by testing the unit tests:

.. code-block:: console

    $ cd SimSoliqTools/tests
    $ python -m unittest discover .

Documentation
-------------

Use sphinx to compile the documentation

.. code-block:: console
   
    $ cd docs/build
    $ make html


Dependencies
------------

SimSoliqTools relies on a few python packages, please be sure to have 
available:

- numpy
- matplotlib
- ASE (atomic simulation environment)
- python-sphinx (for documentation)
