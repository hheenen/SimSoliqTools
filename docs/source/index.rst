.. SimSoliq documentation master file, created by
   sphinx-quickstart on Wed Feb 16 23:08:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SimSoliq's documentation!
====================================

This project is a collective of tools for the analysis and post processing of 
(molecular dynamics) simulations of solid/liquid interfaces

``SimSoliqTools`` is organized as a python (2.7/3.4) module incl. executive 
scripts

- i/o routines for data collection of MD output (VASP and generic ASE only at 
  the moment)
- i/o scripts and functionality for preparation of singlepoint calculations 
  on MD trajectories for further analysis
- routines for analysis of energetics in MD trajectories
- routines for analysis of the water structure in MD trajectories
- routines to track adsorbates on the surface during the AIMD simulations

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
