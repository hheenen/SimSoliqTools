SimSoliqTools
=============

## bundling tools for (AI)MD simulations of solid/liquid interfaces

This project is a collective of tools for the analysis and post processing of 
(molecular dynamics) simulations of solid/liquid interfaces

``SimSoliqTools`` is organized as a python (2.7/3.4) module incl. executive 
scripts

- i/o routines for data collection of MD output (VASP only at the moment)
- i/o scripts and functionality for preparation of singlepoint calculations 
  on MD trajectories for further analysis
- routines for analysis of energetics in MD trajectories
- routines for analysis of the water structure in MD trajectories
- routines to track adsorbates on the surface during the AIMD simulations

### First implementation avalability

- i/o routines structures and energies
- write routines for singlepoint calcs (TODO: add atom-manipulation)
- read routines for singlepoint calcs (energies, fermi-, vacuum-potential and workfunction)
- post-processing energies with averaging + standard plots
- post-processing density, individual trajectories + trajectory averaging + standard plots

### Installation

Clone the repo to your preferred location
```bash
git clone https://github.com/hheenen/SimSoliqTools.git
pip install -e SimSoliqTools
```

Ensure proper funtionality by testing the unit tests:
```bash
cd SimSoliqTools/tests
python -m unittest discover .
```

### Documentation

No documentation yet. Please check the examples

### Dependencies

SimSoliqTools relies on a few python packages, please be sure to have 
available:
- numpy
- matplotlib
- ASE (atomic simulation environment)
- python-sphinx (for documentation -- TODO)


