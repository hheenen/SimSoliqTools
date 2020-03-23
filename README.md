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

### Installation

Clone the repo to your preferred location
```bash
git clone https://github.com/hheenen/SimSoliqTools.git
```

Add to your ``.bashrc`` file a pythonpath to your repository, i.e.
```bash
export PYTHONPATH=path-to-your-repo/SimSoliqTools:$PYTHONPATH
```
(NOTE: proper installation scripts are coming)

Ensure proper funtionality by testing the unit tests:
```bash
cd SimSoliqTools/tests
python -m unittest discover .
```

### Documentation and AIP reference

TODO with Sphinx

### Dependencies

SimSoliqTools relies on a few python packages, please be sure to have 
available:
- numpy
- matplotlib
- ASE (atomic simulation environment)
- python-sphinx (for documentation -- TODO)
- CatKit or pyMatGen (for slab site analysis)


