import json, os
#import numpy as np
from setuptools import setup, Extension


if __name__ == '__main__':
    """
    The first part compiles the broad package, the necessary
    information is given in the setup.json file.

    The second part compiles the files in the package written in Cython.
    As of now this is the cov.pyx file in the _helpers directory. Quite
    possibly you will need to change the c_flags.
    """

    with open('setup.json', 'r') as info:
        kwargs = json.load(info)

    '''
    # assuming gcc, set (aggressive) optimization flags,
    # to be appended to the default compile line
    c_flags = []
    c_flags.append("-O3")
    c_flags.append("-ffast-math")
    c_flags.append("-fopenmp")
    c_flags.append("-march=native")
    c_flags.append("-fPIC")
    c_flags.append("-fopt-info")

    # extra link flags are possible as well;
    #here we use the same as for compilation to link to libgomp
    ld_flags = c_flags

    ext1 = Extension("adsorptionanalyzer._helpers.get_distances",
                     sources=["adsorptionanalyzer/_helpers/get_distances.pyx"],
                     extra_compile_args=c_flags,
                     extra_link_args=ld_flags,
                     include_dirs=[np.get_include()]
    )
    '''

    setup(
        **kwargs,
        #ext_modules=[ext1, ext2]
        )
