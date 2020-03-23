import os
import numpy as np


############## iterator functions to handle output ###################

def _nested_iterator(efunc, wpath0, fident, fnest):
    """
      helper function to iterate into nested folder structure
    
    """
    data = []
    wpath = wpath0
    while os.path.isfile('/'.join([wpath, fident])):
        data.append(efunc(wpath, fident))
        wpath += '/'+fnest
    dat = _concat_data(data)
    return(dat)


def _level_iterator(efunc, wpath0, fident):
    """
      helper function to iterate output files
    
    """
    # find and sort output files matching `fident`
    outfiles = [f for f in os.listdir(wpath0) if os.path.isfile(wpath0+'/'+f) \
            and np.all([f.find(s) != -1 for s in fident.split('*')])]
    outfiles.sort()

    # collect data from outfiles
    data = []
    for outf in outfiles:
        data.append(efunc(wpath0, outf))
    dat = _concat_data(data)
    return(dat)


def _concat_data(data):
    """
      helper function to stack output data
    
    """
    if type(data) == list:
        if type(data[0]) == dict:
            # type list of dicts
            return(_stack_dicts(data))
        else:
            # type list of values
            d = [dd for d in data for dd in d]
            return(d)
    else:
        raise TypeError("unexpected data-type for concatenation")


def _stack_dicts(data):
    """
      helper function to stack dictionaries holding 1d-numpy arrays
    
    """
    dat = data[0]
    for i in range(1,len(data)):
        for key in dat:
            dat[key] = np.hstack((dat[key],data[i][key]))
    return(dat)

