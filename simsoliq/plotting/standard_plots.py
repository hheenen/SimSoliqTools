"""
This module contains some helper function to produce 
standard plots for MD-analysis at solid/liquid interfaces

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.constants import golden_ratio, inch

def set_plotting_env(width=3.37,height=3.37/ golden_ratio *1.5/2,\
                    lrbt=[0.135,0.80,0.25,0.95],fsize=9.0): #,font='helvetica'):
    """
      DOC!!!
    """
    # set plot geometry
    rcParams['figure.figsize'] = (width, height) # x,y
    rcParams['font.size'] = fsize
    rcParams['figure.subplot.left'] = lrbt[0]  # the left side of the subplots of the figure
    rcParams['figure.subplot.right'] = lrbt[1] #0.965 # the right side of the subplots of the figure
    rcParams['figure.subplot.bottom'] = lrbt[2] # the bottom of the subplots of the figure
    rcParams['figure.subplot.top'] = lrbt[3] # the bottom of the subplots of the figure

    rcParams['xtick.top'] = True
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.right'] = True
    rcParams['ytick.direction'] = 'in'

    rcParams['legend.fancybox'] = False
    #rcParams['legend.framealpha'] = 1.0
    rcParams['legend.edgecolor'] = 'k'
    #matplotlibhelpers.set_latex(rcParams,font=font) #poster


def writefig(filename, folder='output',  write_eps=False):
    """
      wrapper for creating figures
    
    """
    # folder for output
    if not os.path.isdir(folder):
        os.makedirs(folder)

    fileloc = os.path.join(folder, filename)
    print("writing {}".format(fileloc+'.pdf'))
    plt.savefig(fileloc+'.pdf')
    
    if write_eps:
        print("writing {}".format(fileloc+'.eps'))
        plt.savefig(fileloc+'.eps')

    #TODO: add closing command for figures


    # TODO: refactor data below
def plot_engs_vs_time(filename, dat, tstart):
    label = {'ekin':r'E$_{\mathrm{kin}}$ (eV)',\
            'epot':r'E$_{\mathrm{pot}}$ (eV)',\
            'etot':r'E$_{\mathrm{tot}}$ (eV)',\
            'x':r'simulation time (ps)'}
    _plot_xe_vs_time(filename,dat,label,_nofunc,tstart=tstart)

def _nofunc(indat):
    return(indat)

def plot_av_engs_vs_time(filename, dat, tstart):
    label = {'ekin':r'$\langle$E$_{\mathrm{kin}} \rangle$ (eV)',\
            'epot':r'$\langle$E$_{\mathrm{pot}} \rangle$ (eV)',\
            'etot':r'E$_{\mathrm{tot}}$ (eV)',\
            'x':r'simulation time (ps)'}
    _plot_xe_vs_time(filename,dat,label,_running_av,tstart=tstart)

def _running_av(indat):
    y = np.cumsum(indat)/np.arange(1,indat.size+1)
    return(y)

def _plot_xe_vs_time(filename,dat,label,yfunc,ekeys=['ekin','epot','etot'],tstart=1e3):
    set_plotting_env(width=3.37,height=3.37,#/ golden_ratio *1.5/2,\
                   lrbt=[0.25,0.95,0.15,0.95],fsize=9.0)
    fig = plt.figure()
    n = 0; axes = []
    for e in ekeys:
        ax = fig.add_subplot(len(ekeys)*100+11+n)
        t = np.arange(tstart,dat[e].size)/1000.
        y = yfunc(dat[e][int(tstart):])
        ax.plot(t[::20],y[::20],color='k')
        ax.set_ylabel(label[e])
        axes.append(ax)
        n += 1
        # add line indicating AIMD traj restarts
        if 'trajlen' in dat:
            ttj = 0.0
            for tt in dat['trajlen']:
                ttj += tt
                ax.axvline(x=ttj/1000.,lw=0.5,ls=':',color='k')
    axes[-1].set_xlabel(label['x'])
    [axes[a].set_xticklabels([]) for a in range(len(axes)-1)]
    plt.subplots_adjust(hspace=0.0)
    writefig(filename)
