import numpy as np
import matplotlib.pyplot as pl
from matplotlib.pyplot import rc
import scipy
import os
import re
import math
import argparse
from matplotlib.colors import LogNorm
import matplotlib.ticker
import h5py
from EoS_HRG.full_EoS import find_param, isentropic

###############################################################################
__doc__ = """Analyse the TmunuTAU.hdf files and plot data"""
###############################################################################
parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
        '--folder', type=str, default='./',
        help='folder containing the TmunuTAU.dat file to analyse')
parser.add_argument(
        '--DTAU', type=float, default=0.1,
        help='bin in time tau [fm/c]')
parser.add_argument(
        '--DX', type=float, default=1.,
        help='bin in X [fm]')
parser.add_argument(
        '--DY', type=float, default=1.,
        help='bin in Y [fm]')
parser.add_argument(
        '--DETA', type=float, default=0.25,
        help='bin in spacetime rapidity eta')
parser.add_argument(
        '--energy', type=float, default=200,
        help='\sqrt(s_{NN}) [GeV]')
args = parser.parse_args()

# info about the grid
DTAU = args.DTAU
DX = args.DX
DY = args.DY
DETA = args.DETA
# path where to look for files
folder = args.folder
# collision info
energy = args.energy
system = 'AuAu{}GeV'.format(energy)

# settings for plots
SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30
rc('axes', linewidth=3) # width of axes
font = {'family' : 'Arial',
        'size' : MEDIUM_SIZE,
        'weight' : 'bold'}
rc('font', **font)  # controls default text sizes
rc('axes', titlesize=MEDIUM_SIZE, titleweight='bold')     # fontsize of the axes title
rc('axes', labelsize=MEDIUM_SIZE, labelweight='bold')    # fontsize of the x and y labels
rc('xtick', labelsize=SMALL_SIZE, direction='in', top='True')    # fontsize of the tick labels
rc('xtick.major', size=7, width=3, pad=10)
rc('ytick', labelsize=SMALL_SIZE, direction='in', right='True')    # fontsize of the tick labels
rc('ytick.major', size=7, width=3, pad=10)
rc('legend', fontsize=SMALL_SIZE, title_fontsize=SMALL_SIZE, handletextpad=0.25)    # legend fontsize
rc('figure', titlesize=BIGGER_SIZE, titleweight='bold')  # fontsize of the figure title
rc('savefig', dpi=300, bbox='tight')

########################################################################
print('search TmunuTAU.hdf5 in this folder: {}'.format(folder))

########################################################################
# list of all quantities and there LateX description
list_quant = {'e':r'$\epsilon$ [GeV.fm$^{-3}$]','vx':r'$v_x$','vy':r'$v_y$','vz':r'$|v_z|$','Plong':r'$P_L$ [GeV.fm$^{-3}$]','Ptrans':r'$P_T$ [GeV.fm$^{-3}$]','nB':r'$n_B$ [fm$^{-3}$]','nQ':r'$n_Q$ [fm$^{-3}$]','nS':r'$n_S$ [fm$^{-3}$]','frac':r'$\epsilon_{QGP}/\epsilon_{tot}$','count':r'$N_{cells}$','T':r'$T$ [GeV]','muB':r'$\mu_B$ [GeV]','muQ':r'$\mu_Q$ [GeV]','muS':r'$\mu_S$ [GeV]'}

########################################################################
file = 'TmunuTAU.hdf5'
with h5py.File(folder+file, 'r') as f:
    data_dict = {}
    for ib,ximpact in enumerate(f.values()):
        data_tau = ximpact['coord']['tau']
        data_x = ximpact['coord']['x']
        data_y = ximpact['coord']['y']
        data_eta = ximpact['coord']['eta']
        data_dict.update({'e': ximpact['e']})
        data_dict.update({'vx': ximpact['v']['vx']})
        data_dict.update({'vy': ximpact['v']['vy']})
        data_dict.update({'vz': ximpact['v']['vz']})
        data_dict.update({'Plong': ximpact['Plong']})
        data_dict.update({'Ptrans': ximpact['Ptrans']})
        data_dict.update({'nB': ximpact['n']['nB']})
        data_dict.update({'nQ': ximpact['n']['nQ']})
        data_dict.update({'nS': ximpact['n']['nS']})
        data_dict.update({'frac': ximpact['frac']})
        data_dict.update({'nval': ximpact['nval']})
        data_dict.update({'T': ximpact['T']})
        data_dict.update({'muB': ximpact['mu']['muB']})
        data_dict.update({'muQ': ximpact['mu']['muQ']})
        data_dict.update({'muS': ximpact['mu']['muS']})

print('TmunuTAU.dat is read')

########################################################################
# unique points in \tau and \eta
taus = np.unique(data_tau)
etas = np.unique(data_eta)
########################################################################
# select data for plots
cond_eta0 = (data_eta == 0.) # midrapidity
cond_x0y0 = (data_x == 0.) & (data_y == 0.) # x = y = 0
cond_x0y0eta0 = cond_x0y0 & cond_eta0  # midrapidity and x = y = 0

########################################################################
def ticks_log(xmin,xmax):
    """
    ticks for log scale
    """
    imin = int(math.log10(xmin))
    imax = int(math.log10(xmax)+1.)
    
    ticks = []
    for n in range(imin,imax):
        nmin = pow(10,n)
        for i in [1,2,3,5]:
            xtick = nmin*i
            if(xtick>xmax): continue
            ticks.append(xtick)  
    return ticks

########################################################################
def plot_mid(tab,aver):
    """ 
    Plot quantities at \eta = 0 as a function of \tau
    """
    f,ax = pl.subplots(figsize=(10,7))

    # condition on x,y,eta
    if(aver=='eta0'):
        coord_cond = cond_eta0 # midrapidity
    if(aver=='x0y0eta0'):
        coord_cond = cond_x0y0eta0 # midrapidity and x = y = 0

    # plot each quantity according to the conditions
    for quant in tab:
        # result for each value of \tau
        result = np.zeros_like(taus)
        for i,tau in enumerate(taus):
            # select coordinates
            cond = (tau == data_tau) & coord_cond
            # select corresponding data
            data1 = data_dict[quant][cond]
            if(len(data1)>0):
                result[i] = np.mean(data1)
            
        if(aver=='eta0'):
            title = r'$\eta$ = 0'
        elif(aver=='x0y0eta0'):
            title = r'$\eta = x = y = 0$'
        # select data to plot when the result is nonzero
        non_zero = (result != 0.)
        ax.plot(taus[non_zero], result[non_zero], linewidth='5', label=list_quant[quant])
        ax.legend(bbox_to_anchor=(0.95, 0.4), loc='center right', borderaxespad=0., frameon=False)

    ax.set(xlabel=r'$\tau$ [fm/c]', title=title)
    #ax.set_ylim([0.001,None])
    ax.set_yscale('log')
    ax.set_xlim([0.1,20.])
    ax.set_xscale('log')
    ax.set_xticks(ticks_log(0.1,20.))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    pl.show()
    f.savefig(folder+system+'_'+aver+'.png')

########################################################################
def cells(aver):
    """
    calculate/average quantities in \tau, \eta plane
    """
    # initialize array
    result = dict(zip(list_quant.keys(),np.zeros((len(list_quant),len(taus),len(etas)))))
    # loop over unique taus
    for i,tau in enumerate(taus):
        cond_tau = (tau == data_tau) # select coordinate for tau
        # loop over unique etas
        for j,eta in enumerate(etas):
            cond_eta = (eta == data_eta) # select coordinate for eta
            if(aver=='x0y0'):
                cond = cond_tau & cond_eta & cond_x0y0
            else:
                cond = cond_tau & cond_eta

            for quant in list_quant:
                # just count the number of cells
                if(quant=='count'):
                    result[quant][i,j] = len(data_dict['e'][cond])
                # average quantities
                else:
                    data1 = data_dict[quant][cond]
                    if(len(data1)>0):
                        result[quant][i,j] = np.mean(data1)

    result_out = {'x': etas, 'y': taus}
    result_out.update(result)
    return result_out

########################################################################
def plot_tau_eta(aver):
    """
    Plot all quantities in the \tau \eta plane
    """
    data = cells(aver)
    print('   plots for each quant')

    # limits for plots
    xmin = -5
    xmax = 5
    ymin = 0.1
    ymax = 10

    # select only data that will appear on the plot
    condy = (ymin <= data['y']) & (data['y'] <= ymax) 
    condx = (xmin <= data['x']) & (data['x'] <= xmax)
    pdat_x = data['x'][condx]
    pdat_y = data['y'][condy]

    for quant in list_quant:
        f,ax = pl.subplots(figsize=(10,7))
        log = False
        if(quant=='e' or quant=='nB' or quant=='nQ' or quant=='Ptrans' or quant=='Plong'):
            log = True
        if(quant=='e'):
            contours = [0.1,0.5,1.,5.,10.,20.,50.,100.]
        elif(quant=='T'):
            contours = [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6]
        else:
            contours = 10

        # select data in z that will appear on the plot
        condz = np.zeros((len(data['y']),len(data['x'])),dtype=bool)
        for i,_ in enumerate(data['y']):
            for j,_ in enumerate(data['x']):
                condz[i,j] = condy[i] & condx[j]
        # extract data for plots
        pdat_z = np.reshape(data[quant][condz],(len(pdat_y),len(pdat_x)))

        if(quant=='vz'):
            pdat_z = 0.5*np.log((1.+pdat_z)/(1.-pdat_z))
            list_quant[quant] = 'rapidity $y$'

        if(log):
            data_nonzero = pdat_z[pdat_z > 0]
            ticks = ticks_log(data_nonzero.min(),data_nonzero.max())

            #im = ax.pcolormesh(data['x']-DETA/2.,data['y']-DTAU/2.,data['z'],norm=LogNorm(),cmap='gist_rainbow_r')
            im = ax.contourf(pdat_x,pdat_y,pdat_z,norm=LogNorm(),cmap='gist_rainbow_r',levels=ticks,extend='both')
            cbar = f.colorbar(im, ax=ax, extend="both", ticks=ticks)
        
            pcontours = ax.contour(pdat_x,pdat_y,pdat_z, contours, norm=LogNorm(), colors='black',alpha=0.5)
            ax.clabel(pcontours, inline=True, fontsize=8)
        else:
            #im = ax.pcolormesh(data['x']-DETA/2.,data['y']-DTAU/2.,data['z'],cmap='gist_rainbow_r')
            im = ax.contourf(pdat_x,pdat_y,pdat_z,30,cmap='gist_rainbow_r',extend='both')
            cbar = f.colorbar(im, ax=ax, extend="both")
        
            pcontours = ax.contour(pdat_x,pdat_y,pdat_z, contours, colors='black',alpha=0.5)
            ax.clabel(pcontours, inline=True, fontsize=8)
        """
        if(log):
            ticks = ticks_log(data[quant][np.nonzero(data[quant])].min(),data[quant][np.nonzero(data[quant])].max())
            #im = ax.pcolormesh(data['x']-DETA/2.,data['y']-DTAU/2.,data['z'],norm=LogNorm(),cmap='gist_rainbow_r')
            im = ax.contourf(data['x']-DETA/2.,data['y']-DTAU/2.,data[quant],norm=LogNorm(),cmap='gist_rainbow_r',levels=ticks,extend='both')
            cbar = f.colorbar(im, ax=ax, extend="both", ticks=ticks)
        
            pcontours = ax.contour(data['x']-DETA/2.,data['y']-DTAU/2.,data[quant], contours, norm=LogNorm(), colors='black',alpha=0.5)
            ax.clabel(pcontours, inline=True, fontsize=8)
        else:
            #im = ax.pcolormesh(data['x']-DETA/2.,data['y']-DTAU/2.,data['z'],cmap='gist_rainbow_r')
            im = ax.contourf(data['x']-DETA/2.,data['y']-DTAU/2.,data[quant],30,cmap='gist_rainbow_r',extend='both')
            cbar = f.colorbar(im, ax=ax, extend="both")
        
            pcontours = ax.contour(data['x']-DETA/2.,data['y']-DTAU/2.,data[quant], contours, colors='black',alpha=0.5)
            ax.clabel(pcontours, inline=True, fontsize=8)
        """
        
        cbar.ax.get_yaxis().labelpad = 30
        cbar.ax.set_ylabel(list_quant[quant], rotation=270)
        ax.set_yscale("log")
        ax.set_ylim([ymin,ymax])
        ax.set_xlim([xmin,xmax])
        ax.set_yticks(ticks_log(ymin,ymax))
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if(aver=='x0y0'):
            title = r'$x = y = 0$'
        else:
            title = 'all (x,y)'
        ax.set(xlabel=r'$\eta$',ylabel=r'$\tau$ [fm/c]',title=title)

        pl.show()
        f.savefig(f'{folder+system}_{quant}_tau_eta.png')

########################################################################
print('plot tau eta x0y0')
plot_tau_eta('x0y0')

########################################################################
# plot T-muB
print('plot T-muB')
f,ax = pl.subplots(figsize=(9,7))
cuts = (data_dict['T'] != 0)
h = ax.hist2d(data_dict['muB'][cuts],data_dict['T'][cuts],bins=200,norm=LogNorm())
pl.colorbar(h[3])
xmuB,xtemp = isentropic('nS0',39.8)
ax.plot(xmuB,xtemp, color='red')
ax.plot(0.1956,0.1575,'*', color='red')
ax.set(xlabel=r'$\mu_B$ [GeV]',ylabel=r'$T$ [GeV]',title=f'Au+Au $\sqrt{{s_{{NN}}}} = {energy}$ GeV')
ax.set_xlim(0.,0.5)
ax.set_ylim(0.,0.5)
f.savefig(folder+system+'_TmuB_plane.png')

########################################################################
print('plot x0y0eta0')
tab = ['e','Plong','Ptrans','nB','nQ','nS']
plot_mid(tab, 'x0y0eta0')

########################################################################
print('plot eta0')
tab = ['e','Plong','Ptrans','nB','nQ','nS']
plot_mid(tab, 'eta0')
