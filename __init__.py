import matplotlib.pyplot as pl
from matplotlib.pyplot import rc
from matplotlib.colors import LogNorm
import matplotlib.ticker
import math
import numpy as np
import argparse

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
if(int(10*energy) % 10 == 0):
    string_energy = f'{int(energy)}'
else:
    string_energy = f'{energy:.1f}'
    
system = f'AuAu{string_energy}GeV'

########################################################################
if(DETA<1):
    string_deta = f'0{int(DETA*100):02d}'.rstrip("0")
else:
    string_deta = f'{int(DETA*10):02d}'

########################################################################
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
def plot_mid(tab,aver,list_quant,data_dict,data_tau,taus,data_eta,etas,cond_x0y0,cond_eta0,cond_x0y0eta0):
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
    f.savefig(folder+system+'_'+aver+f'_DETA{string_deta}.png')

########################################################################
def cells(aver,list_quant,data_dict,data_tau,taus,data_eta,etas,cond_x0y0):
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
                elif(quant=='PLPT'):
                    data1 = data_dict['Plong'][cond]/data_dict['Ptrans'][cond]
                    if(len(data1)>0):
                        result[quant][i,j] = np.mean(data1)
                        if(result[quant][i,j]>10E+3):
                            result[quant][i,j] = 0.
                elif(quant=='muBT'):
                    data1 = data_dict['muB'][cond]/data_dict['T'][cond]
                    if(len(data1)>0):
                        result[quant][i,j] = np.mean(data1)
                        if(result[quant][i,j]>10E+2):
                            result[quant][i,j] = 0.
                else:
                    data1 = data_dict[quant][cond]
                    if(len(data1)>0):
                        result[quant][i,j] = np.mean(data1)

    result_out = {'x': etas, 'y': taus}
    result_out.update(result)
    return result_out

########################################################################
def plot_tau_eta(aver,list_quant,data_dict,data_tau,taus,data_eta,etas,cond_x0y0):
    """
    Plot all quantities in the \tau \eta plane
    """
    data = cells(aver,list_quant,data_dict,data_tau,taus,data_eta,etas,cond_x0y0)
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
        f.savefig(f'{folder+system}_{quant}_tau_eta_DETA{string_deta}.png')
