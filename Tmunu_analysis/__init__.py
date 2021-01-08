"""
To process, analyze and plot data of the energy-momentum tensor and charge currents extracted from the Parton-Hadron-String Dynamics (PHSD) model.
Temperature and chemical potentials are obtained by using the EoS_HRG module.
"""

__version__ = '1.1.0'

import matplotlib.pyplot as pl
from matplotlib.pyplot import rc
from matplotlib.colors import LogNorm
import matplotlib.ticker
import math
import numpy as np
import argparse
import cmath
import os
import re

path = os.getcwd()
print(f'\ncurrent direcory: {path}')

###############################################################################
__doc__ = """Analyse the TmunuTAU files"""
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
parser.add_argument(
        '--bTmuB', type=float, default=2.,
        help='For which impact parameter evaluate T,mu from EoS?')
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

# b for evaluation of T,mu
bTmuB = args.bTmuB
# nucleus radius
Rad = 1.2*197**(1./3.)

########################################################################
if(DETA<1):
    string_deta = f'0{int(DETA*100):02d}'.rstrip("0")
else:
    string_deta = f'{int(DETA*10):02d}'

########################################################################
# Search and process TmunuTAU files
########################################################################
def search_Tmunu(folder):
    """
    search TmunuTAU.dat files in folder
    """
    B = []
    files = []
    # try to find a match in the folders
    for fold in sorted(os.listdir(folder)):
        match = re.match(f'TmunuTAU_ISUB([0-9]+)_B([0-9]+)_{string_deta}.dat', fold)
        try:
            xISUB = int(match.group(1)) # ISUB number
            xB = int(match.group(2)) # impact parameter
            B.append(xB/10.)
            files.append(fold) # add folder to the files_final list
            print(f'- ISUB = {xISUB} ; B = {xB/10.} fm')
        except:
            pass

    # if no files found here, return None
    if(len(B)==0):
        return None,None

    B_final = np.unique(np.array(B,dtype=int)) # table containing each unique value of impact parameter
    files_final = list([[] for xb in B_final])
    # scan file names
    for fold in files:
        match = re.match(f'TmunuTAU_ISUB([0-9]+)_B([0-9]+)_{string_deta}.dat', fold)
        bbb = int(match.group(2)) # impact parameter
        # scan values of b
        for ib,xb in enumerate(B_final):
            if(xb==bbb/10.):
                files_final[ib].append(folder+fold)
                continue

    return B_final,files_final

########################################################################
def solve_Tmunu(Tmunu_comp):
    """
    calculate the eigenvalues of T^{\mu \nu}
    Tmunu_comp is a list containing [T00,T01,T02,T03,T11,T12,T13,T22,T23,T33]
    """
    T00 = Tmunu_comp[0]
    T01 = Tmunu_comp[1]
    T02 = Tmunu_comp[2]
    T03 = Tmunu_comp[3]
    T11 = Tmunu_comp[4]
    T12 = Tmunu_comp[5]
    T13 = Tmunu_comp[6]
    T22 = Tmunu_comp[7]
    T23 = Tmunu_comp[8]
    T33 = Tmunu_comp[9]

    # coefficient of the characteristic polynomial
    # det(x) = c0 + c1*x +c2*x**2 + c3*x**3 + c4*x**4
    a = -1.
    b = T00 - 1.0*T11 - 1.0*T22 - 1.0*T33
    c = 0.5*T00*(2.0*T11 + 2.0*T22 + 2.0*T33) - 1.0*T01**2 - 1.0*T02**2 - 1.0*T03**2 + 0.5*T11*(-1.0*T22 - 1.0*T33) - 0.5*T11*(T22 + T33) + T12**2 + T13**2 - 1.0*T22*T33 + T23**2
    d = 1.0*T00*(1.0*T11*(T22 + T33) - 1.0*T12**2 - 1.0*T13**2 + 1.0*T22*T33 - 1.0*T23**2) - 1.0*T01*(1.0*T01*(T22 + T33) - 1.0*T02*T12 - 1.0*T03*T13) + 1.0*T02*(1.0*T01*T12 - 1.0*T02*T11 - 1.0*T02*T33 + 1.0*T03*T23) - 1.0*T03*(-1.0*T01*T13 - 1.0*T02*T23 + 1.0*T03*T11 + 1.0*T03*T22) - 1.0*T11*(T22*T33 - T23**2) + 1.0*T12*(T12*T33 - T13*T23) - 1.0*T13*(T12*T23 - T13*T22)
    e = 1.0*T00*(1.0*T11*(T22*T33 - T23**2) - 1.0*T12*(T12*T33 - T13*T23) + 1.0*T13*(T12*T23 - T13*T22)) - 1.0*T01*(1.0*T01*(T22*T33 - T23**2) - 1.0*T12*(T02*T33 - T03*T23) + 1.0*T13*(T02*T23 - T03*T22)) + 1.0*T02*(1.0*T01*(T12*T33 - T13*T23) - 1.0*T11*(T02*T33 - T03*T23) + 1.0*T13*(T02*T13 - T03*T12)) - 1.0*T03*(1.0*T01*(T12*T23 - T13*T22) - 1.0*T11*(T02*T23 - T03*T22) + 1.0*T12*(T02*T13 - T03*T12))

    D0 = c**2. -3.*b*d + 12*a*e
    D1 = 2.*c**3. - 9*b*c*d + 27.*(b**2.)*e + 27*a*d**2. - 72.*a*c*e

    pp =(8.*a*c-3.*b**2.)/(8*a**2.)
    qq = (b**3.-4.*a*b*c+8.*(a**2.)*d)/(8*a**3.)

    Q = ((D1+cmath.sqrt(D1**2.-4.*D0**3.))/2.)**(1./3.)
    S = 0.5*cmath.sqrt(-2./3.*pp+1./(3.*a)*(Q+D0/Q))

    e = abs(-b/(4.*a) + S + 0.5*np.sqrt(-4.*S**2. -2.*pp - qq/S))
    P1 = abs(-b/(4.*a) + S - 0.5*np.sqrt(-4.*S**2. -2.*pp - qq/S))
    P2 = abs(-b/(4.*a) - S + 0.5*np.sqrt(-4.*S**2. -2.*pp + qq/S))
    P3 = abs(-b/(4.*a) - S - 0.5*np.sqrt(-4.*S**2. -2.*pp + qq/S))

    #print(e,P1,P2,P3)

    diff12 = abs(P1/P2-1.) # is P1 close to P2?
    diff13 = abs(P1/P3-1.) # is P1 close to P3?
    diff23 = abs(P2/P3-1.) # is P2 close to P3?

    if(diff12<diff13 and diff12<diff23):
        Ptrans = (P1+P2)/2.
        Plong = P3
        #print('P1,P2')   
        #input('pause')   
    elif(diff13<diff12 and diff13<diff23):
        Ptrans = (P1+P3)/2.
        Plong = P2     
        #print('P1,P3') 
        #input('pause')   
    elif(diff23<diff12 and diff23<diff13):
        Ptrans = (P2+P3)/2.
        Plong = P1
        #print('P2,P3')

    # syste to solve to find velocity vector u^\mu
    # is ( T^{\mu \nu} - x g^{\mu \nu} ) u_\mu = 0 
    # x is replaced by energy density e

    # solution is u_\mu = \gamma (1,-vx,-vy,-vz)
    # only 3 equations are necessary to find vx,vy,vz
    # 1) (T11+e)*(-vx) + T12*(-vy) + T13*(-vz) = -T10
    # 2) T21*(-vx) + (T22+e)*(-vy) + T23*(-vz) = -T20
    # 3) T31*(-vx) + T32*(-vy) + (T33+e)*(-vz) = -T30

    vel = np.zeros(3)

    vel[0] = (e**2*T01 - T03*T13*T22 + T03*T12*T23 + T02*T13*T23 - T01*T23**2 - T02*T12*T33 + T01*T22*T33 + e*(-(T02*T12) - T03*T13 + T01*(T22 + T33)))/ \
       (e**3 - T13**2*T22 + 2*T12*T13*T23 - T11*T23**2 - e*(T12**2 + T13**2 - T11*T22 + T23**2) - T12**2*T33 + T11*T22*T33 + e*(T11 + T22)*T33 + \
        e**2*(T11 + T22 + T33))

    vel[1] = (e**2*T02 + T03*T12*T13 - T02*T13**2 - T03*T11*T23 + T01*T13*T23 + T02*T11*T33 - T01*T12*T33 + e*(-(T01*T12) - T03*T23 + T02*(T11 + T33)))/ \
       (e**3 - T13**2*T22 + 2*T12*T13*T23 - T11*T23**2 - e*(T12**2 + T13**2 - T11*T22 + T23**2) - T12**2*T33 + T11*T22*T33 + e*(T11 + T22)*T33 + \
        e**2*(T11 + T22 + T33))

    vel[2] = (e**2*T03 - T03*T12**2 + T02*T12*T13 + T03*T11*T22 - T01*T13*T22 - T02*T11*T23 + T01*T12*T23 + e*(-(T01*T13) + T03*(T11 + T22) - T02*T23))/ \
       (e**3 - T13**2*T22 + 2*T12*T13*T23 - T11*T23**2 - e*(T12**2 + T13**2 - T11*T22 + T23**2) - T12**2*T33 + T11*T22*T33 + e*(T11 + T22)*T33 + \
        e**2*(T11 + T22 + T33))

    #print(vel)

    return e,Plong,Ptrans,vel

########################################################################
def fourprod(vec1,vec2):
    """
    inner product of two four-vectors
    """
    return vec1[0]*vec2[0]-vec1[1]*vec2[1]-vec1[2]*vec2[2]-vec1[3]*vec2[3]

########################################################################
def tensprod(tens1,tens2):
    """
    inner product of two tensors
    """
    # metric tensor
    gmunu = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
    # Einstein summation
    prod = 0.
    for alpha in range(4):
        for beta in range(4):
            for mu in range(4):
                for nu in range(4):
                    prod += tens1[mu,nu]*tens2[alpha,beta]*gmunu[alpha,mu]*gmunu[beta,nu]

    return prod

########################################################################
def density(cur,vel):
    """
    calculate local densities from charge current
    gamma/vel is the Lorentz factor/velocity obtained by Landau condition
    """
    cur = np.array([val for val in cur])
    # gamma factor
    gamma = 1./np.sqrt(1.-sum([vi**2. for vi in vel]))
    # four velocity vector
    umu = gamma*np.array([1.,vel[0],vel[1],vel[2]])
    # local density
    nloc = fourprod(umu,cur)
    # dissipative term
    nmu = cur-nloc*umu
    # Inverse Reynolds number
    norm_nmu = fourprod(nmu,nmu)
    if(norm_nmu>0.):
        Reyn = np.sqrt(norm_nmu)/nloc
    else:
        Reyn = 0.

    return nloc,Reyn

########################################################################
def dissip_Tmunu(Tmunu_comp,e,Piso,vel):
    """
    calculate BVP and shear stress tensor from Tmunu
    and associated Reynolds number
    """
    T00 = Tmunu_comp[0]
    T01 = Tmunu_comp[1]
    T02 = Tmunu_comp[2]
    T03 = Tmunu_comp[3]
    T11 = Tmunu_comp[4]
    T12 = Tmunu_comp[5]
    T13 = Tmunu_comp[6]
    T22 = Tmunu_comp[7]
    T23 = Tmunu_comp[8]
    T33 = Tmunu_comp[9]
    # Tmunu
    Tmunu = np.array([[T00,T01,T02,T03],[T01,T11,T12,T13],[T02,T12,T22,T23],[T03,T13,T23,T33]])
    # gamma factor
    gamma = 1./np.sqrt(1.-sum([vi**2. for vi in vel]))
    # four velocity vector
    umu = gamma*np.array([1.,vel[0],vel[1],vel[2]])
    # metric tensor
    gmunu = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])

    # projection operators
    umuunu = np.zeros_like(Tmunu)
    Dmunu = np.zeros_like(Tmunu)
    for mu in range(4):
        for nu in range(4):
            umuunu[mu,nu] = umu[mu]*umu[nu]
            Dmunu[mu,nu] = gmunu[mu,nu] - umuunu[mu,nu]

    # bulk viscous pressure
    BVP = -1./3.*tensprod(Dmunu,Tmunu)-Piso
    ReynPI = np.sqrt(BVP**2.)/(e+Piso)

    # shear stress tensor
    pimunu = Tmunu -e*umuunu + (Piso+BVP)*Dmunu
    norm_pimunu = tensprod(pimunu,pimunu)
    if(norm_pimunu>0.):
        Reynpi = np.sqrt(norm_pimunu)/(e+Piso)
    else:
        Reynpi = 0.
    
    return Reynpi,ReynPI

########################################################################
# settings for plots
########################################################################
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
rc('animation',ffmpeg_path='/usr/bin/ffmpeg')

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
                # average quantities
                elif(quant=='nQnB'):
                    data1 = data_dict['nQ'][cond]/data_dict['nB'][cond]
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

        levels = 30
        if(quant=='nQnB'):
            levels = np.arange(0.0, 1.0, 0.1)
        if(quant=='PLPT'):
            levels = np.arange(0.5, 2.0, 0.1)

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
            im = ax.contourf(pdat_x,pdat_y,pdat_z,levels,cmap='gist_rainbow_r',extend='both')
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
