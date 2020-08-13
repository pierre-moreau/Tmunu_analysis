import h5py
import matplotlib.pyplot as pl
# import from __init__.py
from . import *

########################################################################
print('search TmunuTAU.hdf5 in this folder: {}'.format(folder))

########################################################################
# list of all quantities and there LateX description
list_quant = {'e':r'$\epsilon$ [GeV.fm$^{-3}$]','vx':r'$v_x$','vy':r'$v_y$','vz':r'$|v_z|$','Plong':r'$P_L$ [GeV.fm$^{-3}$]','Ptrans':r'$P_T$ [GeV.fm$^{-3}$]','PLPT':r'$P_L/P_T$','nB':r'$n_B$ [fm$^{-3}$]','nQ':r'$n_Q$ [fm$^{-3}$]','nS':r'$n_S$ [fm$^{-3}$]','nQnB':r'$n_Q/n_B$','frac':r'$\epsilon_{QGP}/\epsilon_{tot}$','count':r'$N_{cells}$'}

########################################################################
file = f'TmunuTAU_{string_deta}.hdf5'
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
print('plot tau eta x0y0')
plot_tau_eta('x0y0',list_quant,data_dict,data_tau,taus,data_eta,etas,cond_x0y0)

########################################################################
print('plot x0y0eta0')
tab = ['e','Plong','Ptrans','nB','nQ','nS']
plot_mid(tab, 'x0y0eta0',list_quant,data_dict,data_tau,taus,data_eta,etas,cond_x0y0,cond_eta0,cond_x0y0eta0)

########################################################################
print('plot eta0')
tab = ['e','Plong','Ptrans','nB','nQ','nS']
plot_mid(tab, 'eta0',list_quant,data_dict,data_tau,taus,data_eta,etas,cond_x0y0,cond_eta0,cond_x0y0eta0)
