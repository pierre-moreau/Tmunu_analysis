import numpy as np
import os
import re
import h5py
from EoS_HRG.full_EoS import isentropic
import matplotlib.pyplot as pl
# import from __init__.py
from . import *

########################################################################
# list of all quantities and there LateX description
list_quant = {'T':r'$T$ [GeV]','muB':r'$\mu_B$ [GeV]','muQ':r'$\mu_Q$ [GeV]','muS':r'$\mu_S$ [GeV]','muBT':r'$\mu_B/T$'}

########################################################################
file = f'TmuTAU_{string_deta}.hdf5'
with h5py.File(folder+file, 'r') as f:
    data_dict = {}
    for ib,ximpact in enumerate(f.values()):
        data_tau = ximpact['coord']['tau']
        data_x = ximpact['coord']['x']
        data_y = ximpact['coord']['y']
        data_eta = ximpact['coord']['eta']
        data_dict.update({'T': ximpact['T']})
        data_dict.update({'muB': ximpact['mu']['muB']})
        data_dict.update({'muQ': ximpact['mu']['muQ']})
        data_dict.update({'muS': ximpact['mu']['muS']})

print('\nTmuTAU.dat is read')

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
# ratio s/n_B for each energy
dict_snB = {7.7:[18.8,4], 11.5:[26.2,5], 19.6:[39.6,6], 25:[51.4,10], 39:[74.7,15], 62.4:[114.1,20], 130:[200,30], 200:[420,40]}
########################################################################
# plot T-muB
print('plot T-muB-muQ-muS')
# initialize plots
plots = np.array([pl.subplots(figsize=(9,7)) for x in np.arange(3)])
f = plots[:,0]
ax = plots[:,1]

cuts = (data_dict['T'] != 0)

h0 = ax[0].hist2d(data_dict['muB'][cuts],data_dict['T'][cuts],range=[[0, 0.5], [0, 0.5]],norm=LogNorm(),bins=100)
f[0].colorbar(h0[3])
h1 = ax[1].hist2d(-data_dict['muQ'][cuts],data_dict['T'][cuts],range=[[0, 0.05], [0, 0.5]],norm=LogNorm(),bins=100)
f[1].colorbar(h1[3])
h2 = ax[2].hist2d(data_dict['muS'][cuts],data_dict['T'][cuts],range=[[0, 0.2], [0, 0.5]],norm=LogNorm(),bins=100)
f[2].colorbar(h2[3])

xmuB1,xtemp,xmuQ1,xmuS1 = isentropic('nS0',dict_snB[energy][0]-dict_snB[energy][1])
xmuB2,xtemp,xmuQ2,xmuS2 = isentropic('nS0',dict_snB[energy][0])
xmuB3,xtemp,xmuQ3,xmuS3 = isentropic('nS0',dict_snB[energy][0]+dict_snB[energy][1])

ax[0].plot(xmuB2,xtemp, color='red')
ax[0].fill_betweenx(xtemp, xmuB1, xmuB2, alpha=0.6, color='red')
ax[1].plot(-xmuQ2,xtemp, color='red')
ax[1].fill_betweenx(xtemp, -xmuQ1, -xmuQ2, alpha=0.6, color='red')
ax[2].plot(xmuS2,xtemp, color='red')
ax[2].fill_betweenx(xtemp, xmuS1, xmuS2, alpha=0.6, color='red')

ax[0].set(xlabel=r'$\mu_B$ [GeV]',ylabel=r'$T$ [GeV]',title=f'Au+Au $\sqrt{{s_{{NN}}}} = {string_energy}$ GeV')
ax[0].set_xlim(0.,0.5)
ax[0].set_ylim(0.,0.5)
f[0].savefig(folder+system+f'_TmuB_plane_DETA{string_deta}.png')
ax[1].set(xlabel=r'$-\mu_Q$ [GeV]',ylabel=r'$T$ [GeV]',title=f'Au+Au $\sqrt{{s_{{NN}}}} = {string_energy}$ GeV')
ax[1].set_xlim(0.,0.05)
ax[1].set_ylim(0.,0.5)
f[1].savefig(folder+system+f'_TmuQ_plane_DETA{string_deta}.png')
ax[2].set(xlabel=r'$\mu_S$ [GeV]',ylabel=r'$T$ [GeV]',title=f'Au+Au $\sqrt{{s_{{NN}}}} = {string_energy}$ GeV')
ax[2].set_xlim(0.,0.2)
ax[2].set_ylim(0.,0.5)
f[2].savefig(folder+system+f'_TmuS_plane_DETA{string_deta}.png')
