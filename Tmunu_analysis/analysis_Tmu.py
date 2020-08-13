import h5py
from EoS_HRG.full_EoS import isentropic,full_EoS,full_EoS_nS0
import matplotlib.pyplot as pl
from matplotlib.animation import FuncAnimation, PillowWriter
# import from __init__.py
from . import *

########################################################################
# list of all quantities and there LateX description
list_quant = {'T':r'$T$ [GeV]','muB':r'$\mu_B$ [GeV]','muQ':r'$\mu_Q$ [GeV]','muS':r'$\mu_S$ [GeV]','muBT':r'$\mu_B/T$'}

########################################################################
file = f'TmuTAU_{string_deta}_{str(int(energy))}.hdf5'
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
########################################################################
########################################################################

dict_quant = ['Npart','Npart_err','T_{ch}','T_{ch}_err','\mu_{B}','\mu_{B}_err','\mu_{Q}','\mu_{Q}_err','\mu_{S}','\mu_{S}_err','\gamma_{S}','\gamma_{S}_err','s/n_{B}','s/n_{B}_err']

########################################################################
# PHSD HRG fits
dict_values = {}
dict_values.update({200:[370.40277777777777,0.9337147870435566,0.15329284334701443,0.0016672871325625754,0.033842310872574166,0.004221166814968546,-0.004902326591549128,0.001728839328368717,0.017217647147437922,0.0029200756551943704,0.9658086403688102,0.0056599660050613565,875.8577901872889,503.117564999355]})
dict_values.update({130:[369.525,0.7446988057825894,0.14936905995061361,0.0013599982667566318,0.04774817612366807,0.0036636658230462804,-0.005023693238986382,0.0013859252526131904,0.019723127060082002,0.0024167855267351705,0.9662040474571085,0.00474222064647678,384.85776027364466,74.7905681240017]})
dict_values.update({62.4:[368.072,0.5701332810827642,0.14767016580131717,0.001193872489568218,0.07693765235563974,0.0032377666961343546,-0.005994108994399006,0.001115878110161797,0.022901033563660723,0.0020148820856768675,0.96264727737012,0.004208403515031933,182.95529826439937,14.96888623951065]})
dict_values.update({39:[367.988125,0.4479199777836588,0.144,0.001,0.114,0.003,-0.006,0.001,0.025,0.002,0.965,0.004,108.09799529151087,4.582673391836714]})
dict_values.update({27:[367.5285,0.41969307647132914,0.14401032970579594,0.0009460853161920302,0.15709842669335344,0.0026597042219119738,-0.005899777932839079,0.000750664088719849,0.02880098871338571,0.0015508250746618778,0.9587647243075083,0.003606281210119855,71.45632973404149,2.289578066727852]})
dict_values.update({19.6:[368.1727272727273,0.39046461987284065,0.14586927135222474,0.0009820080460830882,0.2039484990109442,0.0027855809460629183,-0.006919590240314355,0.0007625670605642815,0.038557073270045095,0.0016672941177311024,0.9562266889123963,0.0037398625654445627,48.94323750596876,1.3294216669608891]})
dict_values.update({14.5:[368.6206666666667,0.3349083956315341,0.14446070141422782,0.0009161860137077288,0.2507838347530347,0.0025501256655418303,-0.007071055672351834,0.000633811843722562,0.04333190929923722,0.0015363230145252205,0.9503053745630446,0.0034440592381604906,37.143173459832695,0.8083366793417304]})
dict_values.update({11.5:[368.7803125,0.32666040659498363,0.1448331455384299,0.0010511414764387689,0.2997465399155122,0.0027685740688615235,-0.007543690644000567,0.0006280548468247521,0.05209494633034211,0.0016317806185365963,0.9440531992539055,0.0037310645415430943,28.28906745959534,0.5847418352064018]})
dict_values.update({9.2:[369.3310526315789,0.2935258017951511,0.14326320075900534,0.00116802200662329,0.3555219163042149,0.002880415328868291,-0.008065722463773295,0.0005878131235813877,0.06270427053073736,0.001703625813079812,0.9434710085081497,0.003889503856478682,21.77080272105203,0.4110800310711316]})
dict_values.update({7.7:[369.098,0.26183078955782124,0.139456615380197,0.0013019377321629305,0.4137786217951338,0.0027667881223481516,-0.008496239575861014,0.0005063108137417088,0.0732602701723897,0.001641848303120197,0.9523409114727222,0.0036781096113454592,17.23208721281841,0.28743556136191073]})

dict_freeze = dict(zip(dict_quant,dict_values[energy]))

########################################################################
# PHSD HRG fits nS0
dict_values_nS0 = {}
dict_values_nS0.update({200:[370.40277777777777,0.9337147870435566,0.15183182763458075,0.0016249989259462666,0.030910067170856967,0.003964234572767135,-0.00021470483883860927,0.0,0.018870464089788184,0.0,0.9753708362073013,0.0029128008844205855,726.5131218786574,266.35668822939607]})
dict_values_nS0.update({130:[369.525,0.7446988057825894,0.1477330808782195,0.0013041717168708472,0.04472568019226048,0.003476288501047361,-0.0004462746528578964,0.0,0.021229875257118672,0.0,0.9759822705996428,0.00228465067844863,362.3153370498173,52.81504317030637]})
dict_values_nS0.update({62.4:[368.072,0.5701332810827642,0.14506852851764684,0.0011037663286407179,0.07281304980820293,0.0030879861677716378,-0.0008919593194446151,0.0,0.025035890767738767,0.0,0.9766987182140671,0.0018601951418877327,182.12500544806042,11.819992664974308]})
dict_values_nS0.update({39:[367.988125,0.4479199777836588,0.14111605214384096,0.0008600387746846017,0.11010356386828121,0.0026187916861543786,-0.0014119708628353468,0.0,0.027771223148480195,0.0,0.9798128697330024,0.0014418855908782113,111.74472572819792,3.9895910031730666]})
dict_values_nS0.update({27:[367.5285,0.41969307647132914,0.13991437524842995,0.0007830334481464946,0.1529090310332363,0.0025876900464873037,-0.002090315694101202,0.0,0.03307088601439139,0.0,0.9803251242412934,0.0013539597868788555,75.13188624095847,2.042346180276803]})
dict_values_nS0.update({19.6:[368.1727272727273,0.39046461987284065,0.14214924253186914,0.000821538669039959,0.1984382430755847,0.0026682167696175235,-0.0031305032466168855,0.0,0.0432034202609557,0.0,0.9768649537149701,0.0014658703299964548,51.77607786813657,1.2015327453928695]})
dict_values_nS0.update({14.5:[368.6206666666667,0.3349083956315341,0.1400311405102112,0.0007729558810597031,0.24476431483505462,0.0024324921547793227,-0.0040163087758502206,0.0,0.05054211312732888,0.0,0.9763453258130127,0.0013456098548887407,39.754680783327345,0.7540144968164908]})
dict_values_nS0.update({11.5:[368.7803125,0.32666040659498363,0.13926365371439392,0.0008733105570345873,0.2932989782731806,0.002627404564930741,-0.005217442535469965,0.0,0.06051195053772078,0.0,0.9747759235706591,0.0014809463309701232,30.587633606328573,0.5580750404990473]})
dict_values_nS0.update({9.2:[369.3310526315789,0.2935258017951511,0.1372781586968595,0.0009924048849103,0.34826056949422046,0.0026725069050283767,-0.006673753789279605,0.0,0.0711873614223383,0.0,0.9755345823868256,0.0016056838661698825,23.655021110723435,0.39852804606238124]})
dict_values_nS0.update({7.7:[369.098,0.26183078955782124,0.1999999999990032,4.276709468922779e-05,0.5774501976498811,0.003234994180381756,-0.01820019355237443,0.0,0.1716950204266357,0.0,0.7932294876083545,0.003205562974012144,10.317861625143617,0.10710372841894691]})

dict_freeze_nS0 = dict(zip(dict_quant,dict_values_nS0[energy]))

########################################################################
# STAR data for GCE yields and ratios
dict_STAR_quant = ['T_{ch}_R','T_{ch}_R_err','T_{ch}_Y','T_{ch}_Y_err','\mu_{B}_R','\mu_{B}_R_err','\mu_{B}_Y','\mu_{B}_Y_err','\mu_{S}_R','\mu_{S}_R_err','\mu_{S}_Y','\gamma_{S}_Y_err','\gamma_{S}_R','\gamma_{S}_R_err','\gamma_{S}_Y','\gamma_{S}_Y_err']

dict_STAR_GCE = {}
dict_STAR_GCE.update({200:np.array([164.3,5.3,167.8,4.2,28.4,5.8,27.0,11.4,5.6,3.9,5.6,8.3,0.93,0.08,0.95,0.06])/1000.})
dict_STAR_GCE.update({62.4:np.array([160.3,4.9,164.3,3.6,69.8,5.6,69.2,11.4,16.7,3.3,15.8,6.8,0.86,0.06,0.91,0.05])/1000.})
dict_STAR_GCE.update({39:np.array([156.4,5.4,159.9,3.5,103.2,7.4,104.7,11.2,24.5,3.8,23.8,8.1,0.94,0.10,1.05,0.07])/1000.})
dict_STAR_GCE.update({27:np.array([155.0,5.1,159.8,3.0,144.4,7.2,151.9,9.3,33.5,3.6,36.7,6.0,0.98,0.09,1.09,0.05])/1000.})
dict_STAR_GCE.update({19.6:np.array([153.9,5.2,157.5,3.1,187.9,8.6,195.6,9.7,43.2,3.8,45.3,6.3,0.96,0.09,1.09,0.05])/1000.})
dict_STAR_GCE.update({11.5:np.array([149.4,5.2,150.6,3.2,287.3,12.5,292.5,12.6,64.5,4.7,66.0,7.6,0.92,0.09,1.00,0.06])/1000.})
dict_STAR_GCE.update({7.7:np.array([144.3,4.8,143.8,2.7,398.2,16.4,399.8,13.3,89.5,6.0,90.2,7.6,0.95,0.08,1.05,0.06])/1000.})

dict_STAR = dict(zip(dict_STAR_quant,dict_STAR_GCE[energy]))
dict_STAR.update({'\mu_{Q}_R':0.,'\mu_{Q}_R_err':0.})

########################################################################
# plot T-muB-muQ-muS
########################################################################

########################################################################
# calculate isentropic trajectories
data_EoS = full_EoS(dict_freeze_nS0['T_{ch}'],dict_freeze_nS0['\mu_{B}'],dict_freeze_nS0['\mu_{Q}'],dict_freeze_nS0['\mu_{S}'])
snB = data_EoS['s']/data_EoS['n_B']

data_EoS_nS0 = full_EoS_nS0(dict_freeze_nS0['T_{ch}'],dict_freeze_nS0['\mu_{B}'])
snB_nS0 = data_EoS_nS0['s']/data_EoS_nS0['n_B']

list_mu = ['muB','muQ','muS']

xmuB1,xtemp,xmuQ1,xmuS1 = isentropic('nS0',snB_nS0-dict_freeze_nS0['s/n_{B}_err'])
isent1 = dict(zip(list_mu,[xmuB1,xmuQ1,xmuS1]))
xmuB2,xtemp,xmuQ2,xmuS2 = isentropic('nS0',snB_nS0)
isent2 = dict(zip(list_mu,[xmuB2,xmuQ2,xmuS2]))
xmuB3,xtemp,xmuQ3,xmuS3 = isentropic('nS0',snB_nS0+dict_freeze_nS0['s/n_{B}_err'])
isent3 = dict(zip(list_mu,[xmuB3,xmuQ3,xmuS3]))

step_anim = 0.5
taus_anim = np.arange(0,10+step_anim,step_anim)
########################################################################
# plot T-muB-muQ-muS
def plot_Tmu_plane(step,quant,*args):

    latex_dict = dict(zip(list_mu,['\mu_{B}','\mu_{Q}','\mu_{S}']))

    # initialize plot
    if(step==None):
        global f,ax
        f,ax = pl.subplots(figsize=(9,7))

    cuts = (data_dict['T'] != 0)
    if(step!=None):
        cuts = cuts & (taus_anim[step] <= data_tau) & (data_tau < taus_anim[step]+step_anim)

    ymin,ymax = [0.,0.5]
    if(quant=='muB'):
        xmin,xmax = [0.,0.5]
    elif(quant=='muQ'):
        xmin,xmax = [-0.05,0.05]
    elif(quant=='muS'):
        xmin,xmax = [-0.1,0.2]

    ########################################################################
    # plot the PHSD data
    global h,cbar,text_tau
    if(step!=None):
        cbar.remove()
        h[3].remove()

    h = ax.hist2d(data_dict[quant][cuts],data_dict['T'][cuts],range=[[xmin, xmax], [ymin, ymax]],norm=LogNorm(),bins=100)
    cbar = f.colorbar(h[3])
    ax.annotate(f'$|\\eta| \\leq {DETA/2:4.1f}$', xy=(0.15, 0.55), xycoords='axes fraction', fontsize='15')
    # text with time tau
    if(step==None):
        text_tau = ax.annotate('', xy=(0.15, 0.60), xycoords='axes fraction', fontsize='15')
    else:
        #text_tau.set_text(f'$\\tau = {taus_anim[step]:4.2f}$ fm/c')
        text_tau.set_text(f'${taus_anim[step]:4.1f} \\leq \\tau < {taus_anim[step]+step_anim:4.1f}$ fm/c')

    # just update artists from plot if step is given
    if(step!=None):
        return h,cbar,text_tau

    ########################################################################
    # plot the isentropic trajectories
    ax.plot(isent2[quant],xtemp, color='red')
    ax.fill_betweenx(xtemp, isent1[quant], isent3[quant], alpha=0.6, color='red')

    ########################################################################
    # plot the STAR data
    ax.errorbar(dict_STAR[f'{latex_dict[quant]}_R'], dict_STAR['T_{ch}_R'], xerr=dict_STAR[f'{latex_dict[quant]}_R_err'], yerr=dict_STAR['T_{ch}_R_err'], markeredgecolor='red', color='white', fmt='*', markersize=12.5, label='STAR data (GCER)')
    ax.legend(title_fontsize=SMALL_SIZE, loc='upper right', borderaxespad=0., frameon=False)

    ########################################################################
    # plot the HRG fits
    ax.errorbar(dict_freeze[latex_dict[quant]], dict_freeze['T_{ch}'], xerr=dict_freeze[latex_dict[quant]+'_err'], yerr=dict_freeze['T_{ch}_err'], color='cyan', markerfacecoloralt='white', fmt='o', markersize=12.5, label='PHSD fit', fillstyle='right')
    ax.legend(title_fontsize=SMALL_SIZE, loc='upper right', borderaxespad=0., frameon=False)

    ########################################################################
    # plot the HRG fits nS0
    ax.errorbar(dict_freeze_nS0[latex_dict[quant]], dict_freeze_nS0['T_{ch}'], xerr=dict_freeze_nS0[latex_dict[quant]+'_err'], yerr=dict_freeze_nS0['T_{ch}_err'], color='magenta', markerfacecoloralt='white', fmt='o', markersize=12.5, label='PHSD fit\n'+r'$\langle n_S \rangle = 0$; $\langle n_Q \rangle = 0.4 \langle n_B \rangle$', fillstyle='right')
    ax.legend(title_fontsize=SMALL_SIZE, loc='upper right', borderaxespad=0., frameon=False)

    ########################################################################
    # set the labels
    ax.set(xlabel=f'${latex_dict[quant]}$ [GeV]',ylabel='$T$ [GeV]',title=f'Au+Au $\sqrt{{s_{{NN}}}} = {string_energy}$ GeV')
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    f.savefig(folder+system+f'_T{quant}_plane_DETA{string_deta}.png')
    
for quant in list_mu:
    print(f'plot T-{quant}')
    # initialize plot
    plot_Tmu_plane(None,quant)
    # create the animation
    anim = FuncAnimation(f, plot_Tmu_plane, frames=len(taus_anim), fargs=(quant,), repeat=True)
    # save the animation to a file
    anim.save(folder+system+f'_T{quant}_plane_DETA{string_deta}.mp4', fps=2)
