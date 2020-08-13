import re
import h5py
from tqdm import tqdm
from EoS_HRG.full_EoS import find_param
# import from __init__.py
from . import *

########################################################################
print(f'\nsearch TmunuTAU.dat or job_# in this folder: {folder}')

B_final,files_final = search_Tmunu(folder)

try:
    len(B_final)
except:
    # try to find a match in the folders as job_#
    jobs_dir = []
    for fold in sorted(os.listdir(folder)):
        match = re.match('job_([0-9]+)', fold)
        try:
            xjob = int(match.group(1)) # job number
            jobs_dir.append(fold)
        except:
            pass

    B = []
    files = []
    for fold in jobs_dir:
        match = re.match('job_([0-9]+)', fold)
        xjob = int(match.group(1)) # job number
        print(f'job = {xjob}')
        B_fold, files_fold = search_Tmunu(folder+fold+'/')
        try:
            for ib,bbb in enumerate(B_fold):
                B.append(bbb)
                for xfile in files_fold[ib]:
                    files.append(xfile)
        except:
            raise Exception('no files found in ',folder+fold+'/')

    B_final = np.unique(np.array(B,dtype=int)) # table containing each unique value of impact parameter
    files_final = list([[] for xb in B_final])
    # scan file names
    for fold in files:
        match = re.match(f'.+TmunuTAU_ISUB([0-9]+)_B([0-9]+)_{string_deta}.dat', fold)
        bbb = int(match.group(2)) # impact parameter
        # scan values of b
        for ib,xb in enumerate(B_final):
            if(xb==bbb/10.):
                files_final[ib].append(fold)
                continue

###############################################################################
# now read the data file and convert from dat to Python dict
print("\nConverting files to T,mu:")

########################################################################
# choose format to classify the data
dtype = [('coord', [(s, float) for s in ['tau','x','y','eta']]),
        ('T',float),
        ('mu', [(s, float) for s in ['muB','muQ','muS']]),
        ('nval',int)
        ]

########################################################################
def select_cells(line):
    # skip non central cells along eta-axis
    # but evaluate transverse plane when eta=0
    #tau = float(line[0])
    x = float(line[1])
    y = float(line[2])
    eta = float(line[3])
    if(eta!=0 and x==0 and y==0):
        return True
    elif(eta==0):
        return True
    else:
        return False

###############################################################################
# open the output file and analyse the TmuTAU files
with h5py.File(f'{folder}/TmuTAU_{string_deta}.hdf5', 'w') as output:
    
###############################################################################
# loop over the impact parameter b
    for ib,xb in enumerate(B_final):
        
        if(xb != bTmuB):
            continue

        print(f'\n  b = {xb} fm')
###############################################################################
# count the total number of lines for each impact parameter
# scan all ISUBS for this value of impact parameter
        count = 0
        for xfile in files_final[ib]:
            with open(xfile, 'r') as myfile:
                for line in myfile:
                    line_fort = line.split()
                    if(select_cells(line_fort[0:4])):
                        count += 1
            
        print("     Total number of cells:", count)
                 
###############################################################################
# count the number of unique coordinates
        print("     Calculating unique cells")
        data_coord = np.zeros(count,dtype=[(s, float) for s in ['tau','x','y','z']])   
        i = 0
        for xfile in files_final[ib]:
            with open(xfile, 'r') as myfile:
                for line in myfile:
                    line_fort = line.split()
                    if(select_cells(line_fort[0:4])):
                        data_coord[i] = tuple([float(val) for val in line_fort[0:4]])
                        i += 1
         
        # just keep coordinate values (tau,x,y,z) once for the output
        x_unique = np.unique(data_coord)
        del(data_coord)
        print("     Number of cells to output:", len(x_unique))
        
###############################################################################
        # size of grid
        itaumin = int(round(np.amin(np.array([x[0] for x in x_unique]))/DTAU))
        itaumax = int(round(np.amax(np.array([x[0] for x in x_unique]))/DTAU))
        NTAU = itaumax-itaumin+1
        ixmin = int(round(np.amin(np.array([x[1] for x in x_unique]))/DX))
        ixmax = int(round(np.amax(np.array([x[1] for x in x_unique]))/DX))
        NX = ixmax-ixmin+1
        iymin = int(round(np.amin(np.array([x[2] for x in x_unique]))/DY))
        iymax = int(round(np.amax(np.array([x[2] for x in x_unique]))/DY))
        NY = iymax-iymin+1
        ietamin = int(round(np.amin(np.array([x[3] for x in x_unique]))/DETA))
        ietamax = int(round(np.amax(np.array([x[3] for x in x_unique]))/DETA))
        NETA = ietamax-ietamin+1
        # print(itaumin,itaumax,ixmin,ixmax,iymin,iymax,ietamin,ietamax)

        # for each unique coordinate, record the associated values in itau,ix,iy,ieta
        conv = np.zeros((NTAU,NX,NY,NETA),dtype=int)
        for index,x in enumerate(x_unique):
            itau = int(round(x[0]/DTAU))-itaumin
            ix = int(round(x[1]/DX))-ixmin
            iy = int(round(x[2]/DY))-iymin
            ieta = int(round(x[3]/DETA))-ietamin
            conv[itau,ix,iy,ieta] = index
        
        # initialize the data array with the specified datatype
        data = np.zeros(len(x_unique),dtype=dtype)
        TMUNU = np.zeros((len(x_unique),10))
        JCUR = np.zeros((len(x_unique),3,4))
        data['coord'] = x_unique
        del(x_unique)
        
        print("     Reading data and averaging")
        # loop over ISUB
        for xfile in files_final[ib]:
            print('      - ',xfile)
            # determine number of lines
            with open(xfile, 'r') as myfile:
                for line in myfile:
                    list_line = line.split() # convert to list
                    line_fort = np.array([float(val) for val in list_line]) # convert from string to float

                    if(select_cells(line_fort[0:4])):
                        pass
                    else:
                        continue

                    # find where these coordinates are located in data
                    itau = int(round(line_fort[0]/DTAU))-itaumin
                    ix = int(round(line_fort[1]/DX))-ixmin
                    iy = int(round(line_fort[2]/DY))-iymin
                    ieta = int(round(line_fort[3]/DETA))-ietamin
                    index = conv[itau,ix,iy,ieta]

# format is: tau,x,y,eta,T00,T01,T02,T03,T11,T12,T13,T22,T23,T33,JB0,JB1,JB2,JB3,JQ0,JQ1,JQ2,JQ3,JS0,JS1,JS2,JS3,frac

                    # for average quantities
                    TMUNU[index] = line_fort[4:14]
                    JCUR[index,0] = line_fort[14:18]
                    JCUR[index,1] = line_fort[18:22]
                    JCUR[index,2] = line_fort[22:26]
                    data['nval'][index] += 1
    
        print('         Preparing data for output')
        for index,nval in enumerate(tqdm(data['nval'])):
            # dividing quantities by number of values
            TMUNU[index] /= nval
            JCUR[index] /= nval

            # diagonalization of Tmunu
            e,Plong,Ptrans,v,gamma = solve_Tmunu(TMUNU[index])

            # local densities
            nB = density(JCUR[index,0],gamma)
            nQ = density(JCUR[index,1],gamma)
            nS = density(JCUR[index,2],gamma)

            if(e<0.001):
                continue

            # T,muB,muQ,muS
            Tmu = find_param('full',e=e,n_B=nB,n_Q=nQ,n_S=nS)
            data['T'][index] = Tmu['T']
            data['mu'][index] = (Tmu['muB'],Tmu['muQ'],Tmu['muS'])

        output.create_dataset(f'{xb}', data=data)
            
print(f'\nTmunuTAU.dat are read and converted in {folder}/TmuTAU.hdf5')