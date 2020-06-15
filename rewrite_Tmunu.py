import numpy as np
import os
import re
import argparse
import h5py
from math import factorial
import cmath
from tqdm import tqdm
from sympy import diff, symbols, lambdify, solveset
from EoS_HRG.full_EoS import find_param

###############################################################################
__doc__ = """Average of Tmunu + calculation of thermodynamic quantities"""

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
        help='bin in spacetime rapidity eta []')
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
# b for evaluation of T,mu
bTmuB = args.bTmuB

########################################################################
path = os.getcwd()
print(f'\ncurrent direcory: {path}')

print(f'\nsearch TmunuTAU.dat or job_# in this folder: {folder}')

###############################################################################
def search_Tmunu(folder):
    """
    search TmunuTAU.dat files in folder
    """
    B = []
    files = []
    # try to find a match in the folders
    for fold in sorted(os.listdir(folder)):
        match = re.match(f'TmunuTAU_ISUB([0-9]+)_B([0-9]+)_{int(DETA*10):02d}.dat', fold)
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
        match = re.match(f'TmunuTAU_ISUB([0-9]+)_B([0-9]+)_{int(DETA*10):02d}.dat', fold)
        bbb = int(match.group(2)) # impact parameter
        # scan values of b
        for ib,xb in enumerate(B_final):
            if(xb==bbb/10.):
                files_final[ib].append(folder+fold)
                continue

    return B_final,files_final

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
        match = re.match(folder+f'job_([0-9]+)/TmunuTAU_ISUB([0-9]+)_B([0-9]+)_{int(DETA*10):02d}.dat', fold)
        bbb = int(match.group(3)) # impact parameter
        # scan values of b
        for ib,xb in enumerate(B_final):
            if(xb==bbb/10.):
                files_final[ib].append(fold)
                continue

########################################################################
def determinant(A):
    """
    Recursive calculation of the determinant of matrix A
    """
    if(A.shape == (2, 2)):
        return A[0,0]*A[1,1] - A[0,1]*A[1,0]
    else:
        det = 0.
        # iterate over row of matrix A
        for i in range(len(A)):
            sign = (-1.)**i

            # identify submatrix
            sublen = int(len(A)-1)
            As = np.zeros((sublen,sublen),dtype=type(A[0,0]))
            # count the number of row
            row = 0
            # iterate over row of matrix A
            for j in range(len(A)):
                # if row = i, skip
                if(i==j):
                    continue
                As[:,row] = A[1:,j]
                row += 1

            # determinant
            det += sign*A[0,i]*determinant(As)
    return det

###############################################################################
def init_Tmunu():
    """
    calculate the eigenvalues of T^{\mu \nu}
    Algebraic expressions for solve_Tmunu
    """
    T00 = symbols('T00')
    T01 = symbols('T01')
    T02 = symbols('T02')
    T03 = symbols('T03')
    T11 = symbols('T11')
    T12 = symbols('T12')
    T13 = symbols('T13')
    T22 = symbols('T22')
    T23 = symbols('T23')
    T33 = symbols('T33')

    Tmunu = np.array([[T00,T01,T02,T03],[T01,T11,T12,T13],[T02,T12,T22,T23],[T03,T13,T23,T33]])
    gmunu = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])

    # system to solve (x is the vector containing the eigenvalues)
    # ( T^{\mu \nu} - x g^{\mu \nu} ) u_\mu = 0
    x = symbols('x')
    tosolve = Tmunu - x*gmunu
    # calculate determinant of ( T^{\mu \nu} - x g^{\mu \nu} ) = characteristic polynomial
    det = determinant(tosolve)

    print(det)

    # we want det(x) = c0 + c1*x +c2*x**2 + c3*x**3 + c4*x**4
    # find coefficients of the characteristic polynomial
    coeff = 5*[0.]
    coeff[0] = det.subs(x,0)
    for order in range(1,5):
        det = diff(det, x, 1)
        coeff[order] = det.subs(x,0)/factorial(order)

    print(coeff[::-1])


###############################################################################
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

    #diff12 = abs(P1/P2-1.) # is P1 close to P2?
    #diff13 = abs(P1/P3-1.) # is P1 close to P3?
    #diff23 = abs(P2/P3-1.) # is P2 close to P3?

    diff12 = abs(P1-P2) # is P1 close to P2?
    diff13 = abs(P1-P3) # is P1 close to P3?
    diff23 = abs(P2-P3) # is P2 close to P3?

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
    # 3) T31*(-vx) + T32*(-vy) + (T33+e)*(-vz) = -T20

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

    # gamma factor
    gamma = 1./np.sqrt(1.-sum([vi**2. for vi in vel]))

    return e,Plong,Ptrans,vel,gamma

# for tests
"""
Tmunu_comp = [1140.4071, 1.9445, -5.6739, -1134.2047, 0.0033, -0.0097, -1.9339, 0.0282, 5.6431, 1128.0360]
solve_Tmunu(Tmunu_comp)
input('pause')
"""

###############################################################################
def density(cur,gamma):
    """
    calculate local densities
    """
    J0 = cur[0]
    J1 = cur[1]
    J2 = cur[2]
    J3 = cur[3]

    if(J0!=0.):
        vJ2 = (J1**2. + J2**2. + J3**2.)/J0**2
        if(vJ2<1.):
            gamma = 1./np.sqrt(1.-vJ2)

    return J0/gamma

###############################################################################
# now read the data file and convert from dat to Python dict
print("\nConverting files:")

########################################################################
# choose format to classify the data
dtype = [('coord', [(s, float) for s in ['tau','x','y','eta']]),
        ('e', float),
        ('v', [(s, float) for s in ['vx','vy','vz']]),
        ('Plong', float),
        ('Ptrans', float),
        ('n', [(s, float) for s in ['nB','nQ','nS']]),
        ('frac', float),
        ('nval',int)
        ]

###############################################################################
# open the output file and analyse the TmunuTAU files
with h5py.File(f'{folder}/TmunuTAU_{int(DETA*10):02d}.hdf5', 'w') as output:
    
###############################################################################
# loop over the impact parameter b
    for ib,xb in enumerate(B_final):
        
        print(f'\n  b = {xb} fm')
###############################################################################
# count the total number of lines for each impact parameter
# scan all ISUBS for this value of impact parameter
        count = 0
        for xfile in files_final[ib]:
            with open(xfile, 'r') as myfile:
                for line in myfile:
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

                    # find where these coordinates are located in data
                    itau = int(round(line_fort[0]/DTAU))-itaumin
                    ix = int(round(line_fort[1]/DX))-ixmin
                    iy = int(round(line_fort[2]/DY))-iymin
                    ieta = int(round(line_fort[3]/DETA))-ietamin
                    index = conv[itau,ix,iy,ieta]

# format is: tau,x,y,eta,T00,T01,T02,T03,T11,T12,T13,T22,T23,T33,JB0,JB1,JB2,JB3,JQ0,JQ1,JQ2,JQ3,JS0,JS1,JS2,JS3,frac

                    # for each cell calculate quantities
                    e,Plong,Ptrans,v,gamma = solve_Tmunu(line_fort[4:14])
                    nB = density(line_fort[14:18],gamma)
                    nQ = density(line_fort[18:22],gamma)
                    nS = density(line_fort[22:26],gamma)

                    # average Tmunu and currents
                    TMUNU[index] += line_fort[4:14]
                    JCUR[index,0] += line_fort[14:18]
                    JCUR[index,1] += line_fort[18:22]
                    JCUR[index,2] += line_fort[22:26]

                    data['nval'][index] += 1
                    data['frac'][index] += line_fort[26]

    
        print('         Preparing data for output')
        for index,nval in enumerate(data['nval']):
            # dividing quantities by number of values
            TMUNU[index] /= nval
            JCUR[index] /= nval
            data['frac'][index] /= nval

            # diagonalization of Tmunu
            e,Plong,Ptrans,v,gamma = solve_Tmunu(TMUNU[index])
            data['e'][index] = e
            data['Plong'][index] = Plong
            data['Ptrans'][index] = Ptrans
            data['v'][index] = tuple(v)

            # local densities
            nB = density(JCUR[index,0],gamma)
            nQ = density(JCUR[index,1],gamma)
            nS = density(JCUR[index,2],gamma)
            data['n'][index] = (nB,nQ,nS)

        output.create_dataset(f'{xb}', data=data)
            
print(f'\nTmunuTAU.dat are read and converted in {folder}TmunuTAU.hdf5')


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
    tau = float(line[0])
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
with h5py.File(f'{folder}/TmuTAU_{int(DETA*10):02d}.hdf5', 'w') as output:
    
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