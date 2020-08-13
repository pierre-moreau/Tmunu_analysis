from math import factorial
from sympy import diff, symbols, lambdify, solveset
import numpy as np

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

init_Tmunu()