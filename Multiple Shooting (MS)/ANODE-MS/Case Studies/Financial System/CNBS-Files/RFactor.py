"""
Created on Sept 2022
Code by Fredy Vides
For Paper, "Dynamic financial processes identification using sparse regressive 
reservoir computers"
by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: doctor
"""

def RFactor(s,L,tp,method = "random",tol = 1e-15, scale = 1e0):
    from numpy import zeros, where, vstack
    from NLMap import NLMap
    sL = s*L
    if method == "random":
        from numpy.random import randn
        z = scale*randn(sL,1)
        z = NLMap(z, tp)
        z[-1] = scale*randn()
    elif method == "prime":
        from PrimeGenerator import PrimeGenerator
        z0 = PrimeGenerator(sL+1,int(sL/2))
        z = NLMap(z0[:-1], tp)
        z[-1] = z0[-1]
    N = int((sL**(tp+1)-sL)/(sL-1))+1
    w0 = zeros((1,N))
    R = w0.copy()
    R[0,0] = 1
    for j in range(1,N):
        f = where(abs(z-z[j])<tol)
        if f[0][0] == j:
            w0[0,f] = 1
            R = vstack((R,w0/len(f[0])))
            w0[0,f] = 0
    return R