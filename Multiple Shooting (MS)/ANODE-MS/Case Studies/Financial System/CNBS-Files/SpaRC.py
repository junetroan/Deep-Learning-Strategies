"""
Created on Sept 2023
SpaRC  Sparse reservoir computer identifier
   Code by Fredy Vides
   For Paper, "Dynamic financial processes identification using sparse regressive 
   reservoir computers"
   by F. Vides
@author: Fredy Vides
"""

def SpaRC(data,solver,L,S,ss,tp,nz,tol,delta):
    from scipy.linalg import pinv
    from spsolver import spsolver
    from RCDataGen import RCDataGen
    
    s = data.shape
    sL = s[0]*L
    R = s[1]-L+1
    
    D,r = RCDataGen(data,L,S,ss,tp)
    
    if solver == "pinv":
        w0 = D[:sL,1:]@pinv(D[:,:-1])
    else:
        w0 = spsolver(D[:,:-1].T,D[:sL,1:].T,R,solver,nz,tol,delta).T
    return w0,D,r