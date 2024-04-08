"""
Created on Sept 2023
Code by Fredy Vides
For Paper, "Dynamic financial processes identification using sparse regressive 
reservoir computers"
by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: Fredy Vides
"""

def SpaRCSim(w,x0,tp,N):
    from numpy import zeros
    from NLMap import NLMap
    
    lx = len(x0)
    r = zeros((lx,N+1))
    r[:,0] = x0
        
    for k in range(N):
        r[:,k+1] = w@NLMap(r[:,k],tp)
                
    return r