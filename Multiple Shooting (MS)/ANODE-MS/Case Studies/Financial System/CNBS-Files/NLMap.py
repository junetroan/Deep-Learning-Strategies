"""
Created on Sept 2023
NLMap Nonlinear data mapping
   Code by Fredy Vides
   For Paper, "Dynamic financial processes identification using sparse regressive 
   reservoir computers"
   by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: Fredy Vides
"""

def NLMap(x,tp):
    from numpy import append, kron
    p = x
    q = p
    for k in range(tp-1):
        q = kron(x,q)
        p = append(p,q)
    p = append(p,1)
    return p