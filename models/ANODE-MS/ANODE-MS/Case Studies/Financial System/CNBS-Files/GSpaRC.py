"""
Created on Oct 2023
GSpaRC  Generalized sparse reservoir computer identifier
   Code by Fredy Vides
   For Paper, "Dynamic financial processes identification using sparse regressive 
   reservoir computers"
   by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: Fredy Vides
"""

def GSpaRC(data,input_indices,output_indices,L,S,ss,tp,nz,tol,delta,filtering = False):
    from spsolver import spsolver
    from RCGDataGen import RCGDataGen
    
    s = data.shape
    R = s[1]-L+1
    
    D_input,D_output,r = RCGDataGen(data,input_indices,output_indices,L,S,ss,tp,filtering)
    
    w0 = spsolver(D_input.T,D_output.T,R,"sparse",nz,tol,delta).T
    return w0,D_input,D_output,r