"""
Created on Sept 2023
Code by Fredy Vides
For Paper, "Dynamic financial processes identification using sparse regressive 
reservoir computers"
by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: Fredy Vides
"""

def RCDataGen(data,L,S,ss,tp):
    from numpy import zeros
    from scipy.linalg import hankel
    from RFactor import RFactor
    from RNLMap import RNLMap
    
    
    def DataGenerator(X,R,tp,r):
        Y = zeros((r.shape[0],R))
        for j in range(R):
            Y[:,j] = RNLMap(r,X[:,j],tp)
        return Y
    
    data = data[:,0:S:ss]
    s = data.shape
    sL = s[0]*L
    S = s[1]
    R = S-L+1
    
    r = RFactor(s[0], L, tp)
    
    Ldata = zeros((sL,R))
    for k in range(s[0]):
        Ldata[k*L:(k+1)*L,:] = hankel(data[k,:L],data[k,(L-1):S])
    
    D = DataGenerator(Ldata,R,tp,r)
            
    return D,r
