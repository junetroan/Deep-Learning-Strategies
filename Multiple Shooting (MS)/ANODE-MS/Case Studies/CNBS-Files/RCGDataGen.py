"""
Created on Nov 2023
Code by Fredy Vides
For Paper, "Dynamic financial processes identification using sparse regressive 
reservoir computers"
by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: doctor
"""

def RCGDataGen(data,input_indices,output_indices,L,S,ss,tp,filtering=False):
    from numpy import zeros
    from scipy.linalg import hankel
    from RFactor import RFactor
    from RNLMap import RNLMap
    from scipy.signal import savgol_filter
    
    
    def DataGenerator(X,R,tp,r):
        Y = zeros((r.shape[0],R))
        for j in range(R):
            Y[:,j] = RNLMap(r,X[:,j],tp)
        return Y
    
    input_data = data[input_indices,0:S:ss]
    output_data = data[output_indices,0:S:ss]
    
    if filtering:
        for k in range(input_data.shape[0]):
            input_data[k,:] = savgol_filter(input_data[k,:], 5, 2)
        
        for j in range(output_data.shape[0]):
            output_data[j,:] = savgol_filter(output_data[j,:], 5, 2)
    
    s = input_data.shape
    sL = s[0]*L
    S = s[1]
    R = S-L+1
    
    r = RFactor(s[0], L, tp)
    
    Input_Ldata = zeros((sL,R))
    for k in range(s[0]):
        Input_Ldata[k*L:(k+1)*L,:] = hankel(input_data[k,:L],input_data[k,(L-1):S])
        
    D_input = DataGenerator(Input_Ldata,R,tp,r)
    
    s = output_data.shape
    sL = s[0]*L
    S = s[1]
    R = S-L+1
    
    D_output = zeros((sL,R))
    for k in range(s[0]):
        D_output[k*L:(k+1)*L,:] = hankel(output_data[k,:L],output_data[k,(L-1):S])
    
            
    return D_input,D_output,r