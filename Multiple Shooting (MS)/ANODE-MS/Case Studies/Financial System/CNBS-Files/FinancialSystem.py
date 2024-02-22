"""
Created on Sept 2023
Code by Fredy Vides
For Paper, "Dynamic financial processes identification using sparse regressive 
reservoir computers"
by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: Fredy Vides
"""

def FinancialSystem(s = 3.0, c = 0.1, e = 1.0, periodic = False):
    from numpy import arange
    from scipy.integrate import odeint
    
    def FDS(state, t, s, c, e):
        x, y, z = state
        
        dx = z + (y-s)*x
        dy = 1 - c*y - x**2
        dz = -x - e*z
        return [dx, dy, dz]
    
    p = (s, c, e)
    
    if periodic == False:
        y0 = [2.0 , 3.0, 2.0]
    else:
        y0 = [1.0 , 1.0, 1.0]
        
    t = arange(0.0, 120.0, 0.01)
    
    z = odeint(FDS, y0, t, p, rtol = 1e-10, atol = 1e-10)
    
    return t,z