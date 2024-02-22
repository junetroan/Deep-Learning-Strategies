"""
Created on Sept 2023
SPSOLVER  Sparse linear least squares solver
   Code by Fredy Vides
   For Paper, "Dynamic financial processes identification using sparse regressive 
   reservoir computers"
   by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: Fredy Vides
"""

def spsolver(A,Y,L=100,mode = "ls",nz=100,tol=1e-2,delta=1e-2):
    from numpy.linalg import svd,norm,lstsq
    from numpy import zeros,dot,diag,argsort,inf
    N = A.shape[1]
    M = Y.shape[1]
    if nz<0:
        nz = N
    X=zeros((N,M))
    u,s,v=svd(A,full_matrices=0)
    rk=sum(s>tol)
    u=u[:,:rk]
    s=s[:rk]
    s=1/s
    s=diag(s)
    v=v[:rk,:]
    A=dot(u.T,A)
    Y=dot(u.T,Y)
    X0=dot(v.T,dot(s,Y))
    for k in range(M):
        w=zeros((N,))
        K=1
        Error=1+tol
        c=X0[:,k]
        x0=c
        ac=abs(c)
        f=argsort(-ac)
        N0=int(min(sum(ac[f]>delta),nz))
        while (K<=L) & (Error>tol):
            ff=f[:N0]
            X[:,k]=w
            if mode == "ls":
                c = lstsq(A[:,ff],Y[:,k],rcond=None)[0]
                X[ff,k]=c
            else:
                u,s,v=svd(A[:,ff],full_matrices=0)
                rk=sum(s>tol)
                u=u[:,:rk]
                s=s[:rk]
                s=1/s
                s=diag(s)
                v=v[:rk,:]
                c = dot(v.T,dot(s,dot(u.T,Y[:,k])))
            X[ff,k] = c
            Error=norm(x0-X[:,k],inf)
            x0=X[:,k]
            ac=abs(x0)
            f=argsort(-ac)
            N0=int(min(sum(ac[f]>delta),nz))
            K=K+1
    return X