"""
Created on Oct 2023
Code by Fredy Vides
For Paper, "Dynamic financial processes identification using sparse regressive 
reservoir computers"
by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas

@author: Fredy Vides

Example:
    
    from ARNLFinancialSystemID import ARNLFinancialSystemID
    ARNLFinancialSystemID(1)
"""

def ARNLFinancialSystemID(number):
    from matplotlib.pyplot import figure,subplot,show,grid,tight_layout,figaspect,semilogy,spy
    from numpy import zeros, inf, linspace, count_nonzero, prod, vstack
    from numpy.linalg import norm
    from math import sqrt
    from time import time
    from SpaRC import SpaRC
    from SpaRCSim import SpaRCSim
    from RNLMap import RNLMap
    
    if number == 1:
        from FinancialSystem import FinancialSystem
        t,X = FinancialSystem()
        L = X.shape[0]
        s = X.shape[1]
        S = 6000
        print('L = ',L)
        print('Sampling %: '+str(100*S/L)+'%')
        l = 5
        tp = 2
        nz = int(s*l+(s*l)*(s*l+1)/2+1)
        t0 = time()
        w,data,r = SpaRC(X.T,"svd",l,S,1,tp,nz,5e-3,1e-3)
        print("Elapsed time: ",time()-t0)
        wr = w@r
        y = SpaRCSim(wr,data[:s*l,-1],tp,L-S+l-1)
        fig1 = figure(figsize = figaspect(0.4))
        ax00 = fig1.add_subplot(1,2,1, projection='3d')
        ax00.plot(X[:S,0],X[:S,1].T,X[:S,2],'b')
        ax00.set_xlabel('$x_1$')
        ax00.set_ylabel('$x_2$')
        ax00.set_zlabel('$x_3$')
        tight_layout()
        ax00 = fig1.add_subplot(1,2,2, projection='3d')
        ax00.plot(X[(S-l):,0].T,X[(S-l):,1].T,X[(S-l):,2].T,'g')
        ax00.plot(y[0,:].T,y[l,:].T,y[2*l,:].T,'r--')
        ax00.set_xlabel('$x_1$')
        ax00.set_ylabel('$x_2$')
        ax00.set_zlabel('$x_3$')
        tight_layout()
        show()
        fig1.savefig('../Figures/fig_FDS_ID_1.png',dpi=600,format='png')
        fig2 = figure()
        index0 = linspace(1,S-l,S-l)       
        index = linspace(S-l+1, L,L-S+l)
        ax0 = subplot(3,1,1)
        ax0.plot(index0,X[:(S-l),0],'b')
        ax0.plot(index,X[(S-l):,0],'g')
        ax0.plot(index,y[0,:],'r--')
        grid(color='k', linestyle='--', linewidth=0.5)
        ax0.set_xlabel('$t$')
        ax0.set_ylabel('$x_1$')
        tight_layout()
        ax0 = subplot(3,1,2)
        ax0.plot(index0,X[:(S-l),1],'b')
        ax0.plot(index,X[(S-l):,1],'g')
        ax0.plot(index,y[l,:],'r--')
        grid(color='k', linestyle='--', linewidth=0.5)
        ax0.set_xlabel('$t$')
        ax0.set_ylabel('$x_2$')
        tight_layout()
        ax0 = subplot(3,1,3)
        ax0.plot(index0,X[:(S-l),2],'b')
        ax0.plot(index,X[(S-l):,2],'g')
        ax0.plot(index,y[2*l,:],'r--')
        grid(color='k', linestyle='--', linewidth=0.5)
        ax0.set_xlabel('$t$')
        ax0.set_ylabel('$x_3$')
        tight_layout()
        show()
        fig2.savefig('../Figures/fig_FDS_ID_2.png',dpi=600,format='png')
        error = zeros((4,L-l+1))
        for j in range(l,L-1):
            Xj = vstack((vstack((X[(j-l):j,:1],X[(j-l):j,1:2])),X[(j-l):j,2:3]))[:,0]
            Xj1 = vstack((vstack((X[(j-l+1):(j+1),:1],X[(j-l+1):(j+1),1:2])),X[(j-l+1):(j+1),2:3]))[:,0]
            error[0,j-l] = norm((w@RNLMap(r,Xj,2))[:s*l:l]-(Xj1)[:s*l:l],inf)
        index = linspace(l, L,(L-l+1))
        fig2 = figure()
        axs = subplot(1,1,1)
        semilogy(index,error[0,:],'r')
        axs.set_xlabel('$t$')
        axs.set_ylabel('$r_t$')
        grid("True")
        tight_layout()
        show()
        fig2.savefig('../Figures/fig_FDS_ID_3.png',dpi=600,format='png')
        fig3 = figure()
        axs = subplot(2,2,1)
        spy(w)
        axs = subplot(2,2,2)
        spy(r)
        axs = subplot(2,1,2)
        spy(wr)
        show()
        fig3.savefig('../Figures/fig_FDS_ID_4.png',dpi=600,format='png')
        RMSE0 = sqrt(sum(error[0,:]**2)/len(error[0,:]))
        print("One step ahead prediction error estimate: ",RMSE0)
        X0 = X[(S-l):,:].T
        error = norm(y[:3*l:l,:]-X0)/sqrt(X0.shape[0]*X0.shape[1])
        print("RMSD estimate = ",error)
    elif number == 2:
        from FinancialSystem import FinancialSystem
        t,X = FinancialSystem(0.5,0.1,0.1,True)
        L = X.shape[0]
        s = X.shape[1]
        S = 800
        print('L = ',L)
        print('Sampling %: '+str(100*S/L)+'%')
        l = 5
        tp = 2
        nz = int(s*l+(s*l)*(s*l+1)/2+1)
        t0 = time()
        w,data,r = SpaRC(X.T,"svd",l,S,1,tp,nz,5e-3,1e-3)
        print("Elapsed time: ",time()-t0)
        wr = w@r
        y = SpaRCSim(wr,data[:s*l,-1],tp,L-S+l-1)
        fig1 = figure(figsize = figaspect(0.4))
        ax00 = fig1.add_subplot(1,2,1, projection='3d')
        ax00.plot(X[:S,0],X[:S,1].T,X[:S,2],'b')
        ax00.set_xlabel('$x_1$')
        ax00.set_ylabel('$x_2$')
        ax00.set_zlabel('$x_3$')
        tight_layout()
        ax00 = fig1.add_subplot(1,2,2, projection='3d')
        ax00.plot(X[(S-l):,0].T,X[(S-l):,1].T,X[(S-l):,2].T,'g')
        ax00.plot(y[0,:].T,y[l,:].T,y[2*l,:].T,'r--')
        ax00.set_xlabel('$x_1$')
        ax00.set_ylabel('$x_2$')
        ax00.set_zlabel('$x_3$')
        tight_layout()
        show()
        fig1.savefig('../Figures/fig_PFDS_ID_1.png',dpi=600,format='png')
        fig2 = figure()
        index0 = linspace(1,S-l,S-l)       
        index = linspace(S-l+1, L,L-S+l)
        ax0 = subplot(3,1,1)
        ax0.plot(index0,X[:(S-l),0],'b')
        ax0.plot(index,X[(S-l):,0],'g')
        ax0.plot(index,y[0,:],'r--')
        grid(color='k', linestyle='--', linewidth=0.5)
        ax0.set_xlabel('$t$')
        ax0.set_ylabel('$x_1$')
        tight_layout()
        ax0 = subplot(3,1,2)
        ax0.plot(index0,X[:(S-l),1],'b')
        ax0.plot(index,X[(S-l):,1],'g')
        ax0.plot(index,y[l,:],'r--')
        grid(color='k', linestyle='--', linewidth=0.5)
        ax0.set_xlabel('$t$')
        ax0.set_ylabel('$x_2$')
        tight_layout()
        ax0 = subplot(3,1,3)
        ax0.plot(index0,X[:(S-l),2],'b')
        ax0.plot(index,X[(S-l):,2],'g')
        ax0.plot(index,y[2*l,:],'r--')
        grid(color='k', linestyle='--', linewidth=0.5)
        ax0.set_xlabel('$t$')
        ax0.set_ylabel('$x_3$')
        tight_layout()
        show()
        fig2.savefig('../Figures/fig_PFDS_ID_2.png',dpi=600,format='png')
        error = zeros((4,L-l+1))
        for j in range(l,L-1):
            Xj = vstack((vstack((X[(j-l):j,:1],X[(j-l):j,1:2])),X[(j-l):j,2:3]))[:,0]
            Xj1 = vstack((vstack((X[(j-l+1):(j+1),:1],X[(j-l+1):(j+1),1:2])),X[(j-l+1):(j+1),2:3]))[:,0]
            error[0,j-l] = norm((w@RNLMap(r,Xj,2))[:s*l:l]-(Xj1)[:s*l:l],inf)
        index = linspace(l, L,(L-l+1))
        fig2 = figure()
        axs = subplot(1,1,1)
        semilogy(index,error[0,:],'r')
        axs.set_xlabel('$t$')
        axs.set_ylabel('$r_t$')
        grid("True")
        tight_layout()
        show()
        fig2.savefig('../Figures/fig_PFDS_ID_3.png',dpi=600,format='png')
        fig3 = figure()
        axs = subplot(2,2,1)
        spy(w)
        axs = subplot(2,2,2)
        spy(r)
        axs = subplot(2,1,2)
        spy(wr)
        show()
        fig3.savefig('../Figures/fig_PFDS_ID_4.png',dpi=600,format='png')
        RMSE0 = sqrt(sum(error[0,:]**2)/len(error[0,:]))
        print("One step ahead prediction error estimate: ",RMSE0)
        X0 = X[(S-l):,:].T
        error = sqrt(norm(y[:3*l:l,:]-X0)**2/(X0.shape[0]*X0.shape[1]))
        print("RMSE = ",error)
    print("Reduced coupling matrix nonzero entries proportion: ", count_nonzero(w)/prod(w.shape))
    print("Assembling matrix nonzero entries proportion: ", count_nonzero(r)/prod(r.shape))
    print("Coupling matrix nonzero entries proportion: ", count_nonzero(wr)/prod(wr.shape))


