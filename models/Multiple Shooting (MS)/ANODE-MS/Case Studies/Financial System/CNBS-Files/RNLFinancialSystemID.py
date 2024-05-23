"""
Created on Dec 2023
Code by Fredy Vides
For Paper, "Dynamic financial processes identification using sparse regressive 
reservoir computers"
by F. Vides, I. B. R. Nogueira, E. Flores, L. Banegas
@author: Fredy Vides

Example:
    
    from RNLFinancialSystemID import RNLFinancialSystemID
    w0,D_input,D_output,r = RNLFinancialSystemID(1,[1,2],[0])
    w0,D_input,D_output,r = RNLFinancialSystemID(2,[1,2],[0])
    w0,D_input,D_output,r = RNLFinancialSystemID(3,list(set(range(15)).difference({0})),[0],filtering=True)
"""

def RNLFinancialSystemID(number,input_indices,output_indices,filtering = False):
    from GSpaRC import GSpaRC
    from RCGDataGen import RCGDataGen
    from matplotlib.pyplot import plot, semilogy, stem, subplot, xlabel, axis, grid, ylabel, show, figure, legend
    from numpy import where
    if number == 1:
        from FinancialSystem import FinancialSystem
        t,X = FinancialSystem()
        X = X.T
        w0,D_input_train,D_output_train,r = GSpaRC(X,input_indices,output_indices,2,500,1,2,(2*2)**2+2*2,1e-6,5e-2)
        D_input,D_output,r = RCGDataGen(X,input_indices,output_indices,2,X.shape[1],1,2)
        id_out=w0@D_input
        N = D_input.shape[1]
        T = D_input_train.shape[1]
        t = range(N)
        fig1 = figure()
        id_indices = where(abs(w0[0,:])>0)[0]
        labels = []
        for k in range(len(id_indices)):
            labels.append('$x_{'+str(id_indices[k])+'}$')
        subplot(3,1,1),plot(t,D_input[id_indices,:].T),
        legend(labels, ncol = 3),
        xlabel('t (time step)'),ylabel('$x_{k}(t)$'),grid(True),axis('tight')
        subplot(3,1,2),plot(t[:T],D_output[0,:T]),plot(t[T:],D_output[0,T:],'g'),
        plot(t[T:],id_out[0,T:],'r--'),
        xlabel('t (time step)'),ylabel('$x_1(t)$'),grid(True),axis('tight')
        subplot(3,1,3),stem(w0[0,:]),xlabel('$k$'),ylabel('w[k]'),grid(True)
        show()
        fig1.savefig('../Figures/fig_RNLFDS_ID_1.png',dpi=600,format='png')
    elif number == 2:
        from FinancialSystem import FinancialSystem
        t,X = FinancialSystem(0.5,0.1,0.1,True)
        X = X.T
        w0,D_input_train,D_output_train,r = GSpaRC(X,input_indices,output_indices,2,500,1,2,(2*2)**2+2*2,1e-6,5e-2)
        D_input,D_output,r = RCGDataGen(X,input_indices,output_indices,2,X.shape[1],1,2)
        id_out=w0@D_input
        N = D_input.shape[1]
        T = D_input_train.shape[1]
        t = range(N)
        fig2 = figure()
        id_indices = where(abs(w0[0,:])>0)[0]
        labels = []
        for k in range(len(id_indices)):
            labels.append('$x_{'+str(id_indices[k])+'}$')
        subplot(3,1,1),plot(t,D_input[id_indices,:].T),
        legend(labels, ncol = 3),
        xlabel('t (time step)'),ylabel('$x_{k}(t)$'),grid(True),axis('tight')
        subplot(3,1,2),plot(t[:T],D_output[0,:T]),plot(t[T:],D_output[0,T:],'g'),
        plot(t[T:],id_out[0,T:],'r--'),
        xlabel('t (time step)'),ylabel('$x_1(t)$'),grid(True),axis('tight')
        subplot(3,1,3),stem(w0[0,:]),xlabel('$k$'),ylabel('w[k]'),grid(True)
        fig2.savefig('../Figures/fig_RNLFDS_ID_1.png',dpi=600,format='png')
        show()
    else:
        from pandas import read_csv
        S2 = 36
        data = read_csv('../Data/FinancialMarginsData.csv',delimiter=',',header=0)
        margenes = data.values.T
        w0,D_input_train,D_output_train,r = GSpaRC(margenes,input_indices,output_indices,2,S2,1,2,3,5e-4,5e-2,filtering)
        D_input,D_output,r = RCGDataGen(margenes,input_indices,output_indices,2,80,1,2)
        id_out=w0@D_input
        N = D_input.shape[1]
        T = D_input_train.shape[1]
        t = range(N)
        fig3 = figure()
        id_indices = where(abs(w0[0,:])>0)[0]
        labels = []
        for k in range(len(id_indices)):
            labels.append('$x_{'+str(id_indices[k])+'}$')
        #subplot(3,2,1),
        plot(t,D_input[id_indices,:].T),
        legend(labels, ncol = 3),
        xlabel('t (months)'),ylabel('$x_{k}(t)$'),grid(True),axis('tight'),show()
        subplot(2,2,1),#plot(t[:T],D_output[0,:T]),
        semilogy(t[(T-1):],abs(D_output[0,(T-1):]-id_out[0,(T-1):])/max(abs(D_output[0,(T-1):])),'r.-'),
        axis('tight'),xlabel('t (months)'),grid(True),ylabel('$\epsilon(t)$['+str(output_indices[0]+1)+']')
        subplot(2,2,3),stem(w0[0,:]),xlabel('k'),ylabel('$w[k]$'),grid('True')
        input_indices = list(set(range(15)).difference({14}))
        output_indices = [14]
        w0,D_input_train,D_output_train,r = GSpaRC(margenes,input_indices,output_indices,2,S2,1,2,3,5e-4,5e-2,filtering)
        D_input,D_output,r = RCGDataGen(margenes,input_indices,output_indices,2,80,1,2)
        id_out=w0@D_input
        id_indices = where(abs(w0[0,:])>0)[0]
        labels = []
        for k in range(len(id_indices)):
            labels.append('$x_{'+str(id_indices[k])+'}$')
        #subplot(3,2,2),plot(t,D_input[id_indices,:].T),
        #legend(labels, ncol = 3),
        #xlabel('t (months)'),ylabel('$x_{k}(t)$'),grid(True),axis('tight')
        subplot(2,2,2),#plot(t[:T],D_output[0,:T]),
        semilogy(t[(T-1):],abs(D_output[0,(T-1):]-id_out[0,(T-1):])/max(abs(D_output[0,(T-1):])),'r.-'),
        axis('tight'),grid(True),xlabel('t (months)'),ylabel('$\epsilon(t)$['+str(output_indices[0]+1)+']')
        subplot(2,2,4),stem(w0[0,:]),xlabel('k'),ylabel('$w[k]$'),grid('True')
        show()
        fig3.savefig('../Figures/fig_RFMD_ID_1.png',dpi=600,format='png')
    return w0,D_input,D_output,r