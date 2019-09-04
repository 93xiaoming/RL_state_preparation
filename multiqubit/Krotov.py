import numpy as np
from scipy import linalg


def hamiltonian(j):
    j=j.tolist()
    dim=len(j)
    J  = 1.
    H  = np.diag([1. for ii in range(dim-1)],1)+np.diag([1. for ii in range(dim-1)],-1)
    H += 0.5*J*np.diag(j) #The coupling should be (effectively) divided by a factor of 2 
    return H


dim=8
N = 20
T = 0.5*np.pi*(dim-1)
dt = T/N
ep_max = 500 

observable = np.mat(np.zeros(shape=(dim,dim), dtype=complex))
observable[-1, -1] = 1

psi = np.mat(np.zeros(shape=(dim, N+1), dtype=complex))
psi[0,0] = 1    
pseudo = np.mat(np.zeros(shape=(dim, N+1), dtype=complex))    # 

seq = 2*40*np.random.rand(N,dim)
seq_f = seq

for i in range(N):
    psi[:,i+1] = linalg.expm(-(1j) * hamiltonian(seq[i]) * dt).dot(psi[:,i])
pseudo[:,-1] = observable.dot(psi[:,-1])

dx = 0.01   
fid_max=0
for episode in range(ep_max):
    for j in reversed(range(N)):
        pseudo[:,j] = linalg.expm((1j) * hamiltonian(seq[j]) * dt).dot(pseudo[:,j+1])
    for k in range(N):
        for ii in range(dim):
            seq_f[k,ii] = seq[k,ii] + 1*(pseudo[ii,k].conj()* psi[ii,k]).imag
        
        seq_f = np.clip(seq_f,0,2*40)
        psi[:,k+1] = linalg.expm(-(1j) * hamiltonian(seq_f[k]) * dt).dot(psi[:,k])
        seq = seq_f
    fid = (np.absolute(psi[-1,-1]))**2
    pseudo[:,-1] = observable.dot(psi[:,-1])

    if episode > ep_max-11:
        if fid>fid_max:
            fid_max = np.copy(fid)
    
print('Final_fidelity=',fid_max)








    
