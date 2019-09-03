import numpy as np
from scipy import linalg

sx = 1/2 * np.mat([[0, 1],[ 1, 0]], dtype=complex)
sy = 1/2 * np.mat([[0, -1j],[1j, 0]], dtype=complex)
sz = 1/2 * np.mat([[1, 0],[0, -1]], dtype=complex)

def hamiltonian(j):
    J = 4
    H = (j) * J * sz + sx
    return H


T = 2*np.pi       
N = 20
dt = T/N

I = 500
fidelity = np.zeros(I+1)

observable = np.mat(np.zeros(shape=(2,2), dtype=complex))
observable[-1, -1] = 1

psi = np.mat(np.zeros(shape=(2, N+1), dtype=complex))
psi[0,0] = 1   
pseudo = np.mat(np.zeros(shape=(2, N+1), dtype=complex))    # 


seq = np.random.rand(N)
seq_f = seq

for i in range(N):
    psi[:,i+1] = linalg.expm(-(1j) * hamiltonian(seq[i]) * dt).dot(psi[:,i])
fidelity[0]=(np.absolute(psi[-1,-1]))**2
pseudo[:,-1] = observable.dot(psi[:,-1])
dj = 0.01    


for i in range(I):
    for j in reversed(range(N)):
        pseudo[:,j] = linalg.expm((1j) * hamiltonian(seq[j]) * dt).dot(pseudo[:,j+1])
    for k in range(N):
        dH = (hamiltonian(seq[k]+dj) - hamiltonian(seq[k]-dj)) / (2*dj)
        seq_f[k] = seq[k] + (pseudo[:,k].conj().T.dot(dH.dot(psi[:,k]))).imag[0,0]
        psi[:,k+1] = linalg.expm(-(1j) * hamiltonian(seq_f[k]) * dt).dot(psi[:,k])
        seq = seq_f
    fidelity[i+1] += (np.absolute(psi[-1,-1]))**2
    pseudo[:,-1] = observable.dot(psi[:,-1])


print('final_fidelity=',fidelity[-1])

