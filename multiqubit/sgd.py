import numpy as np
from scipy.linalg import expm


def hamiltonian(j):
    j=j.tolist()
    dim=len(j)
    J  = 1.

    H  = np.diag([1. for ii in range(dim-1)],1)+np.diag([1. for ii in range(dim-1)],-1)
    H += 0.5*J*np.diag(j) #The coupling should be (effectively) divided by a factor of 2 
    return H


def cost(seq,T):
    dim=len(seq[0])     
    dt = T/len(seq)
    U = np.matrix(np.identity(dim, dtype=complex)) 

    for ii in seq:
        H = hamiltonian(ii)
        U = expm(-1j * H * dt) * U  # Evolution operator

    p0    = np.zeros([dim,1])
    p0[0] = 1
    pt    = U * p0              #final state
    
    target     = np.zeros([dim,1])
    target[-1] = 1
    

    err = 1-(np.abs(pt.H * target)**2).item(0).real            
    return err

def gradient_descent(x, learning_rate, T, num_iterations):
    delta=0.01
    for i in range(num_iterations):
        v=np.random.random(np.shape(x))
        xp=x+v*delta
        xm=x-v*delta
        error_derivative = (cost(xp,T) - cost(xm,T))/(2*delta)
        x = x - (learning_rate) * error_derivative*v

    return 1 - cost(x,T)


dim=8
N = 20
T = 0.5*np.pi*(dim-1)        
ep_max=500

seq = 2*40*np.random.rand(N,dim)
fidelity= gradient_descent(seq, 0.01, T, ep_max)

print ('Final_fidelity=',fidelity)







