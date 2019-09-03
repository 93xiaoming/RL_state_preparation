import numpy as np
from scipy.linalg import expm

def cost(seq):

    N=len(seq) 

    dt=2*np.pi/N  

    sx=1/2 * np.mat([[0,1],\
                 [1,0]], dtype=complex)
    sz=1/2 * np.mat([[1,0],\
                 [0,-1]], dtype=complex)

    U = np.matrix(np.identity(2, dtype=complex)) #initial Evolution operator

    J=4                                   # control field strength

    for ii in seq:
        H =ii * J * sz + 1*sx # Hamiltonian
        U = expm(-1j * H * dt) * U  # Evolution operator

    p0=np.mat([[1],[0]], dtype=complex) #initial state
    pt=U * p0              #final state

    target = np.mat([[0], [1]], dtype=complex)                             # south pole

    err = 1-(np.abs(pt.H * target)**2).item(0).real            #infidelity (to make it as small as possible)

    return err



delta=0.01
cost_hist = []

def gradient_descent(x, dim, learning_rate, num_iterations):
    for i in range(num_iterations):
        v=np.random.rand(dim) 
        xp=x+v*delta
        xm=x-v*delta
        error_derivative = (cost(xp) - cost(xm))/(2*delta)
        x = x - (learning_rate) * error_derivative*v
        cost_hist.append(cost(xp))
    return cost(x)


N        = 20
seq      = np.random.rand(N)
ep_max   = 500
fidelity = 1-gradient_descent(seq, N, 0.01, ep_max)

print('Final_fidelity=',fidelity)    


