import numpy as np
from scipy.linalg import expm

class Env( object):
    def __init__(self,
        action_space=[0,1,2],
        dt=0.1):
        super(Env, self).__init__()
        self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.n_features = 4
        self.state = np.array([1,0,0,0])
        self.nstep=0 
        self.dt=dt

    def reset(self):

        self.state = np.array([1,0,0,0])
        self.nstep = 0 

        return self.state

    def step(self, action):


        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi=np.mat(psi)

        J = 4  # control field strength
        sx = np.mat([[0, 1], [1, 0]], dtype=complex)
        sz = np.mat([[1, 0], [0, -1]], dtype=complex)

        U = np.matrix(np.identity(2, dtype=complex)) 
        
        H =  J *float(action)/(self.n_actions-1)* sz + 1 * sx
        U = expm(-1j * H * self.dt) 


        psi = U * psi  # final state

        target = np.mat([[0], [1]], dtype=complex) 

        err = 1 - (np.abs(psi.H * target) ** 2).item(0).real  
        rwd = 10 * (err<0.5)+100 * (err<0.1)+5000*(err < 10e-3)   

        done =( (err < 10e-3) or self.nstep>=np.pi/self.dt ) 
        self.nstep +=1  

        psi=np.array(psi)
        psi_T = psi.T
        self.state = np.array(psi_T.real.tolist()[0] + psi_T.imag.tolist()[0])

        return self.state, rwd, done, 1 - err




