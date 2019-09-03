
import numpy as np
from scipy.linalg import expm

class Env( object):
    def __init__(self,
        dt=np.pi/10):
        super(Env, self).__init__()
        self.n_actions = 2
        self.n_states = 4
        self.state = np.array([1,0,0,0])
        self.nstep=0 ##count number of step at each episode
        self.dt=dt
    def reset(self):

        # return observation
        self.state = np.array([1,0,0,0])
        self.nstep = 0 #reset number of step at each episode

        return self.state

    def step(self, action):


        psi = np.array([self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi=np.mat(psi)

        J = 4  # control field strength
        # J=2
        ######## pauli matrix
        sx = np.mat([[0, 1], [1, 0]], dtype=complex)
        sz = np.mat([[1, 0], [0, -1]], dtype=complex)

        U = np.matrix(np.identity(2, dtype=complex))  # initial Evolution operator
        
        H =  J *float(action)/(self.n_actions-1)* sz + 1 * sx
        U = expm(-1j * H * self.dt)  # Evolution operator


        psi = U * psi 
        ########################## target state defined by yourself
        target = np.mat([[0], [1]], dtype=complex)  # south pole
        err = 1 - (np.abs(psi.H * target) ** 2).item(0).real  # infidelity (to make it as small as possible)
################################################################

        #rwd =  10*(err < 10e-3)  # give reward only when the error is small enough
        
        #rwd = -1 +5000.*(err < 10e-3)   #or other type of reward
        
        rwd = (err<0.5)*10 +(err<0.1)*100 + 5000.*(err < 10e-3)   #nice reward


        done =( (err < 10e-3) or self.nstep>=np.pi/self.dt )  #end each episode if error is small enough, or step is larger than 2*pi/dt
        self.nstep +=1  # step counter add one

        psi=np.array(psi)
        ppsi = psi.T
        self.state = np.array(ppsi.real.tolist()[0] + ppsi.imag.tolist()[0])

        return self.state, rwd, done, 1-err




