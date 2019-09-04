import numpy as np
from scipy.linalg import expm
import itertools

SPIN_NUM = 8
MAG = 2*40
COUPLING = 2*1 
DT = 0.05*(SPIN_NUM-1)*np.pi*0.5

def mg_config(x,dim):
    if dim>1:
        Y=[]
        for ii in range(2):
            for xx in x:
                y=xx+[ii]
                Y.append(y)
        Y=mg_config(Y,dim-1)
    else:
        Y=x
    return Y

mag=MAG*np.array(mg_config([[0],[1]],SPIN_NUM))

class State(object):
    def __init__(self):
        super(State, self)
        self.action_space = mag
        self.n_actions = len(self.action_space)
        self.n_features = SPIN_NUM*2
        

    
    def reset(self):
        psi = [0 for i in range(SPIN_NUM)]
        psi[0] = 1

        self.state = np.array([str(i) for i in psi])
        self.state = np.array(list(itertools.chain(*[(i.real, i.imag) for i in psi])))

        return self.state
    
    def step(self, actionnum, stp):
        actions = self.action_space[actionnum]
        ham = np.diag([COUPLING for i in range(SPIN_NUM-1)], 1)*(1-0j) + np.diag([COUPLING for i in range(SPIN_NUM-1)], -1)*(1+0j) + np.diag(actions)

        statess = [complex(self.state[2*i], self.state[2*i+1]) for i in range(SPIN_NUM)]
        statelist = np.transpose(np.mat(statess))
        next_state = expm(-1j*ham*DT)*statelist
        fidelity = (abs(next_state[-1])**2)[0,0]
        
        xi=0.999
        if fidelity<xi:
            reward = fidelity *10

        doned = False
        if fidelity >= xi:
            reward = 2500 
            doned = True


        reward = reward*(0.95**stp)

        next_states = [next_state[i,0] for i in range(SPIN_NUM)]
        next_states = np.array(list(itertools.chain(*[(i.real, i.imag) for i in next_states])))

        self.state = next_states
        return next_states, reward, doned, fidelity





