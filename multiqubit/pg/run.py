from environment import State
from Net_pg import PolicyGradient
import numpy as np


N=20
env=State()

RL = PolicyGradient(
    n_actions=env.n_actions,
    n_features=env.n_features,
    learning_rate=0.01,
    reward_decay=0.99,
)

fid_max=0
for episode in range(500):

    observation =env.reset()
    for ii in range(N):
        
        action = RL.choose_action(observation)
        observation_, reward, done, fid = env.step(action,ii)
        RL.store_transition(observation, action, reward)
        observation = observation_
        
        if done or ii>=N-1:

            break
        
    if episode>=490:
        if fid > fid_max:
            fid_max = np.copy(fid)
    
    RL.learn()
        
print('Final_fidelity=',fid_max)        
