from Net_pg import PolicyGradient
import numpy as np
from environment import Env


N=20
env=Env(dt=np.pi/N)

RL = PolicyGradient(
    n_actions=env.n_actions,
    n_features=env.n_states,
    learning_rate=0.002,
    reward_decay=0.99,
)

fid_10 = 0
ep_max=500
for episode in range(ep_max):

    observation =env.reset()

    for ii in range(N):
        
        action = RL.choose_action(observation)
        observation_, reward, done, fid = env.step(action)
        
        RL.store_transition(observation, action, reward)
        observation = observation_
        if done:
            if episode >= ep_max-11:
                fid_10 = max(fid_10,fid)
            break
 
    RL.learn()
    

print('Final_fidelity=',fid_10)