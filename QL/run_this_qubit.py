import environment
from RL_brain import QLearningTable
import numpy as np

env = Maze()
RL = QLearningTable(actions=list(range(env.n_actions)))


N      = 20 
dt     = 2 * np.pi/N   
ep_max = 500
fidelity=np.zeros(ep_max)

RL       = QLearningTable(actions=list(range(env.n_actions)))
fid_10 = 0
for episode in range(ep_max):
    observation = env.reset()
    while True:
        
        action = RL.choose_action(str(observation))    
        observation_, reward, done, fid = env.step(action)
        RL.learn(str(observation), action, reward, str(observation_))
        observation = observation_
        if done:
            if episode >= ep_max-11:
                fid_10 = max(fid_10,fid)

            break
                
print('Final_fidelity=',fid_10)




