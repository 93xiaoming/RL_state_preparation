from environment import State
from Net_dql import DeepQNetwork
import numpy as np

N=20
env = State()

RL = DeepQNetwork(env.n_actions, env.n_features,
          learning_rate=0.01,
          reward_decay=0.9,
          e_greedy=0.99,
          replace_target_iter=200,
          memory_size=2000,
          e_greedy_increment=0.001,
          )



step = 0
fid_max=0
for episode in range(500):
    observation = env.reset()
    
    for i in range(N):
        action = RL.choose_action(observation)
        observation_, reward, done, fidelity = env.step(action,i)            
        RL.store_transition(observation, action, reward, observation_)

        if (step > 500) and (step%5 == 0):
            RL.learn()
      
        observation = observation_

        if done:
            break
        
        step += 1

    if episode>=490:
        if fidelity > fid_max:
            fid_max = np.copy(fidelity)

print('Final_fidelity=',fid_max)
            
        

    
    
    
    
    
    
    
    