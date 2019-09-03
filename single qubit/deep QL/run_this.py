from environment import Env
from Net_dql import DeepQNetwork
import numpy as np

def run_maze():
    step = 0
    fid_10 = 0
    ep_max = 500
    for episode in range(ep_max):
        observation = env.reset()

        while True:
            action = RL.choose_action(observation)
            observation_, reward, done, fid = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            RL.learn()

            observation = observation_
            if done:
                if episode >= ep_max-11:
                    fid_10 = max(fid_10,fid)
                break

            step += 1

    return fid_10


if __name__ == "__main__":
    
    dt_=np.pi/20
    env = Env(action_space=list(range(2)),   #allow two actions
               dt=dt_)              
        
    RL = DeepQNetwork(env.n_actions, env.n_features,
              learning_rate=0.01,
              reward_decay=0.9,
              e_greedy=0.99,
              replace_target_iter=200,
              memory_size=2000,
              e_greedy_increment=0.001,
              )
    fidelity = run_maze()
    print("Final_fidelity=", fidelity)
        


