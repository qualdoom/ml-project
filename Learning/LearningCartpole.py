from torchrl.envs import *
from torchrl.envs.libs.gym import *
from tensordict import TensorDict
import numpy as np
import matplotlib.pyplot as plt
import gym

from Constants.constants import *
from Environment.ImageCartpole import ImageCartPole

class Learning:
    def __init__(self, game_name):
        self.game_name = game_name
    

    def play_session(self, agent, t_max=1e4, epsilon=-1):
        t_max = int(t_max)
        env = gym.make("CartPole-v1")
        n_actions = env.action_space.n

        total_reward = 0
        
        state = env.reset()
        
        for t in range(t_max):
            action = agent.select_action(torch.as_tensor(state), epsilon=epsilon)

            next_state, reward, done, info = env.step(action)

            _action = torch.tensor(action)
            # _action[action] = 1

            data = TensorDict({
                "state" : torch.as_tensor(state),
                "action" : np.asarray([np.asarray(_action)]),
                "reward": [reward],
                "next_state" : torch.as_tensor(next_state),
                "done" : [done]
                }, 
                batch_size=[]
            )

            agent.record_experience(data)

            # loss = agent.compute_loss(state['pixels_trsf'], np.asarray([np.asarray(_action)]), [next_state['reward']], next_state['pixels_trsf'], [next_state['done']])

            total_reward += reward

            state = next_state
            
            if done:
                break
        
        return total_reward