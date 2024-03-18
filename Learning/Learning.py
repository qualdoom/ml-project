from torchrl.envs import *
from torchrl.envs.libs.gym import *
from tensordict import TensorDict
import numpy as np
import matplotlib.pyplot as plt

from Constants.constants import *
from Environment.Environment import get_environment

class Learning:
    def __init__(self, game_name):
        self.game_name = game_name
    

    def play_session(self, agent, t_max=1e4, epsilon=-1):
        t_max = int(t_max)
        env = get_environment(self.game_name)
        n_actions = env.action_space.n

        total_reward = 0
        
        state = env.reset()

        for t in range(t_max):
            action = agent.select_action(state['pixels_trsf'], epsilon=epsilon)

            state['action'] = torch.zeros(n_actions)
            state['action'][action] = 1

            next_state = env.step(state)['next']

            next_state['done'] = next_state['done'] + next_state['terminated'] + next_state['truncated']
            next_state.pop('truncated', None)
            next_state.pop('terminated', None)
            state['next'] = next_state

            # print("current")
            # for i in range(FRAME_SKIP):
            #     plt.imshow(state['pixels_trsf'][i].cpu().permute(0, 1), cmap="gray")
            #     plt.show()


            # print("next")
            # for i in range(FRAME_SKIP):
            #     plt.imshow(state['pixels_trsf'][i].cpu().permute(0, 1), cmap="gray")
            #     plt.show()

            _action = torch.tensor(action)
            # _action[action] = 1

            data = TensorDict({
                "state" : state['pixels_trsf'],
                "action" : np.asarray([np.asarray(_action)]),
                "reward": [next_state['reward']],
                "next_state" : next_state['pixels_trsf'],
                "done" : [next_state['done']]
                }, 
                batch_size=[]
            )

            agent.record_experience(data)

            # loss = agent.compute_loss(state['pixels_trsf'], np.asarray([np.asarray(_action)]), [next_state['reward']], next_state['pixels_trsf'], [next_state['done']])

            total_reward += next_state['reward']

            state = next_state
            
            if next_state['done']:
                break
        
        return total_reward