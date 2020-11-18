''' A toy example of playing Whale with random agents
'''


from random_agent import RandomAgent
from human_agent import HumanAgent
from whale.whale import WhaleEnv

from datetime import datetime
import pprint
pp = pprint.PrettyPrinter(indent=4)
p = pp.pprint

# Make environment
env = WhaleEnv(
    config={
        'active_player': 0,
        'seed': datetime.utcnow().microsecond,
        'env_num': 1,
        'num_players': 4})
episode_num = 1


# Set up agents
agent_0 = HumanAgent(action_num=env.action_num)
agent_1 = RandomAgent(action_num=env.action_num)
agent_2 = RandomAgent(action_num=env.action_num)
agent_3 = RandomAgent(action_num=env.action_num)
env.set_agents([agent_0, agent_1, agent_2, agent_3])

for episode in range(episode_num):

    # Generate data from the environment
    trajectories = env.run(is_training=False)

    # Print out the trajectories
    print('\nEpisode {}'.format(episode))
    i = 0
    for trajectory in trajectories:
        print('\tPlayer {}'.format(i))
        p(trajectory[-1])
        i += 1
