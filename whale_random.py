''' A toy example of playing Whale with random agents
'''


from random_agent import RandomAgent
from whale.whale import WhaleEnv
from whale.utils import set_global_seed

# Make environment
env = WhaleEnv(
    config={
        'active_player': 0,
        'seed': 0,
        'env_num': 1,
        'num_players': 4})
episode_num = 1

# Set a global seed
set_global_seed(0)

# Set up agents
agent_0 = RandomAgent(action_num=env.action_num)
agent_1 = RandomAgent(action_num=env.action_num)
agent_2 = RandomAgent(action_num=env.action_num)
agent_3 = RandomAgent(action_num=env.action_num)
env.set_agents([agent_0, agent_1, agent_2, agent_3])

for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=False)

    # Print out the trajectories
    print('\nEpisode {}'.format(episode))
    i = 0
    for trajectory in trajectories:
        print('\tPlayer {}'.format(i))
        for ts in trajectory:
            print('\tState: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.
                  format(ts[0], ts[1], ts[2], ts[3], ts[4]))
        i += 1
