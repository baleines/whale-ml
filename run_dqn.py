
from dqn_agent import DqnAgent
from random_agent import RandomAgent
from whale.whale import WhaleEnv
from whale.utils import set_global_seed
from datetime import datetime
import pprint

pp = pprint.PrettyPrinter(indent=4)
p = pp.pprint


def run_model(game_count=1):
    """
    run model for game_count games
    """

    # Make environment
    env = WhaleEnv(
        config={
            'active_player': 0,
            'seed': 0,
            'env_num': 1,
            'num_players': 5})
    # Set a global seed using time
    set_global_seed(datetime.utcnow().microsecond)
    # Set up agents
    action_num = 3
    agent = DqnAgent(action_num=action_num, player_num=5)
    agent_0 = RandomAgent(action_num=action_num)
    agent_1 = RandomAgent(action_num=action_num)
    agent_2 = RandomAgent(action_num=action_num)
    agent_3 = RandomAgent(action_num=action_num)
    agents = [agent, agent_0, agent_1, agent_2, agent_3]
    env.set_agents(agents)
    agent.load_pretrained()
    for game in range(game_count):

        # Generate data from the environment
        trajectories = env.run(is_training=False)

        # Print out the trajectories
        print('\nEpisode {}'.format(game))
        i = 0
        for trajectory in trajectories:
            print('\tPlayer {}'.format(i))
            p(trajectory[-1])
            i += 1


run_model(game_count=1)
