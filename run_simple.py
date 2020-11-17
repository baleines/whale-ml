
from simple_agent import SimpleAgent
from random_agent import RandomAgent
from whale.whale import WhaleEnv

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
            'seed': datetime.utcnow().microsecond,
            'env_num': 1,
            'num_players': 5})
    # Set up agents
    action_num = 3
    agent = SimpleAgent(action_num=action_num, player_num=5)
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
            [print(t) for t in trajectory]
            i += 1


run_model(game_count=1)
