"""
Training loop

This module trains the DQN agent by trial and error. In this module the DQN
agent will play the game episode by episode, store the gameplay experiences
and then use the saved gameplay experiences to train the underlying model.
"""
from simple_agent import SimpleAgent
from random_agent import RandomAgent
from whale.whale import WhaleEnv
from whale.utils import set_global_seed
from datetime import datetime

EVAL_EPISODES_COUNT = 1


def evaluate_training_result(env, agent):
    """
    Evaluates the performance of the current DQN agent by using it to play a
    few episodes of the game and then calculates the average reward it gets.
    The higher the average reward is the better the DQN agent performs.

    :param env: the game environment
    :param agent: the DQN agent
    :return: average reward across episodes
    """
    total_reward = 0.0
    for i in range(EVAL_EPISODES_COUNT):
        trajectories, _ = env.run(is_training=True)
        # calculate reward
        episode_reward = 0.0
        for ts in trajectories[0]:
            print(
                'State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.
                format(ts[0], ts[1], ts[2], ts[3], ts[4]))
            episode_reward += ts[2]
        total_reward += episode_reward

    average_reward = total_reward / EVAL_EPISODES_COUNT
    return average_reward


def run_model():
    """
    Trains a DQN agent to play the CartPole game by trial and error

    :return: None
    """

    # buffer = ReplayBuffer()
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
    agent = SimpleAgent(action_num=action_num, player_num=5)
    agent_0 = RandomAgent(action_num=action_num)
    agent_1 = RandomAgent(action_num=action_num)
    agent_2 = RandomAgent(action_num=action_num)
    agent_3 = RandomAgent(action_num=action_num)
    agents = [agent, agent_0, agent_1, agent_2, agent_3]
    env.set_agents(agents)
    agent.load_pretrained()

    avg_reward = evaluate_training_result(env, agent)
    print(
        'perf:{0:.2f} on {1} games'.format(avg_reward, EVAL_EPISODES_COUNT))
    print('run end')


run_model()
