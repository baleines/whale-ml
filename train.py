"""
Training loop

This module trains the DQN agent by trial and error. In this module the DQN
agent will play the game episode by episode, store the gameplay experiences
and then use the saved gameplay experiences to train the underlying model.
"""
from dqn_agent import DqnAgent
from replay_buffer import ReplayBuffer
from random_agent import RandomAgent
from whale.whale import WhaleEnv
from whale.utils import set_global_seed


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
    episodes_to_play = 10
    for i in range(episodes_to_play):
        trajectories, _ = env.run(is_training=True)
        # calculate reward
        episode_reward = 0.0
        for ts in trajectories[0]:
            # print(
            #    'State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.
            #    format(ts[0], ts[1], ts[2], ts[3], ts[4]))
            episode_reward += ts[2]
        total_reward += episode_reward

    average_reward = total_reward / episodes_to_play
    return average_reward


def collect_gameplay_experiences(env, agent, buffer):
    """
    Collects gameplay experiences by playing env with the instructions
    produced by agent and stores the gameplay experiences in buffer.

    :param env: the game environment
    :param agent: the DQN agent
    :param buffer: the replay buffer
    :return: None
    """
    trajectories, _ = env.run(is_training=False)
    # Feed transitions into agent memory, and train the agent
    for ts in trajectories[0]:
        # print('State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.
        #       format(ts[0], ts[1], ts[2], ts[3], ts[4]))
        buffer.store_gameplay_experience(ts[0], ts[3],
                                         ts[2], ts[1], ts[4])


def train_model(max_episodes=50000):
    """
    Trains a DQN agent to play the CartPole game by trial and error

    :return: None
    """

    buffer = ReplayBuffer()
    # Make environment
    env = WhaleEnv(
        config={
            'allow_step_back': False,
            'allow_raw_data': False,
            'single_agent_mode': False,
            'active_player': 0,
            'record_action': False,
            'seed': 0,
            'env_num': 1,
            'num_players': 5})
    # Set a global seed
    set_global_seed(0)
    # Set up agents
    action_num = 3
    agent = DqnAgent(dim=1, action_num=action_num)
    agent_0 = RandomAgent(action_num=action_num)
    agent_1 = RandomAgent(action_num=action_num)
    agent_2 = RandomAgent(action_num=action_num)
    agent_3 = RandomAgent(action_num=action_num)
    agents = [agent, agent_0, agent_1, agent_2, agent_3]
    env.set_agents(agents)
    for _ in range(10):
        collect_gameplay_experiences(env, agents, buffer)
    for episode_cnt in range(max_episodes):
        collect_gameplay_experiences(env, agents, buffer)
        gameplay_experience_batch = buffer.sample_gameplay_batch()
        loss = agent.train(gameplay_experience_batch)
        avg_reward = evaluate_training_result(env, agent)
        print('Episode {0}/{1} and so far the performance is {2} and '
              'loss is {3}'.format(episode_cnt, max_episodes,
                                   avg_reward, loss))
        if episode_cnt % 20 == 0:
            agent.update_target_network()
    env.close()
    print('No bug lol!!!')


train_model()
