"""
Training loop

This module trains the DQN agent by trial and error. In this module the DQN
agent will play the game episode by episode, store the gameplay experiences
and then use the saved gameplay experiences to train the underlying model.
"""
from dqn_agent import DqnAgent
from random_agent import RandomAgent
from whale.whale import WhaleEnv
from whale.utils import set_global_seed
from datetime import datetime
import numpy as np


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
    episodes_to_play = 100
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


def collect_gameplay_experiences(env, agent, game_count):
    """
    Collects gameplay experiences by playing env with the instructions
    produced by agent and stores the gameplay experiences in buffer.

    :param env: the game environment
    :param agent: the DQN agent
    :param buffer: the replay buffer
    :return: None
    """
    # TODO fix this function
    state_batch = []
    next_state_batch = []
    action_batch = []
    reward_batch = []
    done_batch = []
    for i in range(1, game_count):
        env.reset()
        trajectories, _ = env.run(is_training=False)
        # here we should have 8,n tensor
        # t = np.zeros(shape=(8, len(trajectories[0])))
        # # for each trajectory
        # for i in range(len(trajectories[0])):
        #     # populate each value
        #     for j in range(len(t)):
        #         t[j][i] = trajectories[0][i][0]["obs"][j][0]
        # state_batch.append(t)
        # # for each trajectory
        # for i in range(len(trajectories[0])):
        #     # populate each value
        #     for j in range(len(t)):
        #         t[j][i] = trajectories[0][i][3]["obs"][j][0]
        # next_state_batch.append(t)

        state_batch.append(np.array([state[0]["obs"]
                                     for state in trajectories[0]]))
        next_state_batch.append(
            np.array([state[3]["obs"] for state in trajectories[0]]))
        action_batch.append([state[1]for state in trajectories[0]])
        reward_batch.append([state[2]for state in trajectories[0]])
        done_batch.append([state[4]for state in trajectories[0]])
    return (state_batch, next_state_batch, action_batch,
            reward_batch, done_batch)


def train_model(max_episodes=1000):
    """
    Trains a DQN agent to play the CartPole game by trial and error

    :return: None
    """

    # buffer = ReplayBuffer()
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
    # Set a global seed using time
    set_global_seed(datetime.utcnow().microsecond)
    # Set up agents
    action_num = 3
    agent = DqnAgent(dim=1, action_num=action_num)
    agent_0 = RandomAgent(action_num=action_num)
    agent_1 = RandomAgent(action_num=action_num)
    agent_2 = RandomAgent(action_num=action_num)
    agent_3 = RandomAgent(action_num=action_num)
    agents = [agent, agent_0, agent_1, agent_2, agent_3]
    env.set_agents(agents)
    agent.load_pretrained()
    UPDATE_TARGET_RATE = 20
    GAME_COUNT_PER_EPISODE = 2
    min_perf, max_perf = 1.0, 0.0
    for episode_cnt in range(1, max_episodes+1):
        loss = agent.train(collect_gameplay_experiences(
            env, agents, GAME_COUNT_PER_EPISODE))
        avg_reward = evaluate_training_result(env, agent)
        target_update = episode_cnt % UPDATE_TARGET_RATE == 0
        if avg_reward > max_perf:
            max_perf = avg_reward
            agent.save_weight()
        if avg_reward < min_perf:
            min_perf = avg_reward
        print(
            '{0:03d}/{1} perf:{2:.2f}(min:{3} max:{4})'
            'up:{5:1d} loss:{6}'.format(
                episode_cnt, max_episodes, avg_reward, min_perf, max_perf,
                target_update, loss))
        if target_update:
            agent.update_target_network()
    # env.close()
    print('training end')


train_model()
