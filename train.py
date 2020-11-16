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

EVAL_EPISODES_COUNT = 100


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
        env.reset()
        trajectories, _ = env.run(is_training=True)
        # calculate reward
        episode_reward = 0.0
        for ts in trajectories[0]:
            # print(
            #    'State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.
            #    format(ts[0], ts[1], ts[2], ts[3], ts[4]))
            episode_reward += ts[2]
        total_reward += episode_reward

    average_reward = total_reward / EVAL_EPISODES_COUNT
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
    state_batch = np.zeros((0, 9), dtype=float)
    next_state_batch = np.zeros((0, 9), dtype=float)
    action_batch = []
    reward_batch = []
    done_batch = []
    for _ in range(0, game_count):
        env.reset()
        trajectories, _ = env.run(is_training=False)
        state_l = []
        next_state_l = []
        action = []
        reward = []
        done = []
        for state in trajectories[0]:
            state_l.append(state[0]['hand'] +
                           [state[0]['gain']]+state[0]['scores'])
            next_state_l.append(
                state[3]['hand']+[state[3]['gain']]+state[3]['scores'])
            action.append(state[1])
            reward.append(state[2])
            done.append(state[4])
        state_batch = np.concatenate((state_batch, np.array(state_l)))
        next_state_batch = np.concatenate(
            (next_state_batch, np.array(next_state_l)))
        action_batch = action_batch + action
        reward_batch = reward_batch + reward
        done_batch = done_batch + done
    return (state_batch, next_state_batch, action_batch,
            reward_batch, done_batch)


def train_model(max_episodes=10):
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
    agent = DqnAgent(dim=1, action_num=action_num, player_num=5)
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
        # print(f'{datetime.utcnow()} train ...')
        loss = agent.train(collect_gameplay_experiences(
            env, agents, GAME_COUNT_PER_EPISODE))
        # print(f'{datetime.utcnow()} eval  ...')
        avg_reward = evaluate_training_result(env, agent)
        # print(f'{datetime.utcnow()} calc  ...')
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
