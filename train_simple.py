"""
Training loop

This module trains the DQN agent by trial and error. In this module the DQN
agent will play the game episode by episode, store the gameplay experiences
and then use the saved gameplay experiences to train the underlying model.
"""
from simple_agent import SimpleAgent
from no_draw_agent import NoDrawAgent
from whale.whale import WhaleEnv
from whale.utils import set_global_seed
from datetime import datetime
from train import collect_gameplay_experiences, evaluate_training_result

EVAL_EPISODES_COUNT = 100
GAME_COUNT_PER_EPISODE = 20


def train_model(max_episodes=100):
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
    agent_0 = NoDrawAgent(action_num=action_num)
    agent_1 = NoDrawAgent(action_num=action_num)
    agent_2 = NoDrawAgent(action_num=action_num)
    agent_3 = NoDrawAgent(action_num=action_num)
    # agent_train = RandomAgent(action_num=action_num)
    agents = [agent, agent_0, agent_1, agent_2, agent_3]
    # train_agents = [agent_train, agent_0, agent_1, agent_2, agent_3]
    env.set_agents(agents)
    agent.load_pretrained()
    min_perf, max_perf = 1.0, 0.0
    for episode_cnt in range(1, max_episodes+1):
        # print(f'{datetime.utcnow()} train ...')
        loss = agent.train(collect_gameplay_experiences(
            env, agents, GAME_COUNT_PER_EPISODE))
        # print(f'{datetime.utcnow()} eval  ...')
        avg_rewards = evaluate_training_result(
            env, agents, EVAL_EPISODES_COUNT)
        # print(f'{datetime.utcnow()} calc  ...')
        if avg_rewards[0] > max_perf:
            max_perf = avg_rewards[0]
            agent.save_weight()
        if avg_rewards[0] < min_perf:
            min_perf = avg_rewards[0]
        print(
            '{0:03d}/{1} perf:{2:.2f}(min:{3:.2f} max:{4:.2f})'
            'loss:{5:.4f} rewards:{6:.2f} {7:.2f} {8:.2f} {9:.2f}'.format(
                episode_cnt, max_episodes, avg_rewards[0], min_perf, max_perf,
                loss[0], avg_rewards[1], avg_rewards[2], avg_rewards[3],
                avg_rewards[4]))
    # env.close()
    print('training end')


train_model()
