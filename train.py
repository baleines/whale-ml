import numpy as np


def collect_gameplay_experiences(env, agents, game_count, winner=True):
    """
    Collects gameplay experiences by playing env with the instructions
    produced by agent and stores the gameplay experiences in buffer.

    :param env: the game environment
    :param agent: the DQN agent
    :param buffer: the replay buffer
    :return: None
    """
    # TODO fix this function
    state_batch = np.zeros((0, 4), dtype=int)
    action_batch = []
    score_batch = []
    done_batch = []
    env.set_agents(agents)
    game = 0
    while game < game_count:
        env.reset()
        trajectories = env.run(is_training=False)
        winner_id = 0
        if winner:
            winner_id = 0
            # get the winning trajectory for training
            for trajectory in trajectories:
                if trajectory[-1]['win']:
                    break
                winner_id += 1
            # if not winner go to next game
            if winner_id >= len(trajectories):
                continue
        game += 1
        states = []
        actions = []
        scores = []
        dones = []
        for trajectory in trajectories[winner_id]:
            states.append(trajectory['state']['hand'] +
                          [trajectory['state']['score']])
            actions.append(trajectory['action'])
            scores.append(trajectory['state']['score'])
            dones.append(trajectory['done'])
        state_batch = np.concatenate((state_batch, np.array(states)))
        action_batch += actions
        score_batch += scores
        done_batch += dones
    return (state_batch, action_batch,
            score_batch, done_batch)


def evaluate_training_result(env, agents, game_num):
    """
    Evaluates the performance of the current DQN agent by using it to play a
    few episodes of the game and then calculates the average reward it gets.
    The higher the average reward is the better the DQN agent performs.

    :param env: the game environment
    :param agent: the DQN agent
    :return: average reward across episodes
    """
    average_rewards = [0.0] * len(agents)
    env.set_agents(agents)
    for i in range(game_num):
        env.reset()
        trajectories = env.run(is_training=False)
        # get the win on the last state of the game
        for j in range(len(trajectories)):
            if trajectories[j][-1]['win']:
                average_rewards[j] += 1.0/game_num

    return average_rewards
