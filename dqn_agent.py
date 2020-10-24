import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


class DqnAgent:
    """
    DQN Agent

    The agent that explores the game and learn how to play the game by
    learning how to predict the expected long-term return, the Q value given
    a state-action pair.
    """

    def __init__(self, dim, action_num):
        ''' Initilize the random agent

        Args:
            action_num (int): The size of the ouput action space
        '''
        self.gamma = 0.80
        self.use_raw = False
        self.dim = dim
        self.action_num = action_num
        self.q_net = self._build_dqn_model(dim)
        self.target_q_net = self._build_dqn_model(dim)

    @staticmethod
    def _build_dqn_model(dim):
        """
        Builds a deep neural net which predicts the Q values for all possible
        actions given a state. The input should have the shape of the state, and
        the output should have the same shape as the action space since we want
        1 Q value per possible action.

        :return: Q network
        """
        q_net = Sequential()
        q_net.add(Dense(64, input_dim=dim, activation='relu',
                        kernel_initializer='he_uniform'))
        q_net.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        q_net.add(
            Dense(1, activation='linear', kernel_initializer='he_uniform'))
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                      loss='mse')
        return q_net

    def random_policy(self, state):
        """
        Outputs a random action

        :param state: not used
        :return: action
        """
        # todo see if this has to be changed
        return np.random.randint(0, self.action_num)

    def collect_policy(self, state):
        """
        Similar to policy but with some randomness to encourage exploration.

        :param state: the game state
        :return: action
        """
        if np.random.random() < 0.05:
            return self.random_policy(state['legal_actions'])
        return self.policy(state)

    def remove_illegal(self, action_probs, legal_actions):
        ''' Remove illegal actions and normalize the
            probability vector

        Args:
            action_probs (numpy.array): A 1 dimention numpy array.
            legal_actions (list): A list of indices of legal actions.

        Returns:
            probd (numpy.array): A normalized vector without legal actions.
        '''
        probs = np.zeros(self.action_num)
        # todo access predicted actions and replace them in probs
        probs[legal_actions] = action_probs[legal_actions]
        if np.sum(probs) == 0:
            probs[legal_actions] = 1 / len(legal_actions)
        else:
            probs /= sum(probs)
        return probs

    def step(self, state):
        ''' Predict the action for generating training data

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        state_input = tf.convert_to_tensor(
            state['obs'], dtype=tf.float32)
        action_q = self.q_net(state_input)
        action_l = self.remove_illegal(
            action_q.numpy(), state['legal_actions'])
        action = np.argmax(action_l, axis=0)
        return action

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            probs (list): a list of probabilies
        '''
        state_input = tf.convert_to_tensor(
            state['obs'], dtype=tf.float32)
        action_q = self.q_net(state_input)
        action_l = self.remove_illegal(
            action_q.numpy(), state['legal_actions'])
        action = np.argmax(action_l, axis=0)
        return action, action_l

    def update_target_network(self):
        """
        Updates the current target_q_net with the q_net which brings all the
        training in the q_net to the target_q_net.

        :return: None
        """
        with open('dqn_weights.npy', 'wb') as f:
            np.save(f, self.q_net.get_weights())
        self.target_q_net.set_weights(self.q_net.get_weights())

    def load_pretrained(self):
        """
        Loads previously trained model.

        :return: None
        """
        weights = None
        try:
            with open('dqn_weights.npy', 'rb') as f:
                weights = np.load(f, allow_pickle=True)
        except FileNotFoundError:
            print('no pretrained file found')
        if weights is not None:
            self.target_q_net.set_weights(weights)
            self.q_net.set_weights(weights)

    def train(self, batch):
        """
        Trains the underlying network with a batch of gameplay experiences to
        help it better predict the Q values.

        :param batch: a batch of gameplay experiences
        :return: training loss
        """
        # state_batch, next_state_batch, action_batch, reward_batch, \
        # done_batch = batch
        # print('State: {}, Action: {}, Reward: {}, Next State: {}, Done: {}'.
        #       format(ts[0], ts[1], ts[2], ts[3], ts[4]))
        state_batch = [state[0]["obs"] for state in batch]
        next_state_batch = [state[3]["obs"] for state in batch]
        action_batch = [state[1]for state in batch]
        reward_batch = [state[2]for state in batch]
        done_batch = [state[4]for state in batch]
        target_q_agg = list()
        for i in range(len(state_batch)):
            current_q = self.q_net(state_batch[i]).numpy()
            target_q = np.copy(current_q)
            next_q = self.target_q_net(next_state_batch[i]).numpy()
            # get the max from possible actions
            max_next_q = np.amax(next_q)

            target_q_val = reward_batch[i] + \
                (next_state_batch[i][3] - state_batch[i][3])/10.0

            if not done_batch[i]:
                target_q_val += self.gamma * max_next_q
                # this is replacing the value for done action
                target_q[action_batch[i]] = target_q_val
            target_q_agg.append(target_q)

        # idea of training maximize the number of victories
        training_history = self.q_net.fit(
            x=state_batch[i], y=target_q, verbose=0)
        loss = training_history.history['loss'][0]
        return loss
