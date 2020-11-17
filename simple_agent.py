import tensorflow as tf
import numpy as np

SIMPLE_WEIGHTS_FILE = 'simple_weights.npy'


class SimpleAgent:
    """
    Simple Agent

    The simple agent is train by simply using wining previous games
    """

    def __init__(self, action_num, player_num):
        ''' Initilize the random agent

        Args:
            action_num (int): The size of the ouput action space
        '''
        self.gamma = 0.80
        self.use_raw = False
        self.action_num = action_num
        self.player_num = player_num
        self.net = self._build_model(action_num, player_num)

    @staticmethod
    def _build_model(action_num, player_num):
        """
        Builds a deep neural net which predicts the Q values for all possible
        actions given a state. The input should have the shape of the state,
        and the output should have the same shape as the action space since
        we want 1 Q value per possible action.

        :return: Q network
        """
        # card counts + score for this round
        shape_size = 3 + 1
        inputs = tf.keras.layers.Input(shape=(shape_size,))
        mid = tf.keras.layers.Dense(
            32,
            activation='relu')(inputs)
        mid = tf.keras.layers.Dense(
            32,
            activation='relu')(mid)
        outputs = tf.keras.layers.Dense(
            action_num,
            activation='linear')(mid)
        # normalize
        outputs = tf.keras.layers.Lambda(
            lambda x: x / tf.keras.backend.sum(x))(outputs)
        net = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                    loss='mse')
        # net.summary()
        return net

    def remove_illegal(self, action_probs, legal_actions):
        ''' Remove illegal actions and normalize the
            probability vector

        Args:
            action_probs (numpy.array): A 1 dimension numpy array.
            legal_actions (list): A list of indices of legal actions.

        Returns:
            probd (numpy.array): A normalized vector without legal actions.
        '''
        probs = np.zeros(self.action_num)

        # todo access predicted actions and replace them in probs
        for action in legal_actions:
            probs[action] = action_probs[0][action]
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
        action, _ = self.eval_step(state)
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
            [state['hand']+[state['score']]],
            dtype=tf.float32)
        action_q = self.net(state_input)
        action_l = self.remove_illegal(
            action_q, state['legal_actions'])
        action = np.argmax(action_l, axis=0)
        action = int(action)
        return action, action_l

    def save_weight(self):
        with open(SIMPLE_WEIGHTS_FILE, 'wb') as f:
            np.save(f, self.net.get_weights())

    def load_pretrained(self):
        """
        Loads previously trained model.

        :return: None
        """
        weights = None
        try:
            with open(SIMPLE_WEIGHTS_FILE, 'rb') as f:
                weights = np.load(f, allow_pickle=True)
        except FileNotFoundError:
            print('Starting from scratch no pretrained file found')
        if weights is not None:
            print('Loading previously saved model')
            self.net.set_weights(weights)

    def train(self, batch):
        """
        Trains the underlying network with a batch of gameplay experiences to
        help it better predict the Q values.

        :param batch: a batch of gameplay experiences
        :return: training loss
        """

        state_batch, action_batch, \
            reward_batch, done_batch = batch
        target_q = np.full(
            (state_batch.shape[0], self.action_num), 0.2, dtype=float)
        for i in range(state_batch.shape[0]):
            target_q[i][action_batch[i]] = 0.6
        training_history = self.net.fit(epochs=10,
                                        x=state_batch, y=target_q, verbose=0)
        loss = training_history.history['loss']
        return loss
