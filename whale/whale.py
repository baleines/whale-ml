import numpy as np

from whale.game import WhaleGame as Game
from whale.utils import encode_hand
from whale.utils import cards2list
import whale.seeding as seeding

# todo inject env super class methods to remove deps with env


class WhaleEnv():
    '''
        WhaleEnv is used to replay a whale game or a set of whale games
    '''

    def __init__(self, config):
        self.name = 'whale'
        self.game = Game(num_players=config["num_players"])
        # Get the number of players/actions in this game
        self.player_num = self.game.get_player_num()
        self.action_num = self.game.get_action_num()

        # A counter for the timesteps
        self.timestep = 0

        # Modes
        self.active_player = config['active_player']

        # Set random seed, default is None
        self._seed(config['seed'])

    def _extract_state(self, state):
        hand = encode_hand(state['hand'])
        score = state['score']
        legal_actions = self._get_legal_actions()
        extracted_state = {'hand': hand,
                           'score': score,
                           'legal_actions': legal_actions}
        return extracted_state

    def get_wins(self):
        return self.game.get_wins()

    def _get_legal_actions(self):
        return self.game.get_legal_actions()

    def get_perfect_information(self):
        ''' Get perfect information of current state

        Returns:
            (dict): Dictionary of perfect information for current state
        '''
        state = {}
        state['player_num'] = self.game.get_player_num()
        state['hand_cards'] = [cards2list(player.hand)
                               for player in self.game.players]
        state['played_cards'] = cards2list(self.game.round.played_cards)
        state['target'] = self.game.round.target.str
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.round.get_legal_actions(
            self.game.players, state['current_player'])
        return state

    def run(self, is_training=False):
        '''
        Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.

        TODO update this
        Note: The trajectories are 3-dimension list.
              The first dimension is for different players.
              The second dimension is for different transitions.
              The third dimension is for the contents of each transiton
        '''

        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.reset()

        # Loop to play the game
        # trajectories[player_id].append(state)
        while not self.is_over():
            # Agent choose action
            action = self.agents[player_id].step(state)

            # Agent plays action
            next_state, next_player_id = self.step(action)
            # Save action
            # TODO store this a better way
            trajectories[player_id].append(
                {'state': state,
                 'action': action,
                 'scores': self.game.get_scores(),
                 'win': False,
                 'done': False})

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            # if not self.game.is_over():
            # trajectories[player_id].append(state)

        wins = self.get_wins()

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(
                {'state': state,
                 'action': None,
                 'win': wins[player_id],
                 'done': True})

        return trajectories

    def set_agents(self, agents):
        '''
        Set the agents that will interact with the environment.
        This function must be called before `run`.

        Args:
            agents (list): List of Agent classes
        '''

        self.agents = agents

    def reset(self):
        '''
        Reset environment in single-agent mode
        Call `_init_game` if not in single agent mode
        '''
        state, player_id = self.game.init_game()

        return self._extract_state(state), player_id

    def is_over(self):
        ''' Check whether the curent game is over

        Returns:
            (boolean): True if current game is over
        '''
        return self.game.is_over()

    def step(self, action):
        ''' Step forward

        Args:
            action (int): The action taken by the current player

        Returns:
            (tuple): Tuple containing:

                (dict): The next state
                (int): The ID of the next player
        '''

        self.timestep += 1
        next_state, player_id = self.game.step(action)

        return self._extract_state(next_state), player_id

    def get_state(self, player_id):
        ''' Get the state given player id

        Args:
            player_id (int): The player id

        Returns:
            (numpy.array): The observed state of the player
        '''
        return self._extract_state(self.game.get_state(player_id))

    def get_scores(self):
        ''' Get the current scores
        '''
        self.game.get_scores()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game.np_random = self.np_random
        return seed
