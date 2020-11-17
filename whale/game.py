from whale.dealer import WhaleDealer as Dealer
from whale.player import WhalePlayer as Player
from whale.round import WhaleRound as Round

import numpy as np

# inital card per hand
CARD_PER_HAND = 3
# number of possible action (not all legal)
ACTION_NUM = 3


class WhaleGame(object):
    '''
    Represents a whale game.
    '''

    def __init__(self, num_players):
        self.np_random = np.random.RandomState()
        self.num_players = num_players

    def init_game(self):
        ''' Initialize players and state

        Returns:
            (tuple): Tuple containing:

                (dict): The first state in one game
                (int): Current player's id
        '''
        # Initalize wins
        self.wins = [False for _ in range(self.num_players)]

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize num_players players to play the game
        self.players = [Player(i, self.np_random)
                        for i in range(self.num_players)]

        # Deal 3 cards to each player to prepare for the game
        for player in self.players:
            self.dealer.deal_cards(player, CARD_PER_HAND)

        # Initialize a Round
        self.round = Round(self.dealer, self.num_players, self.np_random)

        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''

        self.round.proceed_round(self.players, action)
        player_id = self.round.current_player
        state = self.get_state(player_id)
        return state, player_id

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        state = self.round.get_state(self.players, player_id)
        state['player_num'] = self.get_player_num()
        state['current_player'] = self.round.current_player
        return state

    def get_wins(self):
        ''' Return the wins of the game

        Returns:
            (list): Each entry corresponds to the win state of one player
        '''
        winner = self.round.winner
        if winner is not None and len(winner) == 1:
            self.wins[winner[0]] = True
        return self.wins

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''

        return self.round.get_legal_actions(self.players,
                                            self.round.current_player)

    def get_player_num(self):
        ''' Return the number of players in Limit Texas Hold'em

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    def get_scores(self):
        return self.round.get_scores(self.players)

    @staticmethod
    def get_action_num():
        ''' Return the number of actions (not all legal)

        Returns:
            (int): The number of actions.
        '''
        return ACTION_NUM

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.round.current_player

    def is_over(self):
        ''' Check if the game is over

        Returns:
            (boolean): True if the game is over
        '''
        return self.round.is_over
