from whale.card import WhaleCard
from whale.utils import cards2list

# possible actions
DRAW = 0
SINGLE_WATER = 1
DOUBLE_WATER = 2


class WhaleRound(object):
    '''
    Represents a round in the game.
    '''

    def __init__(self, dealer, num_players, np_random):
        ''' Initialize the round class

        Args:
            dealer (object): the object of WhaleDealer
            num_players (int): the number of players in game
        '''
        self.np_random = np_random
        self.dealer = dealer
        self.current_player = 0
        self.num_players = num_players
        self.direction = 1
        self.played_cards = []
        self.is_over = False
        self.winner = None

    def proceed_round(self, players, action):
        ''' Call other Classes's functions to keep one round running

        Args:
            player (object): object of WhalePlayer
            action (str): string of legal action
        '''
        player = players[self.current_player]

        # perform the number action
        if action != DRAW:
            self._preform_action(players, action)
        # perform draw action
        else:
            if not self.played_cards and not self.dealer.deck:
                self.is_over = True
            else:
                self._perform_draw_action(players)

        # todo: create variable for this
        if player.water >= 5:
            self.is_over = True
            self.winner = [self.current_player]

        self.current_player = (
            self.current_player + self.direction) % self.num_players

    def get_legal_actions(self, players, player_id):
        legal_actions = []
        hand = players[player_id].hand
        wave = 0
        double_wave = 0
        water = 0

        # get available wave
        for card in hand:
            if card.type == 'wave':
                wave += 1
                continue
            if card.type == 'double_wave':
                double_wave += 1
                continue
            if card.type == 'water':
                water += 1

        # draw is always legal
        legal_actions = [DRAW]
        # simplify only allow 2 water with double wave
        if double_wave >= 1 and water >= 2:
            legal_actions.append(DOUBLE_WATER)

        if wave >= 1 and water >= 1:
            legal_actions.append(SINGLE_WATER)

        # print(f'{player_id}:legal_actions:{legal_actions}')
        return legal_actions

    def get_state(self, players, player_id):
        ''' Get player's state

        Args:
            players (list): The list of WhalePlayer
            player_id (int): The id of the player
        '''
        state = {}
        player = players[player_id]
        state['hand'] = cards2list(player.hand)
        state['played_cards'] = cards2list(self.played_cards)
        state['legal_actions'] = self.get_legal_actions(players, player_id)
        state['score'] = player.water
        return state

    @staticmethod
    def get_scores(players):
        water_levels = []
        for player in players:
            water_levels.append(player.water)
        return water_levels

    def replace_deck(self):
        ''' Add cards have been played to deck
        '''
        self.dealer.deck.extend(self.played_cards)
        self.dealer.shuffle()
        self.played_cards = []

    def _perform_draw_action(self, players):
        # replace deck if there is no card in draw pile
        if not self.dealer.deck:
            self.replace_deck()
        card = self.dealer.deck.pop()
        players[self.current_player].hand.append(card)
        # print(
        #     f'{self.current_player}:draw:{players[self.current_player].hand}')

    def _preform_action(self, players, action):
        current = self.current_player
        player = players[current]
        # print(f'{current}:{action}')
        if action == SINGLE_WATER:
            self._remove_hand(player, 'wave')
            self._remove_hand(player, 'water')
            player.water += 1
            # recycle wave but not water
            self.played_cards.append(WhaleCard('wave'))
            return
        elif action == DOUBLE_WATER:
            self._remove_hand(player, 'double_wave')
            self._remove_hand(player, 'water')
            self._remove_hand(player, 'water')
            player.water += 2
            # recycle wave but not water
            self.played_cards.append(WhaleCard('double_wave'))
            return

        raise Exception(NotImplementedError(
            f'action:{action} is not implemented')
        )

    def _remove_hand(self, player, card_name):
        for index, card in enumerate(player.hand):
            if card_name == card.get_str():
                player.hand.pop(index)
                return
