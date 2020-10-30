import numpy as np

from whale.env import Env
from whale.game import WhaleGame as Game
from whale.utils import encode_hand
from whale.utils import ACTION_SPACE, ACTION_LIST
from whale.utils import cards2list

# todo inject env super class methods to remove deps with env


class WhaleEnv(Env):

    def __init__(self, config):
        self.name = 'whale'
        self.game = Game(num_players=config["num_players"])
        super().__init__(config)
        self.state_shape = [1, 3]

    def _load_model(self):
        ''' Load pretrained/rule model

        Returns:
            model (Model): A Model object
        '''
        raise Exception(NotImplemented)

    def _extract_state(self, state):
        # TODO simplify this
        hand = encode_hand(state['hand'])
        obs = np.array(hand + state['water'])
        legal_action_id = self._get_legal_actions()
        extracted_state = {'obs': obs, 'legal_actions': legal_action_id}
        if self.allow_raw_data:
            extracted_state['raw_obs'] = state
            extracted_state['raw_legal_actions'] = [
                a for a in state['legal_actions']]
        if self.record_action:
            extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def get_payoffs(self):

        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        legal_ids = self._get_legal_actions()
        if action_id in legal_ids:
            return ACTION_LIST[action_id]
        # TODO: clarify or remove this
        # if (len(self.game.dealer.deck) +
        # len(self.game.round.played_cards)) > 17:
        #    return ACTION_LIST[60]
        return ACTION_LIST[np.random.choice(legal_ids)]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = [ACTION_SPACE[action] for action in legal_actions]
        return legal_ids

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): Dictionary of all the perfect information of current state
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
