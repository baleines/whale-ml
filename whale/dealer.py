
from whale.card import WhaleCard as Card


def init_deck():
    ''' Generate whale deck of 108 cards
    '''
    deck = []

    # init wave cards
    for _ in range(1, 32):
        deck.append(Card('wave'))

    # init double_wave cards
    for _ in range(1, 8):
        deck.append(Card('double_wave'))

    # init water cards
    for _ in range(1, 40):
        deck.append(Card('water'))

    return deck


class WhaleDealer(object):
    ''' Initialize a whale dealer class
    '''

    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_deck()
        self.shuffle()

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, player, num):
        ''' Deal some cards from deck to one player

        Args:
            player (object): The object of DoudizhuPlayer
            num (int): The number of cards to be dealed
        '''
        for _ in range(num):
            player.hand.append(self.deck.pop())
