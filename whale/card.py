class WhaleCard(object):

    info = {'type':  ['water', 'wave', "double_wave"],
            }

    def __init__(self, card_type):
        ''' Initialize the class of WhaleCard

        Args:
            card_type (str): The type of card
        '''
        self.type = card_type
        self.str = self.get_str()

    def get_str(self):
        ''' Get the string representation of card

        Return:
            (str): The string of card's type
        '''
        return self.type

    @staticmethod
    def print_cards(cards):
        ''' Print out card in a nice form

        Args:
            card (str or list): The string form or a list of a Whale card
        '''
        if isinstance(cards, str):
            cards = [cards]
        for i, card in enumerate(cards):
            print(card)
            if i < len(cards) - 1:
                print(', ', end='')
