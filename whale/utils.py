from pathlib import Path

# Read required docs
ROOT_PATH = Path(__file__).parent

# a map of trait to its index
CARD_MAP = {'water': 0, 'wave': 1, 'double_wave': 2}


def cards2list(cards):
    ''' Get the corresponding string representation of cards

    Args:
        cards (list): list of WhaleCards objects

    Returns:
        (string): string representation of cards
    '''
    cards_list = []
    for card in cards:
        cards_list.append(card.get_str())
    return cards_list


def hand2dict(hand):
    ''' Get the corresponding dict representation of hand

    Args:
        hand (list): list of string of hand's card

    Returns:
        (dict): dict of hand
    '''
    hand_dict = {}
    for card in hand:
        if card not in hand_dict:
            hand_dict[card] = 1
        else:
            hand_dict[card] += 1
    return hand_dict


def encode_hand(hand):
    ''' Encode hand and represerve it into plane
    TODO move this function to a better place
    Args:
        hand (list): list of string of hand's card

    Returns:
        (array): 3 numpy array
    '''
    plane = [0, 0, 0]
    hand = hand2dict(hand)
    # populate each card
    for card in CARD_MAP.items():
        # print(f'card{card}')
        if hand.get(card[0]):
            plane[card[1]] = hand[card[0]]
    return plane
