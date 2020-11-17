from pathlib import Path
import numpy as np


from whale.card import WhaleCard as Card

# Read required docs
ROOT_PATH = Path(__file__).parent

# a map of trait to its index
CARD_MAP = {'water': 0, 'wave': 1, 'double_wave': 2}


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


def set_global_seed(seed):
    ''' Set the global see for reproducing results

    Args:
        seed (int): The seed

    Note: If using other modules with randomness, they also need to be seeded
    '''
    if seed is not None:
        import subprocess
        import sys

        reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
        if 'tensorflow' in installed_packages:
            import tensorflow as tf
            tf.random.set_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
