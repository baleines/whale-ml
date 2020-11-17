
class NoDrawAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, action_num):
        ''' Initilize the random agent

        Args:
            action_num (int): The size of the ouput action space
        '''
        self.action_num = action_num

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random
            agent
        '''
        if 2 in state['legal_actions']:
            return 2
        if 1 in state['legal_actions']:
            return 1
        return 0

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is
            equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by
            the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.action_num)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])
        return self.step(state), probs
