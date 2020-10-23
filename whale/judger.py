
class WhaleJudger(object):

    @staticmethod
    def judge_winner(players, np_random):
        ''' Judge the winner of the game

        Args:
            players (list): The list of players who play the game

        Returns:
            (list): The player id of the winner
        '''
        self.np_random = np_random
        for index, players in enumerate(players):
            # todo create variable for this
            if players.water == 5:
                return [index]
