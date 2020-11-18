import run_random as random
import run_simple as simple
import run_dqn as dqn


def test_random():
    random.run_model()


def test_simple():
    simple.run_model()


def test_dqn():
    dqn.run_model()
