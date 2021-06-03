import numpy as np


class Policy:
    def __init__(self, initial_probs: list, n: int):
        """
        Initializations
        ----------------
        initial_probs: list
          List with the initial probabilites for the 4 available actions
        n : int
          Size of the Grid World
        """
        # Copies sames probabilities in all states
        self.state_actions_probs = np.full((n*n, 4), initial_probs)

    def get_action(self, environment: object):
        """
        Returns action in terms of probabilities of the actions for each state (improved each cycle)
        ------------
        environment : object
          Environment to get available actions
        """

        return np.random.choice(environment.actions, p=self.state_actions_probs[environment.current])

    def get_action_probs(self, state, action):
        return self.state_actions_probs[state, action]
