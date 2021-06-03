import numpy as np


class GridWorld:

    def __init__(self, n: int):
        """
        Grid World initializations
        ----------
        n : int
            Number of rows and columns of the Grid World
        """
        self.n = n
        self.action_probs = [0.25, 0.25, 0.25, 0.25]  # Default action probabilities

        # Terminal States
        self.tl_terminal = 0  # Top-left corner
        self.rb_terminal = (n * n) - 1  # Bottom-right corner
        self.terminal_states = [self.tl_terminal, self.rb_terminal]

        self.actions = ['up', 'down', 'left', 'right']  # Available actions

        # Initial Status
        self.current = None
        self.previous = None

    def reset(self):
        """
        Reset Method - Starts in a state that is not terminal
        """
        self.current = np.random.randint(1, (self.n * self.n) - 1)
        self.previous = None

        return self.current

    def sample_action(self):
        """
        Returns a sample action based on the action_probs.
        """
        return np.random.choice(self.actions, p=self.action_probs)

    def step(self, action: str):
        """
        Performs a movement in the Grid World and gets the new State and Reward
        ------------
        action : str
          One of the following actions: up, down, left, right
        """
        self.current = self.move(action)

        return self.current, self.get_reward(), self.is_terminal_state()

    def move(self, action: str):
        """
        Calculates the new State given a current State and an Action
        -----------
        action: str
          One of the following actions: up, down, left, right
        """
        self.previous = self.current

        if action not in self.actions:
            raise Exception(f"'{action}' is not a valid action!")

        if self.is_terminal_state():
            return self.current

        if action == 'up':
            return self.current - self.n if self.current > self.n - 1 else self.current
        elif action == 'down':
            return self.current + self.n if self.current < self.n * (self.n - 1) else self.current
        elif action == 'left':
            return self.current - 1 if self.current % self.n != 0 else self.current
        elif action == 'right':
            return self.current + 1 if (self.current + 1) % self.n != 0 else self.current

    def get_reward(self):
        """
        Gets Reward of the last movement
        """
        if self.is_terminal_state():
            return 5 if self.tl_terminal == self.current else 10  # If Terminal State returns different reward in each different corner

        # Returns -2 if the movement was outbounds, -1 in all other cases
        return -2 if self.previous == self.current else -1

    def is_terminal_state(self):
        return self.tl_terminal == self.current or self.rb_terminal == self.current

    def set_state(self, state: int):
        """
        Forces change to a specific state
        ---------
        state : int
           New state id
        """
        self.current = state
