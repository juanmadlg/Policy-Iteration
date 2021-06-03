import numpy as np


class IterativePolicyEvaluation:
    def __init__(self, n: int, theta: float, discount: float, policy: object, environment: object):
        """
        Initialization
        """
        self.n = n  # Size of the GridWorld
        self.theta = theta  # Limit value for the delta between values of a state
        self.discount = discount  # Discount from the previous values of a state

        self.V = None  # Stores value function
        self.Vc = None # Stores the previous values of V

        self.policy = policy
        self.environment = environment

    def reset(self):
        """
        Reset is executed at the beginning of each episode.
        """
        self.V = np.zeros((self.n * self.n,))
        self.Vc = np.zeros((self.n * self.n, ))

    def get_action(self):
        """
        Gets the next action from the Policy
        """
        return self.policy.get_action(self.environment)

    def eval(self):
        """
        Evaluates the current policy calculating the function value for each state
        """
        self.reset()
        self.environment.reset()

        end = False
        while not end:
            delta = 0

            for state in range(self.n * self.n):
                # Calculates the value for all possible actions in the state
                values_by_action = [self.calc_value(self.environment, state, action) * self.policy.get_action_probs(
                    state, self.environment.actions.index(action)) for action in self.environment.actions]

                self.Vc[state] = round(np.sum(values_by_action), 4)
                # Gets the maximum difference between current and previous values
                delta = max(delta, abs(self.Vc[state] - self.V[state]))

            self.V = self.Vc
            self.Vc = np.zeros((self.n * self.n, ))

            # Only ends if the delta is lower than theta (small value)
            end = delta < self.theta

        return self.V

    def calc_value(self, env: object, state: int, action: str):
        """
        Calculates the function value of the state
        """
        env.set_state(state)
        if env.is_terminal_state():
            return 0

        new_state, reward, _ = env.step(action)

        return reward + self.discount * self.V[new_state]
