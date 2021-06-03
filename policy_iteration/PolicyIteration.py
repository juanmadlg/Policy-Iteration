import numpy as np
from policy_iteration import IterativePolicyEvaluation
from policy_iteration.Utils import print_v, print_policy


class PolicyIteration:
    def __init__(self, configuration: dict, environment: object, policy: object):
        """
        Initialization
        ---------------
        configuration : dict
          Dictionary that includes configuration settings
        environment : object
          The Grid World in our case
        policy : object
          The policy that is going to be improved
        """
        self.environment = environment
        self.policy = policy

        self.V = None

        self.N = configuration['N']
        self.theta = configuration['theta']
        self.discount = configuration['discount']

    def policy_evaluation(self):
        """
        Evaluates the policy. Returns the Function Value for all States
        """
        return IterativePolicyEvaluation(self.N, self.theta, self.discount, self.policy, self.environment).eval()

    def policy_improvement(self):
        """
        Improves a Policy using the current Function Value.
        Returns the new version of the Policy and it has been updated or not.
        """

        # Set max value to terminal states in order to force actions to them.
        self.V[self.environment.terminal_states] = np.max(self.V)+1

        policy_stable = True  # Used to check if the Policy has changed or not in this step

        for state in range(1, (self.N * self.N) - 1):
            old_actions_probs = np.copy(self.policy.state_actions_probs[state])

            self.policy.state_actions_probs[state] = self.get_greedy_actions(state)

            if not np.array_equal(old_actions_probs, self.policy.state_actions_probs[state]):
                policy_stable = False

        print(f"The policy is stable: {policy_stable}.")
        return self.policy, policy_stable

    def get_greedy_actions(self, state):
        """
        Returns action probabilities in order to move to the states that will provide the higher value
        """
        state_action_values = self.get_action_values(state)  # What are the value that we could get from current state

        max_action_value = max(state_action_values)  # What is the higher value
        max_value_indices = [i for i, value in enumerate(state_action_values) if
                             value == max_action_value]  # Gets their indices

        # Prepares action probabilites for the ones with the higher value
        action_probs = np.zeros((4,))
        action_probs[max_value_indices] = 1 / (len(max_value_indices) if type(max_value_indices) is list else 1)

        return action_probs

    def get_action_values(self, state):
        """
        Gets the values for each posible action from the current state
        """
        up = self.V[state - self.N] if state > self.N - 1 else self.V[state]
        down = self.V[state + self.N] if state < self.N * (self.N - 1) else self.V[state]
        left = self.V[state - 1] if state % self.N != 0 else self.V[state]
        right = self.V[state + 1] if (state + 1) % self.N != 0 else self.V[state]

        return [up, down, left, right]

    def execute(self, verbose=False):
        policy_stable = False

        while not policy_stable:
            self.V = self.policy_evaluation()

            if verbose:
                print("Policy Value Function")
                print_v(self.V, self.N)

            self.policy, policy_stable = self.policy_improvement()

            if verbose:
                print("Policy")
                print_policy(self.policy.state_actions_probs, self.N)

        return self.policy, self.V

