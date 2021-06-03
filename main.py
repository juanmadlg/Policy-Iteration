import numpy as np

from policy_iteration.Policy import Policy
from policy_iteration.PolicyIteration import PolicyIteration
from policy_iteration import GridWorld

N = 4

configuration = {
  "N": N,
  "theta": 0.001,
  "discount": 0.9
}

environment = GridWorld(configuration['N'])
policy = Policy(np.array([0.25, 0.25, 0.25, 0.25]), N)

policy, v = PolicyIteration(configuration, environment, policy).execute(verbose=True)
