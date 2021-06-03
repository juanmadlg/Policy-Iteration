def print_v(v, n: int):
    """
    Print Grid
    -----------
    v : np_array
    Array that stores the function value for each state.
    n : int
    Size of the Grid World
    """
    grid = ""
    for i, v in enumerate(v):
        grid = grid + f"{v:.2f}" + (f'\n' if (i + 1) % n == 0 else '\t')

    print(grid)


def print_policy(policy, n: int):
    """
    Prints the Policy in a "grid" mode
    --------
    policy
        As an array
    n : int
        Size of the Grid World
    """
    grid = ""
    actions = [u"\u2191", u"\u2193", u"\u2190", u"\u2192"]

    for i, state_policy in enumerate(policy):
        grid = grid + f"{(''.join([actions[i] for i, value in enumerate(state_policy) if value != 0])).center(4, ' ')}" + (f'\n' if (i + 1) % n == 0 else '\t')

    print(grid)
