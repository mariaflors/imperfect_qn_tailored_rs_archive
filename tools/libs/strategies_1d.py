"""A collection of measurement patterns on 1D cluster states.
"""

import graphepp as gg
from itertools import permutations
from noisy_graph_states import State
from noisy_graph_states import Strategy
from libs.aux_functions import pattern_to_sequence


def _all_patterns_1d_pair(N):
    """This function gives a list of all possible measurement patterns of inner neighbours consisting of 'y' and 'x'
    measurements. It also includes the patterns of only 'y' and only 'x'.
    These measurement patterns manipulate a 1D cluster to a Bell pair between the end nodes.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.

    Returns
    -------
    all_patterns : list
        List of all possible measurement patterns.
    """
    # Create an empty list to save the patterns.
    all_patterns = []
    # Establish the counter to 0.
    r = 0
    # Loop that ends when the counter is larger than the number of measured qubits.
    while r < N - 1:
        # Create a list of all possible permutations of the corresponding number of 'y' and 'x' measurements.
        patterns = permutations(["y"] * (N - 2 - r) + ["x"] * r)
        # Add the empty measurements for the target qubits.
        patterns = [["."] + list(p) + ["."] for p in patterns]
        # Save the patterns avoiding repetition.
        for i in patterns:
            if i not in all_patterns:
                all_patterns.append(i)
        r += 1
    return all_patterns


def all_strategies_1d_pair(N):
    """This function gives a list of all possible strategies of inner neighbours related to all the possible measurement
    patterns consisting of 'y' and 'x' measurements. It also includes the measurement patterns of only 'y' and only 'x'.
    These strategies manipulate a 1D cluster to a Bell pair between the end nodes.
    The function checks internally that the strategy leads to the desired target state.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.

    Returns
    -------
    good_strategies : list
        List of all possible strategies.
    """
    # Create the linear cluster, where the two end, labelled with 0 and N - 1, are the target qubits.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Generate the initial state.
    state = State(graph=linear_cluster, maps=[])
    # Create the final Bell pair.
    bell_pair = gg.Graph(N=N, E=[(0, N - 1)])
    # Generate the target state.
    target_state = State(graph=bell_pair, maps=[])
    # Get all patterns.
    patterns = _all_patterns_1d_pair(N)
    # Transform each pattern to a sequence.
    sequences = [pattern_to_sequence(p, linear_cluster) for p in patterns]
    # Create a strategy for each sequence.
    strategies = [Strategy(sequence=s, graph=linear_cluster) for s in sequences]
    # Create an empty list to save the strategies that lead to the target state.
    good_strategies = []
    for strategy in strategies:
        # Apply each strategy and check if it leads to the target state or not.
        final_state = strategy(state)
        if final_state == target_state:
            good_strategies.append(strategy)
        else:
            print("bad strategy", strategy)
    return good_strategies
