"""Collection of functions to analyze and search the optimal measurement pattern in the introduction of asymmetries in
memory and position for the implementation of the central and local protocols using a 1D cluster topology with a Bell
pair between the end nodes as the target state.
"""

import numpy as np
from memory_asymmetry import dephasing_times
from position_asymmetry import range_positions
from asymmetric_1d import asymmetric_1d_local
from asymmetric_1d import asymmetric_1d_central
from libs.aux_functions import sequence_to_pattern
from libs.strategies_1d import all_strategies_1d_pair


def optimal_pattern(
    N, protocol, positions, coefficients, dephasing_times, processing_times
):
    """Find best and worst fidelities, strategies, and patterns for the specified protocol in an asymmetric 1D cluster.
    For the central protocol, this function assumes that all nodes are placed in a line and the coordinator is the one
    in the center, the one corresponding to the label int(N / 2).

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    positions : list
        List of the positions of all the nodes. Each position should specify an x and a y coordinate in a numpy.ndarray.
        The ordering of the elements of this list correspond to the entanglement ordering of the qubits.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
        The ordering of the elements of this list correspond to the entanglement ordering of the qubits.
    dephasing_times : List of scalar
        Dephasing times of the memories in seconds. The length of this list is N.
        The ordering of the elements of this list correspond to the entanglement ordering of the qubits.
    processing_times : List of scalar
        Processing times or times required to perform a measurement in seconds. The length of this list is N - 2.
        There is no processing time for the target qubits.

    Returns
    -------
     dict
        result dictionary with the following keys
        "best_fidelity": scalar
            The optimal fidelity.
        "best_strategy": list of tuple
            List of all the optimal strategies.
        "best_pattern": list
            List of the measurement patterns of the optimal strategies.
        "worst_fidelity": scalar
            The worst fidelity.
        "worst_strategy": list of tuple
            List of all the worst strategies.
        "worst_pattern": list
            List of the measurement patterns of the worst strategies.
    """
    # Create a list of all the possible strategies.
    strategies = all_strategies_1d_pair(N)
    # Create an empty list to store all possible fidelities.
    fidelities = []
    # Compute the outcome for all strategies depending on the chosen protocol.
    for s in strategies:
        if protocol == "central":
            outcome = asymmetric_1d_central(
                N,
                positions,
                positions[int(N / 2)],
                coefficients,
                s,
                dephasing_times,
                processing_times,
            )
        elif protocol == "local":
            outcome = asymmetric_1d_local(
                N, positions, coefficients, s, dephasing_times, processing_times
            )
        else:
            raise ValueError("protocol must be central or local.")
        fidelities.append(outcome["fidelity"])

    # Find best and worst fidelities.
    best_f = max(fidelities)
    worst_f = min(fidelities)
    # Create empty lists to store outcomes.
    best_strategies = []
    best_patterns = []
    worst_strategies = []
    worst_patterns = []
    # Find the best and worst strategies and patterns.
    for i, f in enumerate(fidelities):
        if f == best_f:
            strategy = strategies[i]
            pattern = sequence_to_pattern(strategy.sequence, strategy.graph)
            best_strategies.append(strategy)
            if pattern not in best_patterns:
                best_patterns.append(pattern)
        elif f == worst_f:
            strategy = strategies[i]
            pattern = sequence_to_pattern(strategy.sequence, strategy.graph)
            worst_strategies.append(strategy)
            if pattern not in worst_patterns:
                worst_patterns.append(pattern)
    return {
        "best_fidelity": best_f,
        "best_strategy": best_strategies,
        "best_pattern": best_patterns,
        "worst_fidelity": worst_f,
        "worst_strategy": worst_strategies,
        "worst_pattern": worst_patterns,
    }


def position_optimal_pattern(N, protocol, num=50):
    """This function searches for the optimal measurement pattern considering asymmetries in the position of one of the
    qubits.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    num : int
        Number of points to search over. Default is 50.

    Returns
    -------
    not_x : list
        List containing all the optimal measurement patterns that are not all X and the corresponding asymmetric qubit
        and its position.
    """
    # Define the index of the asymmetric qubit.
    q_move = range(N)
    # Range of the position of the asymmetric qubit.
    position_move = np.linspace(5000, 15000, num=num)
    # Create the sets of positions.
    positions = [range_positions(N, q, position_move) for q in q_move]
    # Define the dephasing times.
    dephasing_times = [0.1] * N
    # Define the initial depolarizing noise.
    p = 0.99
    c = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    coefficients = [c] * N
    # Define the processing times.
    processing_times = [0.000001] * N
    # Search the optimal pattern.
    not_x = []
    for q in q_move:
        for p in positions[q]:
            optimal = optimal_pattern(
                N, protocol, p, coefficients, dephasing_times, processing_times
            )["best_pattern"]
            # If the optimal pattern is not all x, print the result.
            if optimal != [["."] + ["x"] * (N - 2) + ["."]]:
                not_x.append([q, p[q], optimal])
    return not_x


def memory_optimal_pattern(N, protocol, num=50):
    """This function searches for the optimal measurement pattern considering asymmetries in the memory of one of the
    qubits.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    num : int
        Number of points to search over. Default is 50.

    Returns
    -------
    not_x : list
        List containing all the optimal measurement patterns that are not all X and the corresponding asymmetric qubit
        and its dephasing time.
    """
    # Define the index of the faulty qubit.
    q_move = range(N)
    # Range of the dephasing time of the faulty qubit.
    dt = np.logspace(-4, -1, num=num)
    # Create the sets of dephasing times.
    dts = [dephasing_times(N, dt, q) for q in q_move]
    # Define the positions.
    positions = [np.array([0.0, j * 15000.0]) for j in range(N)]
    # Define the initial depolarizing noise.
    p = 0.99
    c = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    coefficients = [c] * N
    # Define the processing times.
    processing_times = [0.000001] * (N - 2)
    # Search the optimal pattern.
    not_x = []
    for q in q_move:
        for t in dts[q]:
            optimal = optimal_pattern(
                N, protocol, positions, coefficients, t, processing_times
            )["best_pattern"]
            # If the optimal pattern is not all x, print the result.
            if optimal != [["."] + ["x"] * (N - 2) + ["."]]:
                not_x.append([q, t[q], optimal])
    return not_x


if __name__ == "__main__":
    Ns = [4, 5, 6, 7, 8, 9]
    num = 20
    protocols = ["central", "local"]
    for protocol in protocols:
        for N in Ns:
            print(
                N,
                protocol,
                "position:",
                position_optimal_pattern(N, protocol, num=num),
                "memory:",
                memory_optimal_pattern(N, protocol, num),
            )
