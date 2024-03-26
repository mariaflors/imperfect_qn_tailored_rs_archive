"""Implementations of the central and local protocols using a 1D cluster topology with a Bell pair between the end nodes
as the target state in an asymmetric scenario with a natural entanglement topology.
"""

import numpy as np
import graphepp as gg
from time import time
import libs.aux_functions as aux
import noisy_graph_states as ngs
import noisy_graph_states.libs.graph as gt


def asymmetric_1d_local(
    N, positions, coefficients, strategy, dephasing_times, processing_times, cs=2e8
):
    """This function computes the fidelity of a Bell pair achieved by connecting the end nodes of a 1D cluster of N
    qubits. The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent
    noise model.
    The measurements of the inner neighbors is determined by the specified strategy.
    The followed protocol is a local one, where the actions start at one of the end nodes of the cluster.
    Since all measurements can be performed at once, once a node starts a measurement it already notifies the following
    node.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    positions : list
        List of the positions of all the nodes. Each position should specify an x and a y coordinate in a numpy.ndarray.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
    strategy : Strategy
        Measurement strategy.
    dephasing_times : list of scalar
        Dephasing times of the memories in seconds. The length of this list is N.
    processing_times : list of scalar
        Processing times or times required to perform a measurement in seconds. The length of this list is N - 2.
        There is no processing time for the target qubits.
    cs : scalar
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).

    Returns
    -------
    dict
        result dictionary with the following keys
        "run_time": scalar
            The run time in seconds.
        "fidelity": scalar
             The fidelity of the output state.
        "final_time": scalar
            Total time required to do the protocol.
    """
    start_time = time()
    if N < 3:
        raise ValueError("N should be larger than 2")
    # Create the linear cluster, where the two end, labelled with 0 and N - 1, are the target qubits.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Generate the state.
    state = ngs.State(graph=linear_cluster, maps=[])
    # Apply the chosen Pauli noise as the time-independent noise model.
    for i, c in enumerate(coefficients):
        state = ngs.pauli_noise(state=state, indices=[i], coefficients=c)
    # Compute the distances between neighbouring nodes.
    distances = [aux.distance(positions[i], positions[i + 1]) for i in range(N - 1)]
    # Compute distances from inner neighbour to target 0.
    distances_0 = [sum(distances[:i]) for i in range(1, N - 1)]
    # Compute distances from inner neighbour to target N - 1.
    distances_N_1 = [sum(distances[i:]) for i in range(1, N - 1)]
    # Compute the waiting times for each qubit.
    # Target 0: wait for the inner neighbour that takes longer to receive the message, measure and send outcome.
    target_0 = max(2 * distances_0[i] / cs + processing_times[i] for i in range(N - 2))
    # Target N - 1: wait for the inner neighbour that takes longer to receive the message, measure and send outcome.
    target_N_1 = max(
        (distances_0[i] + distances_N_1[i]) / cs + processing_times[i]
        for i in range(N - 2)
    )
    # Both target qubits should wait for each other, then they should have the same waiting time.
    target_time = max(target_0, target_N_1)
    # Qubits 1 to N - 2: wait for the message and perform measurement.
    waiting_times = (
        [target_time]
        + [distances_0[i] / cs + processing_times[i] for i in range(N - 2)]
        + [target_time]
    )
    # Final time of the protocol, corresponds to the maximal waiting time.
    final_time = max(waiting_times)
    # Apply the time dependent noise corresponding to the computed waiting times.
    for i, t, dt in zip(range(N), waiting_times, dephasing_times):
        state = ngs.z_noise(state, [i], aux.time_noise_coefficient(t, dt))
    # Perform the manipulation according to the chosen strategy.
    output_state = strategy(state)
    # Compute the noisy density matrix.
    noisy_dm = ngs.noisy_bp_dm(output_state, [0, N - 1])
    # Compute the fidelity.
    fidelity = aux.fidelity(gt.bell_pair_ket, noisy_dm)
    return {
        "run_time": time() - start_time,
        "fidelity": fidelity[0][0],
        "final_time": final_time,
    }


def asymmetric_1d_central(
    N,
    positions,
    central_position,
    coefficients,
    strategy,
    dephasing_times,
    processing_times,
    cs=2e8,
):
    """This function computes the fidelity of a Bell pair achieved by connecting the end nodes of a 1D cluster of N
    qubits. The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent
    noise model.
    The measurements of the inner neighbors is determined by the specified strategy.
    The followed protocol is a central one, where all the classical communication is between each qubit to the
    coordinator or central node.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    positions : list
        List of the positions of all the nodes. Each position should specify an x and a y coordinate in a numpy.ndarray.
    central_position : numpy.ndarray
        x and y coordinates of the position of the central node.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
    strategy : Strategy
        Measurement strategy.
    dephasing_times : List of scalar
        Dephasing times of the memories in seconds. The length of this list is N.
    processing_times : List of scalar
        Processing times or times required to perform a measurement in seconds. The length of this list is N - 2.
        There is no processing time for the target qubits.
    cs : scalar
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).

    Returns
    -------
    dict
        result dictionary with the following keys
        "run_time": scalar
            The run time in seconds.
        "fidelity": scalar
            The fidelity of the output state.
        "final_time": scalar
            Total time required to do the protocol.
    """
    start_time = time()
    if N < 3:
        raise ValueError("N should be larger than 2")
    # Create the linear cluster, where the two end, labelled with 0 and N - 1, are the target qubits.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Generate the state.
    state = ngs.State(graph=linear_cluster, maps=[])
    # Apply the chosen Pauli noise as the time-independent noise model.
    for i, c in enumerate(coefficients):
        state = ngs.pauli_noise(state=state, indices=[i], coefficients=c)
    # Compute all the distances between each node and the central one.
    distances = [aux.distance(p, central_position) for p in positions]
    # Compute the communication time of each distance.
    communication_times = [d / cs for d in distances]
    # Compute the waiting times for each qubit.
    # Target: wait for the inner neighbour that takes longer to receive the message, measure and send outcome, plus
    # the time for the central node to communicate the outcome to 0.
    target_0 = (
        max(2 * ct + pt for ct, pt in zip(communication_times[1:-1], processing_times))
        + communication_times[0]
    )
    # Target N - 1: wait for the inner neighbour that takes longer to receive the message, measure and send outcome, plus
    # the time for the central node to communicate the outcome to N - 1.
    target_N_1 = (
        max(2 * ct + pt for ct, pt in zip(communication_times[1:-1], processing_times))
        + communication_times[N - 1]
    )
    # Both target qubits should wait for each other, then they should have the same waiting time.
    target_time = max(target_0, target_N_1)
    # Qubits 1 to N - 2: wait for the message and perform measurement.
    waiting_times = (
        [target_time]
        + [ct + pt for ct, pt in zip(communication_times[1:-1], processing_times)]
        + [target_time]
    )
    # Final time of the protocol, corresponds to the maximal waiting time.
    final_time = max(waiting_times)
    # Apply the time dependent noise corresponding to the computed waiting times.
    for i, t, dt in zip(range(N), waiting_times, dephasing_times):
        state = ngs.z_noise(state, [i], aux.time_noise_coefficient(t, dt))
    # Perform the manipulation according to the chosen strategy.
    output_state = strategy(state)
    # Compute the noisy density matrix.
    noisy_dm = ngs.noisy_bp_dm(output_state, [0, N - 1])
    # Compute the fidelity.
    fidelity = aux.fidelity(gt.bell_pair_ket, noisy_dm)
    return {
        "run_time": time() - start_time,
        "fidelity": fidelity[0][0],
        "final_time": final_time,
    }


if __name__ == "__main__":
    # Here we make an example on how to use the functions of this file.
    # Set the size of the 1D cluster.
    N = 3
    # Set the positions.
    positions = [np.array([1, 2]), np.array([3, 4]), np.array([3.5, 1.5])]
    # Set the central position.
    central_position = positions[int(N / 2)]
    # Set the dephasing times.
    dephasing_times = [100e-3] * N
    # Set depolarizing noises as the initial noise.
    p = 0.99
    coefficients = [[p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]] * N
    # Set the processing times.
    processing_times = [0.000001] * (N - 2)
    # Use the all x measurement pattern.
    pattern = ["."] + ["x"] * (N - 2) + ["."]
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    sequence = aux.pattern_to_sequence(pattern, linear_cluster)
    strategy = ngs.Strategy(sequence=sequence, graph=linear_cluster)
    # Compute the outcome using the local protocol.
    print(
        asymmetric_1d_local(
            N, positions, coefficients, strategy, dephasing_times, processing_times
        )
    )
    # Compute the outcome using the central protocol.
    print(
        asymmetric_1d_central(
            N,
            positions,
            central_position,
            coefficients,
            strategy,
            dephasing_times,
            processing_times,
        )
    )
