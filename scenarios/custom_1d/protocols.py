"""Implementations of the central and local protocols using a 1D cluster topology with a Bell pair between the second
and second to last nodes as the target state in a general scenario.
"""

import graphepp as gg
import libs.aux_functions as aux
import noisy_graph_states as ngs
import noisy_graph_states.libs.graph as gt


def local_protocol(
    N,
    distances_1,
    distances_N_2,
    coefficients,
    strategy,
    dephasing_times,
    processing_times,
    cs=2e8,
):
    """This function computes the fidelity of a Bell pair achieved by connecting two qubits in a 1D cluster
    such that these two qubits are spaced by N - 4 inner neighbours, and the two targets have an outer neighbour each.
    We consider that the nodes of the network are distributed in a 2D plane following the structure of a 1D chain.
    The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent noise
    model. The manipulation of the resource state is done using the specified strategy.
    The followed protocol is a local one, where the actions start at one of the target qubits. Since all measurements
    can be performed at once, once a node starts a measurement it already notifies the following node.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.
    distances_1 : list
        List of the distance of the physical channels between a qubit and the target qubit that starts the protocol.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    distances_N_2 : list
        List of the distance of the physical channels between a qubit and the target qubit that does not start the
        protocol. The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    strategy : Strategy
        Measurement order of the manipulation.
    dephasing_times : list of scalar
        Dephasing times of the memories in seconds.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    processing_times : list of scalar
        Processing times or times required to perform a measurement in seconds.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    cs : scalar
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).

    Returns
    -------
    dict
        result dictionary with the following keys
        "fidelity": scalar
            Fidelity of the output state.
        "final_time": scalar
            Total time required to do the protocol.
    """
    # Generate the cluster.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Generate the state.
    state = ngs.State(graph=linear_cluster, maps=[])
    # Apply the chosen Pauli noise as the initial noise model.
    for i, c in enumerate(coefficients):
        state = ngs.pauli_noise(state=state, indices=[i], coefficients=c)
    # Create a list of the qubits that need to be measured in the manipulation.
    measure = [0] + [i for i in range(2, N - 2)] + [N - 1]
    # Compute the waiting times for target qubits.
    # Target 1: wait to the inner neighbour that takes longer to receive the message, measure and send outcome.
    # If the measurement of the outer neighbour takes longer, that has to be taken into account.
    target_1 = max(2 * distances_1[i] / cs + processing_times[i] for i in measure[:-1])
    # Target N - 2: wait to the inner neighbour that takes longer to receive the message, measure and send outcome.
    # If the measurement of the outer neighbour takes longer, that has to be taken into account.
    target_N_2 = max(
        (distances_1[i] + distances_N_2[i]) / cs + processing_times[i]
        for i in measure[1:]
    )
    # Both target qubits should wait for each other, then they should have the same waiting time.
    target_time = max(target_1, target_N_2)
    # All other qubits: wait for the message and perform measurement.
    waiting_times = [distances_1[i] / cs + processing_times[i] for i in measure]
    waiting_times.insert(1, target_time)
    waiting_times.insert(N - 2, target_time)
    # Final time of the protocol, corresponds to the maximal waiting time.
    final_time = max(waiting_times)
    # Apply the time dependent noise corresponding to the computed waiting times.
    for i, t, dt in zip(range(N), waiting_times, dephasing_times):
        state = ngs.z_noise(state, [i], aux.time_noise_coefficient(t, dt))
    # Perform the manipulation according to the chosen strategy.
    output_state = strategy(state)
    # Compute the noisy density matrix.
    noisy_dm = ngs.noisy_bp_dm(output_state, [1, N - 2])
    # Compute the fidelity.
    fidelity = aux.fidelity(gt.bell_pair_ket, noisy_dm)
    return {
        "fidelity": fidelity[0][0],
        "final_time": final_time,
    }


def central_protocol(
    N,
    positions,
    central_position,
    coefficients,
    strategy,
    dephasing_times,
    processing_times,
    cs=2e8,
):
    """This function computes the fidelity of a Bell pair achieved by connecting two qubits in a 1D cluster
    such that these two qubits are spaced by N - 4 inner neighbours, and the two targets have an outer neighbour each.
    We consider that the nodes of the network are distributed in a 2D plane following the structure of a 1D chain.
    The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent noise
    model. The manipulation of the resource state is done using the specified strategy.
    The followed protocol is a central one, where all the classical communication is between each qubit to the central
    node.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.
    positions : list
        List of the positions of all the nodes. Each position should specify an x and a y coordinate in a numpy.ndarray.
    central_position : numpy.ndarray
        Position of the orchestrator of the protocol.
    coefficients : list of scalars
        List of the coefficients of the initial noise model
    strategy : Strategy
        Measurement order of the manipulation.
    dephasing_times : list of scalar
        Dephasing times of the memories in seconds.
    processing_times : list of scalar
        Processing times or times required to perform a measurement in seconds.
    cs : scalar
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).

    Returns
    -------
    dict
        result dictionary with the following keys
        "fidelity": scalar
            Fidelity of the output state.
        "final_time": scalar
            Total time required to do the protocol.
    """
    # Generate the cluster.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Generate the state.
    state = ngs.State(graph=linear_cluster, maps=[])
    # Apply the chosen Pauli noise as the initial noise model.
    for i, c in enumerate(coefficients):
        state = ngs.pauli_noise(state=state, indices=[i], coefficients=c)
    # Compute all the distances between each node and the central one.
    distances = [aux.distance(p, central_position) for p in positions]
    # Compute the communication time of each distance.
    communication_times = [d / cs for d in distances]
    # Create a list of the indices of the measured qubits.
    measured = [0] + [i for i in range(2, N - 2)] + [N - 1]
    # Compute the waiting times for target qubits.
    # Target 1: wait for the inner neighbour that takes longer to receive the message, measure and send outcome, plus
    # the time for the central node to communicate the outcome to 1.
    target_1 = (
        max(
            (2 * communication_times[i] + processing_times[i] for i in measured[1:]),
            default=0,
        )
        + communication_times[1]
    )
    # Target 2: wait for the inner neighbour that takes longer to receive the message, measure and send outcome, plus
    # the time for the central node to communicate the outcome to N - 2.
    target_2 = (
        max(
            (2 * communication_times[i] + processing_times[i] for i in measured[:-1]),
            default=0,
        )
        + communication_times[N - 2]
    )
    # Both target qubits should wait for each other, then they should have the same waiting time.
    target_time = max(target_1, target_2)
    # All other qubits: wait for the message and perform measurement.
    waiting_times = (
        [communication_times[0] + processing_times[0]]
        + [target_time]
        + [communication_times[i] + processing_times[i] for i in range(2, N - 2)]
        + [target_time]
        + [communication_times[N - 1] + processing_times[N - 1]]
    )
    # Final time of the protocol, corresponds to the maximal waiting time.
    final_time = max(waiting_times)
    # Apply the time dependent noise corresponding to the computed waiting times.
    for i, t, dt in zip(range(N), waiting_times, dephasing_times):
        state = ngs.z_noise(state, [i], aux.time_noise_coefficient(t, dt))
    # Perform the manipulation according to the chosen strategy.
    output_state = strategy(state)
    # Compute the noisy density matrix.
    noisy_dm = ngs.noisy_bp_dm(output_state, [1, N - 2])
    # Compute the fidelity.
    fidelity = aux.fidelity(gt.bell_pair_ket, noisy_dm)
    return {
        "fidelity": fidelity[0][0],
        "final_time": final_time,
    }


if __name__ == "__main__":
    import random_setting as rnd

    # Here we make an example on how to use the functions of this file.
    # Set the size of the 1D cluster.
    N = 5
    # Set depolarizing noise as the initial noise.
    p = 0.99
    c = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    coefficients = [c] * N
    # Set the processing times.
    processing_times = [0.000001] * N
    # Set the node positioning.
    positions = rnd.random_positions(N)
    # Set the dephasing times.
    dephasing_times = rnd.random_dephasing_times(N)
    # Use the all x measurement pattern for the inner neighbors.
    pattern = ["z", "."] + ["x"] * (N - 4) + [".", "z"]
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    sequence = aux.pattern_to_sequence(pattern, linear_cluster)
    strategy = ngs.Strategy(sequence=sequence, graph=linear_cluster)
    # Compute the result for the central protocol.
    result_central = central_protocol(
        N,
        positions,
        positions[int(N / 2)],
        coefficients,
        strategy,
        dephasing_times,
        processing_times,
    )
    print(result_central)
