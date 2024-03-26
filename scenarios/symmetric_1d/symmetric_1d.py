"""Implementations of the central and local protocols using a 1D cluster topology with a Bell pair between the end nodes
as the target state in a symmetric scenario with a natural entanglement topology.
"""

from time import time
import graphepp as gg
import noisy_graph_states as ngs
import libs.aux_functions as aux
import noisy_graph_states.libs.graph as gt


def symmetric_1d_central(
    N,
    central_index,
    distance,
    coefficients,
    strategy,
    dephasing_time,
    processing_time,
    cs=2e8,
):
    """This function computes the fidelity of a Bell pair achieved by connecting the end nodes of a 1D cluster of N
    qubits. The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent
    noise model.
    The measurements of the inner neighbors is determined by the specified strategy.
    The followed protocol is a central one, where all the classical communication is between each qubit to the central
    node or coordinator. This function assumes all nodes are placed in a line such that only the label of the
    coordinator needs to be specified.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    central_index : int
        Index of the coordinating node. Can take values from 0 to N-1.
    distance : scalar
        Distance between two consecutive nodes in the network.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
    strategy : Strategy
        Measurement strategy.
    dephasing_time : scalar
        Dephasing time of the memories in seconds.
    processing_time : scalar
        Processing time or time required to perform a measurement in seconds.
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
    # Generate the cluster.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Generate the state.
    state = ngs.State(graph=linear_cluster, maps=[])
    # Apply the chosen Pauli noise as the initial noise model.
    state = ngs.pauli_noise(state=state, indices=range(N), coefficients=coefficients)
    # Compute the communication time of each node to the central node or coordinator.
    communication_times = [abs(i - central_index) * distance / cs for i in range(N)]
    # Compute the waiting times for target qubits. Both wait for the inner neighbour that takes longer to receive the
    # message, measure and send outcome, plus the time for the central node to communicate the outcome to each.
    target_0 = (
        2 * max(communication_times[1:-1]) + processing_time + communication_times[0]
    )
    target_N_1 = (
        2 * max(communication_times[1:-1])
        + processing_time
        + communication_times[N - 1]
    )
    # Both target qubits should wait for each other, then they should have the same waiting time.
    target_time = max(target_0, target_N_1)
    # All other qubits: wait for the message and perform measurement.
    waiting_times = (
        [target_time]
        + [t + processing_time for t in communication_times[1:-1]]
        + [target_time]
    )
    # Final time of the protocol, corresponds to the maximal waiting time.
    final_time = max(waiting_times)
    # Apply the time dependent noise corresponding to the computed waiting times.
    for i, t in enumerate(waiting_times):
        state = ngs.z_noise(state, [i], aux.time_noise_coefficient(t, dephasing_time))
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


def symmetric_1d_local(
    N, distance, coefficients, strategy, dephasing_time, processing_time, cs=2e8
):
    """This function computes the fidelity of a Bell pair achieved by connecting the end nodes of a 1D cluster of N
    qubits. The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent
    noise model. The measurements of the inner neighbors is determined by the specified strategy.
    The followed protocol is a local one, where the actions start at one of the end nodes of the cluster.
    Since all measurements can be performed at once, once a node starts a measurement it already notifies the following
    node.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    distance : scalar
        Distance between two consecutive nodes in the network.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
    strategy : Strategy
        Measurement strategy.
    dephsing_time : scalar
        Dephasing time of the memories in seconds.
    processing_time : scalar
        Processing time or time required to perform a measurement in seconds.
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
            Total time required to do the protocol
    """
    start_time = time()
    if N < 3:
        raise ValueError("N should be larger than 2")
    # Generate the cluster.
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Generate the state.
    state = ngs.State(graph=linear_cluster, maps=[])
    # Apply the chosen Pauli noise as the initial noise model.
    state = ngs.pauli_noise(state=state, indices=range(N), coefficients=coefficients)
    # Compute the communication time between neighbouring nodes.
    communication_time = distance / cs
    # Compute the waiting times for target qubits. Both wait for the inner neighbour that takes longer to receive the
    # message, measure and send outcome.
    target_0 = 2 * (N - 2) * communication_time + processing_time
    target_N_1 = (N - 1) * communication_time + processing_time
    # Both target qubits should wait for each other, then they should have the same waiting time.
    target_time = max(target_0, target_N_1)
    # All other qubits: wait for the message and perform measurement.
    waiting_times = (
        [target_time]
        + [i * communication_time + processing_time for i in range(1, N - 1)]
        + [target_time]
    )
    # Final time of the protocol, corresponds to the maximal waiting time.
    final_time = max(waiting_times)
    # Apply the time dependent noise corresponding to the computed waiting times.
    for i, t in enumerate(waiting_times):
        state = ngs.z_noise(state, [i], aux.time_noise_coefficient(t, dephasing_time))
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
    N = 5
    # Set the distance.
    distance = 15000.0
    # Set the dephasing time.
    dephasing_time = 0.1
    # Set depolarizing noise as the initial noise.
    p = 0.99
    coefficients = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    # Set the processing time.
    processing_time = 0.000001
    # Use the all x measurement pattern.
    pattern = ["."] + ["x"] * (N - 2) + ["."]
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    sequence = aux.pattern_to_sequence(pattern, linear_cluster)
    strategy = ngs.Strategy(sequence=sequence, graph=linear_cluster)
    # Compute the outcome using the central protocol.
    result_central = symmetric_1d_central(
        N,
        int(N / 2),
        distance,
        coefficients,
        strategy,
        dephasing_time,
        processing_time,
    )
    print(result_central)
    # Compute the outcome using the local protocol.
    result_local = symmetric_1d_local(
        N, distance, coefficients, strategy, dephasing_time, processing_time
    )
    print(result_local)
