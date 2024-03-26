"""Computation of averages of the implementations of the central and local protocols using a 1D cluster with basic
entanglement topology with a Bell pair two nodes as the target state in a general scenario.
"""


import numpy as np
from time import time
import graphepp as gg
from noisy_graph_states import Strategy
from libs.aux_functions import distance
from libs.aux_functions import pattern_to_sequence
from protocols import central_protocol, local_protocol


def no_shuffle_desired_average(
    hops,
    protocol,
    positions,
    shuffle,
    coefficients,
    dephasing_times,
    processing_times,
    cs=2e8,
    save=False,
    path="scenarios/custom_1d/",
):
    """This function computes the average fidelity of a Bell pair between two nodes of the network, achieved by
    connecting two qubits inside a 1D cluster which are placed in two nodes that are likely to require a pair, such that
    for an optimized resource state these two nodes would be spaced by a certain number of entanglement hops or edges of
    the optimized resource state. This is specified by "shuffle", which indicates the custom shuffling of qubits.
    We consider that the nodes of the network are distributed in a 2D plane following the structure of a 1D chain.
    Note that we consider a not-shuffled scenario, which means that the ordering of the qubits and the ordering of the
    nodes of the network coincide. However, we compute for a certain number of hops based of the custom shuffle case.
    All the possible configurations of two qubits are computed and thus, the averaged fidelity and the corresponding
    standard deviation are computed.
    The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent noise
    model.
    The manipulation of the resource state is done using the all X measurement pattern.
    The followed protocol is specified as one of the parameters.

    The way this function works is that take a large quantum networks and studies at each round of a loop a sub-part of
    the network that corresponds to a sub-cluster which takes nodes that in the custom shuffle are a certain number of
    hops away.

    Parameters
    ----------
    hops : int
        Number of hops between the two target qubits in the custom structure.
    protocol : str
        Determines if the protocol used is a local or central one.
    positions : list
        List of the positions of all the nodes. Each position should specify an x and a y coordinate in a numpy.ndarray.
    shuffle : list
        List which indicates the positions of the qubits in the physical structure for the optimized 1D cluster.
        Such that 'shuffle[i]' corresponds to the label of the qubit which is in position 'i' in the optimized cluster.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
    dephasing_times : list of scalar
        Dephasing times of the memories in seconds.
    processing_times : list of scalar
        Processing times or times required to perform a measurement in seconds.
    cs : scalar
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).
    save : True or False
        If True, the outcome is saved in a .npy file. Default is False.
    path : str
        Specification on where to save the .npy file. Default is 'scenarios/custom_1d/'.

    Returns
    -------
    dict or None:
        If none of the sub-clusters can be computed the function returns None.
        Else, result dictionary with the following keys:
        "run_time": scalar
            The run time in seconds.
        "average_fidelity": scalar
            Averaged fidelity of the output state.
        "std_fidelity": scalar
            Standard deviation of the computed fidelities.
    """
    # The number of hops should be larger than 0.
    if hops <= 0:
        raise ValueError("The number of hops must be equal or larger to 1.")
    # Define the size of the sub-cluster that is targeted in the optimized case.
    sub_graph = hops + 3
    # The size of the sub-cluster cannot be larger than the size of the original cluster.
    if sub_graph > len(positions):
        raise ValueError("The number of hops is too large.")
    start_time = time()
    # Create empty list to collect the data.
    fidelities = []
    # Iterate over the possible sub-clusters of the optimized 1D cluster of the fixed size.
    for graph_index in range(0, len(positions) - (hops + 2)):
        # Get the indices of the targeted nodes in the optimized case with the specified number of hops.
        target_a = shuffle.index(graph_index + 1)
        target_b = shuffle.index(graph_index + 1 + hops)
        target_nodes = sorted([target_a, target_b])
        # Discard the cases where the start and end nodes of the chain are the target nodes.
        # This is done, such that all the studied cases have two outer neighbors.
        if 0 in target_nodes or (len(positions) - 1) in target_nodes:
            continue
        # Define the indices of the nodes in the non-optimized structure.
        start_index = target_nodes[0] - 1
        end_index = target_nodes[1] + 2
        # Define the size of the sub-cluster in the non-optimized structure.
        N = end_index - start_index
        # Redefine positions, initial noise, dephasing and processing times for the non-optimized structure.
        new_positions = positions[start_index:end_index]
        new_coefficients = coefficients[start_index:end_index]
        new_dephasing_times = dephasing_times[start_index:end_index]
        new_processing_times = processing_times[start_index:end_index]
        # Define the all x measurement pattern for the size of the sub-cluster in the non-optimized structure.
        pattern = ["z", "."] + ["x"] * (N - 4) + [".", "z"]
        linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
        sequence = pattern_to_sequence(pattern, linear_cluster)
        strategy = Strategy(sequence=sequence, graph=linear_cluster)
        # Compute the outcome depending on the chosen protocol.
        if protocol == "local":
            # Compute the distances between neighbouring nodes in the sub-cluster.
            neighbour_distances = [
                distance(new_positions[i], new_positions[i + 1]) for i in range(N - 1)
            ]
            # Create two lists with the distances between each qubit and each target qubit.
            distances_1 = []
            distances_N_2 = []
            for qubit in range(N):
                # Sort the index of the loop compared to the target qubit 1.
                start_1, stop_1 = sorted([qubit, 1])
                # Add to the list the distance from the qubit of the loop to the target qubit 1.
                distances_1.append(sum(neighbour_distances[start_1:stop_1]))
                # Sort the index of the loop compared to the target qubit N-2.
                start_N_2, stop_N_2 = sorted([qubit, N - 2])
                # Add to the list the distance from the qubit of the loop to the target qubit N-2.
                distances_N_2.append(sum(neighbour_distances[start_N_2:stop_N_2]))
            # Compute the fidelity.
            outcome = local_protocol(
                N,
                distances_1,
                distances_N_2,
                new_coefficients,
                strategy,
                new_dephasing_times,
                new_processing_times,
                cs,
            )
        elif protocol == "central":
            # Determine the central node as the node placed in the middle of the 1D cluster.
            central_position = positions[int(len(positions) / 2)]
            # Compute the fidelity.
            outcome = central_protocol(
                N,
                new_positions,
                central_position,
                new_coefficients,
                strategy,
                new_dephasing_times,
                new_processing_times,
                cs,
            )
        else:
            # Raise an error if the value of "protocol" is neither "local" nor "central".
            raise ValueError("Specify the correct protocol, local or central.")
        # Save the fidelities.
        fidelities.append(outcome["fidelity"])
    if save is True:
        # Save the data of each computed fidelity in a .npy file.
        np.save(
            path + f"no_shuffle_desired_{protocol}_{len(positions)}_hops={hops}",
            fidelities,
        )
    if len(fidelities) > 0:
        # Compute the average fidelity.
        average_fidelity = sum(fidelities) / len(fidelities)
        # Compute the standard deviation.
        std_fidelity = np.std(fidelities)
        # Return the outcome and also the time that it took.
        return {
            "run_time": time() - start_time,
            "average_fidelity": average_fidelity,
            "std_fidelity": std_fidelity,
        }
    else:
        # If there was no fidelity computed, return None.
        return None


if __name__ == "__main__":
    import random_setting as rnd

    # Here we use the function for an example.
    # Set the number of hops.
    hops = 1
    # Set the size of the 1D cluster.
    N = 10
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
    # Set the shuffling.
    shuffle = rnd.random_shuffle(N)
    # Compute the outcome using the central protocol.
    result_shuffle_central = no_shuffle_desired_average(
        hops=hops,
        protocol="central",
        positions=positions,
        shuffle=shuffle,
        coefficients=coefficients,
        dephasing_times=dephasing_times,
        processing_times=processing_times,
    )
    print(result_shuffle_central)
    # Compute the outcome using the local protocol.
    result_shuffle_local = no_shuffle_desired_average(
        hops=hops,
        protocol="local",
        positions=positions,
        shuffle=shuffle,
        coefficients=coefficients,
        dephasing_times=dephasing_times,
        processing_times=processing_times,
    )
    print(result_shuffle_local)
