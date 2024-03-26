"""Collection of plotting functions to analyze the introduction of asymmetries in memories for the implementation of
the central and local protocols using a 1D cluster topology with a Bell pair between the end nodes as the target state.
"""

import numpy as np
import seaborn as sns
import graphepp as gg
import matplotlib.pyplot as plt
import libs.aux_functions as aux
from noisy_graph_states import Strategy
from asymmetric_1d import asymmetric_1d_local
from asymmetric_1d import asymmetric_1d_central


def dephasing_times(N, range_dephasing_time, index):
    """
    This function outputs a list of lists of possible dephasing times in seconds.
    Each inner list corresponds to a set of dephasing times in which a certain qubit labeled with 'index' has a
    different dephasing time which is taken from 'range_dephasing_time'. The rest have a dephasing time of 0.1 s.

    Parameters
    ----------
    N : int
        Number of qubits, which include the inner neighbours and the target qubits.
    range_dephasing_time : list
        List of all the possible values of dephasing times of the faulty qubit.
    index : int
        Label of the faulty qubit.

    Returns
    -------
    dephasing_times : list
        List of lists of possible dephasing times.
    """
    dephasing_times = [[] for i in range(len(range_dephasing_time))]
    for i, t in enumerate(range_dephasing_time):
        for j in range(N):
            if j == index:
                dephasing_times[i].append(t)
            else:
                dephasing_times[i].append(0.1)
    return dephasing_times


# Let us define some parameters for the plotting function.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 24
legend_font_size = 24
legend_handle_length = 1.3
legend_label_spacing = 0.17
line_width = 3


def plot_nodes_fidelity_vs_memory(
    N, protocol, positions, coefficients, processing_times, num=50, save=False, path=""
):
    """
    This function plots the fidelity of a Bell pair obtained from an N-qubit 1D cluster in terms of the dephasing time
    of the qubit with bad memory. Each of the series represents each of the qubits in the cluster being the faulty
    qubit. The used protocol depends on the specified one, if the chosen protocol is the central one, the coordinator is
    the middle qubit.

    Note that the parameters here are prepared to plot the precise parameters in the main part of the file.

    Parameters
    ----------
    N : int
        Number of qubits, which include the inner neighbours and the target qubits.
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    positions : list
        List of the positions of all the nodes. Each position should specify an x and a y coordinate in a numpy.ndarray.
        The ordering of the elements of this list correspond to the entanglement ordering of the qubits.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    processing_time : list of scalars
        List of the processing times or times required to perform a measurement in seconds.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    num : int
        Number of points per axis to be plotted. Default is 50.
    save : True or False
        If True, the plot is saved in a .pdf file. If False, the plot is shown. Default is False.
    path : str
        Specification on where to save the plot. Default is ''.
    """
    # Define the index of the faulty qubit.
    q_move = range(N)
    # Range of the dephasing time of the faulty qubit.
    dt = np.logspace(-5, 0, num=num)
    # Create the sets of dephasing times.
    dts = [dephasing_times(N, dt, q) for q in q_move]
    # Use the all x measurement pattern.
    pattern = ["."] + ["x"] * (N - 2) + ["."]
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    sequence = aux.pattern_to_sequence(pattern, linear_cluster)
    strategy = Strategy(sequence=sequence, graph=linear_cluster)
    # Create an empty list to store the outcomes.
    fidelities = [[] for q in q_move]
    # Compute the fidelity for the different sets of dephasing times.
    for q in q_move:
        for t in dts[q]:
            if protocol == "central":
                f = asymmetric_1d_central(
                    N,
                    positions,
                    positions[int(N / 2)],
                    coefficients,
                    strategy,
                    t,
                    processing_times,
                )
            elif protocol == "local":
                f = asymmetric_1d_local(
                    N,
                    positions,
                    coefficients,
                    strategy,
                    t,
                    processing_times,
                )
            else:
                raise ValueError("protocol must be central or local.")
            # Save the outcome.
            fidelities[q].append(np.real_if_close(f["fidelity"]))
    # Plot the results.
    fig, ax = plt.subplots(figsize=[6.4, 4.8])
    if protocol == "central":
        colors1 = sns.blend_palette(["#2D004B", "white"], int(N / 2) + 2)
        colors = colors1[:-1] + colors1[::-1][2:]
        labels = (
            [""] * int(N / 2)
            + [f"Qubit {int(N/2) + 1}"]
            + [f"Qubit {q + 1} or {N - q}" for q in reversed(range(int(N / 2)))]
        )
    elif protocol == "local":
        colors1 = sns.blend_palette(["#2D004B", "white"], N)
        colors = [colors1[-2]] + colors1[:-1]
        labels = (
            [""] + [f"Qubit {q + 1}" for q in range(1, N - 1)] + [f"Qubit {1} or {N}"]
        )
    else:
        raise ValueError("protocol must be central or local.")
    for q in q_move:
        ax.plot(
            dt, fidelities[q], label=labels[q], linewidth=line_width, color=colors[q]
        )
    ax.set_xlabel(r"$T$ [s]")
    ax.set_ylabel("Fidelity")
    ax.set_xscale("log")
    # Customize ticks.
    ax.tick_params(axis="both", which="both", direction="in")
    ax.tick_params(axis="both", which="major", length=8)
    ax.tick_params(axis="x", which="minor", length=4)
    ax.tick_params(axis="x", which="both", pad=6)
    ax.set_xlim(left=9 * 10 ** (-5), right=0.11)
    ax.set_ylim(top=0.96, bottom=0.495)
    ax.set_yticks([0.5, 0.7, 0.9])
    # Add legend.
    ax.legend(
        prop={"size": legend_font_size},
        handlelength=legend_handle_length,
        labelspacing=legend_label_spacing,
    )
    # Save or show the plot.
    if save is True:
        fig.savefig(path + f"Memory_asymmetry_{protocol}_{N}.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


if __name__ == "__main__":
    # Here we use the functions of this file to produce the plots presented in the paper.
    # Set the protocol.
    protocols = ["central", "local"]
    # Set the size of the network.
    N = 7
    # Set the positions.
    positions = [np.array([0.0, i * 15000.0]) for i in range(N)]
    # Set depolarizing noises as the initial noise.
    p = 0.99
    c = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    coefficients = [c] * N
    # Set the processing times.
    processing_times = [0.000001] * (N - 2)
    for protocol in protocols:
        plot_nodes_fidelity_vs_memory(
            N, protocol, positions, coefficients, processing_times, num=100, save=False
        )
