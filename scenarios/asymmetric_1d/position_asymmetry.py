"""Collection of plotting functions to analyze the introduction of asymmetries in position for the implementation of
the central and local protocols using a 1D cluster topology with a Bell pair between the end nodes as the target state.
"""

import numpy as np
import graphepp as gg
import matplotlib.pyplot as plt
import libs.aux_functions as aux
from noisy_graph_states import Strategy
from asymmetric_1d import asymmetric_1d_local
from asymmetric_1d import asymmetric_1d_central


def range_positions(N, index, ps):
    """This function gives a list of possible positions of a 1D cluster place in a line, such that the distances between
    nodes are always 15km but for qubit "index". The position of the latter it moves from being 5km from the previous
    node, to being 5 km to the next node.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    index : int
        Label of the qubit which position changes.
    ps : list
        List of the range in which the position of said qubit ranges.

    Returns
    -------
    positions : List
        List in which element is a list comprising the positions of a certain instance of the described 1D cluster.
        Each position should specify an x and a y coordinate in a numpy.ndarray.
    """
    positions = [[] for i in range(len(ps))]
    for i, p in enumerate(ps):
        for j in range(N):
            if j == index:
                positions[i].append(np.array([0.0, (j - 1) * 15000.0 + p]))
            else:
                positions[i].append(np.array([0.0, j * 15000.0]))
    return positions


# Let us define some parameters for the plotting function.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 24
legend_font_size = 24
legend_handle_length = 1.3
legend_label_spacing = 0.2
line_width = 3


def plot_nodes_fidelity_vs_distance(
    N,
    protocol,
    coefficients,
    dephasing_times,
    processing_times,
    num=50,
    save=False,
    path="",
):
    """This function plots the fidelity of a Bell pair obtained from an N-qubit 1D cluster in terms of the shifted
    position of one of the nodes. Each of the series represents each of the qubits in the cluster being the asymmetric
    one. The used protocol depends on the specified one, if the chosen protocol is the central one, the coordinator is
    chosen to be the middle qubit.

    Note that the parameters here are prepared to plot the precise parameters in the main part of the file.

    Parameters
    ----------
    N : int
        Number of qubits, which include the inner neighbours and the target qubits.
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    dephasing_times : List of scalar
        Dephasing times of the memories in seconds.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    processing_times : List of scalar
        Processing times or times required to perform a measurement in seconds.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    num : int
        Number of points per axis to be plotted. Default is 50.
    save : True or False
        If True, the plot is saved in a .pdf file. If False, the plot is shown. Default is False.
    path : str
        Specification on where to save the plot. Default is ''.
    """
    # Define the index of the asymmetric qubit.
    q_move = range(N)
    # Range of the position of the asymmetric qubit.
    position_move = np.linspace(5000, 25000, num=num)
    # Create the sets of positions.
    positions = [range_positions(N, q, position_move) for q in q_move]
    # Use the all x measurement pattern.
    pattern = ["."] + ["x"] * (N - 2) + ["."]
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    sequence = aux.pattern_to_sequence(pattern, linear_cluster)
    strategy = Strategy(sequence=sequence, graph=linear_cluster)
    # Create an empty list to store the outcomes.
    fidelities = [[] for q in q_move]
    # Compute the fidelity for the different sets of dephasing times.
    for q in q_move:
        for p in positions[q]:
            if protocol == "central":
                f = asymmetric_1d_central(
                    N,
                    p,
                    p[int(N / 2)],
                    coefficients,
                    strategy,
                    dephasing_times,
                    processing_times,
                )
            elif protocol == "local":
                f = asymmetric_1d_local(
                    N,
                    p,
                    coefficients,
                    strategy,
                    dephasing_times,
                    processing_times,
                )
            else:
                raise ValueError("protocol must be central or local.")
            # Save the outcome.
            fidelities[q].append(np.real_if_close(f["fidelity"]))
    # Plot the results.
    fig, ax = plt.subplots(figsize=[6.4, 4.8])
    colors = ["#DE9151", "#297481", "#7BC5BD", "#8EA604", "#A41623"]
    line_styles = [":", "--", "-", (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    if protocol == "central":
        ax.set_ylim(top=0.96711, bottom=0.96544)
        ax.set_yticks([0.967, 0.966])
        labels = [f"Qubit {q + 1}" for q in range(N)]
    elif protocol == "local":
        ax.set_ylim(top=0.9657, bottom=0.96235)
        ax.set_yticks([0.965, 0.964, 0.963])
        colors = [colors[0]] + [colors[1]] * (N - 3) + [colors[2]] + [colors[3]]
        line_styles = (
            [line_styles[0]]
            + [line_styles[1]] * (N - 3)
            + [line_styles[2]]
            + [line_styles[3]]
        )
        labels = ["Qubit 1", "Qubit 2 or 3", "", "Qubit 4", "Qubit 5"]
    else:
        raise ValueError("protocol must be central or local.")
    for q in q_move:
        ax.plot(
            position_move,
            fidelities[q],
            label=labels[q],
            color=colors[q],
            linewidth=line_width,
            linestyle=line_styles[q],
        )
    ax.set_xlabel(r"Distance to previous node [km]")
    ax.set_ylabel("Fidelity")
    # Customize ticks.
    ax.yaxis.set_ticks_position("both")  # Put ticks on left and right axis.
    ax.tick_params(axis="both", which="both", direction="in")
    ax.tick_params(axis="both", which="major", length=8)
    ax.tick_params(axis="x", which="minor", length=4)
    ax.tick_params(axis="x", which="both", pad=6)
    ax.set_xlim(left=5000, right=25000)
    ax.set_xticks([5000, 15000, 25000])
    # Define the ticks in the upper and lower axis.
    def left(x):
        return x

    def right(x):
        return abs(30000 - x)

    sec_ax = ax.secondary_xaxis("top", functions=(left, right))
    sec_ax.set_xlabel("Distance to next node [km]")
    sec_ax.set_xticks([5000, 15000, 25000])
    sec_ax.tick_params(which="both", direction="in")
    sec_ax.tick_params(axis="both", which="major", length=8)
    sec_ax.tick_params(axis="x", which="minor", length=4)
    sec_ax.tick_params(axis="x", which="both", pad=6)
    # Redefine ticks to be in km instead of m.
    m2km = lambda x, _: f"{x / 1000:g}"
    ax.xaxis.set_major_formatter(m2km)
    sec_ax.xaxis.set_major_formatter(m2km)
    # Add legend.
    ax.legend(
        prop={"size": legend_font_size},
        handlelength=legend_handle_length,
        labelspacing=legend_label_spacing,
    )
    # Save or show the plot.
    if save is True:
        fig.savefig(
            path + f"Position_asymmetry_{protocol}_{N}.pdf", bbox_inches="tight"
        )
    else:
        plt.show()
    return


if __name__ == "__main__":
    # Here we use the functions of this file to produce the plots presented in the paper.
    # Set the protocol.
    protocols = ["central", "local"]
    # Set the size of the network.
    N = 5
    # Set the dephasing times.
    dts = [0.1] * N
    # Set depolarizing noises as the initial noise.
    p = 0.99
    c = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    coefficients = [c] * N
    # Set the processing times.
    pts = [0.000001] * (N - 2)
    # Plot the results.
    for protocol in protocols:
        plot_nodes_fidelity_vs_distance(
            N, protocol, coefficients, dts, pts, num=100, save=False
        )
