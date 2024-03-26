"""Plots of averages and percentages of the implementations of the central and local protocols using a 1D cluster
as a resource state, with both the basic and the custom entanglement topologies, targeting a Bell pair between two
nodes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from shuffle import shuffle_average_outcome
from no_shuffle import no_shuffle_desired_average


def shuffle_vs_no_shuffle_desired(
    hops,
    protocol,
    positions,
    shuffle,
    coefficients,
    dephasing_times,
    processing_times,
    cs=2e8,
    save=False,
    path="scenarios/custom_1d/setting/",
):
    """This function produces the data corresponding to the average fidelity and corresponding standard deviations for a
    set of number of hops for both the custom shuffle and the no-shuffle cases.
    The target is a Bell pair between two nodes of the network, achieved by connecting two qubits inside a 1D cluster
    which are placed in two nodes that are likely to require a pair, such that for an optimized resource state these two
    nodes would be spaced by a certain number of entanglement hops or edges of the optimized resource state.
    This is specified by "shuffle", which indicates the custom shuffling of qubits.
    We consider that the nodes of the network are distributed in a 2D plane following the structure of a 1D chain.
    Note that we compute this for both the custom shuffle case and the no-shuffle case. Importantly, in the shuffled
    scenario the ordering of the qubits and the ordering of the nodes of the network do not coincide, whereas in the
    not-shuffled scenario the ordering of the qubits and the ordering of the nodes of the network coincide. However, for
    the latter we compute for a certain number of hops based of the custom shuffle case.
    All the possible configurations of two qubits are computed and thus, the averaged fidelity and the corresponding
    standard deviation are computed for both the custom shuffle and the no-shuffle case.
    The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent noise
    model.
    The manipulation of the resource state is done using the all X measurement pattern.
    The followed protocol is specified as one of the parameters.

    The way this function works is that take a large quantum networks and studies at each round of a loop a sub-part of
    the network that corresponds to a sub-cluster which takes nodes that in the custom shuffle are a certain number of
    hops away.

    Parameters
    ----------
    hops : list
        List of number of hops between the two target qubits in the custom structure.
    protocol : str
        Determines if the protocol used is a local or central one.
    positions : list
        List of the positions of all the nodes. Each position should specify an x and a y coordinate in a numpy.ndarray.
    shuffle : list
        List which indicates the positions of the qubits in the physical structure for the optimized 1D cluster.
        Such that 'shuffle[i]' corresponds to the label of the qubit which is in position 'i'.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    dephasing_times : list of scalar
        Dephasing times of the memories in seconds.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    processing_times : list of scalar
        Processing times or times required to perform a measurement in seconds.
        The ordering of this list correspond to the entanglement ordering of the qubits in the 1D cluster.
    cs : scalar
        Communication speed in m/s. Default is 2e8 (speed of light in optical fibre).
    save : True or False
        If True, the outcomes are saved in a .npy file. Default is False.
    path : str
        Specification on where to save the .npy file. Default is 'scenarios/custom_1d/setting/'.

    Returns
    -------
    hops_shuffled : list
        List of the hops for the custom shuffle case.
    shuffled : list
        List of the fidelity of each number of hops for the custom shuffle case.
    std_shuffled : list
        List of the standard deviation of each fidelity of each number of hops for the custom shuffle case.
    hops_not_shuffled : list
        List of the hops for the no-shuffle case.
    not_shuffled : list
        List of the fidelity of each number of hops for the no-shuffle case.
    std_not_shuffled : list
        List of the standard deviation of each fidelity of each number of hops for the no-shuffle case.
    """
    # Create empty lists to store data.
    hops_shuffled = []
    hops_not_shuffled = []
    shuffled = []
    std_shuffled = []
    not_shuffled = []
    std_not_shuffled = []
    # Iterate over the possible number of hops.
    for hop in hops:
        # Compute the outcome for the custom shuffle case.
        result_shuffled = shuffle_average_outcome(
            hops=hop,
            protocol=protocol,
            positions=positions,
            shuffle=shuffle,
            coefficients=coefficients,
            dephasing_times=dephasing_times,
            processing_times=processing_times,
            cs=cs,
            save=save,
            path=path,
        )
        # Compute the outcome for the no-shuffle case.
        result_not_shuffled = no_shuffle_desired_average(
            hops=hop,
            protocol=protocol,
            positions=positions,
            shuffle=shuffle,
            coefficients=coefficients,
            dephasing_times=dephasing_times,
            processing_times=processing_times,
            cs=cs,
            save=save,
            path=path,
        )
        # Save the outcomes.
        hops_shuffled.append(hop)
        shuffled.append(np.real_if_close(result_shuffled["average_fidelity"]))
        std_shuffled.append(result_shuffled["std_fidelity"])
        if result_not_shuffled is not None:
            # Note that for the no-shuffle case, the function that computes the outcome can return None.
            hops_not_shuffled.append(hop)
            not_shuffled.append(
                np.real_if_close(result_not_shuffled["average_fidelity"])
            )
            std_not_shuffled.append(result_not_shuffled["std_fidelity"])
    if save is True:
        # Save the data.
        np.save(
            path + f"desired_shuffle_{protocol}_{N}",
            (
                hops_shuffled,
                shuffled,
                std_shuffled,
                hops_not_shuffled,
                not_shuffled,
                std_not_shuffled,
            ),
        )
    return (
        hops_shuffled,
        shuffled,
        std_shuffled,
        hops_not_shuffled,
        not_shuffled,
        std_not_shuffled,
    )


# Let us define some parameters for the plotting function.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"


def plot_shuffle_vs_no_shuffle(
    protocol, limit, path="scenarios/custom_1d/setting/", save=False
):
    """This function plots the data corresponding to the average fidelity and corresponding standard deviations for a
    set of number of hops for both the custom shuffle and the no-shuffle cases.
    The target is a Bell pair between two nodes of the network, achieved by connecting two qubits inside a 1D cluster
    which are placed in two nodes that are likely to require a pair, such that for an optimized resource state these two
    nodes would be spaced by a certain number of entanglement hops or edges of the optimized resource state.
    This is specified by "shuffle", which indicates the custom shuffling of qubits.
    We consider that the nodes of the network are distributed in a 2D plane following the structure of a 1D chain.
    Note that we compute this for both the custom shuffle case and the no-shuffle case. Importantly, in the shuffled
    scenario the ordering of the qubits and the ordering of the nodes of the network do not coincide, whereas in the
    not-shuffled scenario the ordering of the qubits and the ordering of the nodes of the network coincide. However, for
    the latter we compute for a certain number of hops based of the custom shuffle case.
    All the possible configurations of two qubits are computed and thus, the averaged fidelity and the corresponding
    standard deviation are computed for both the custom shuffle and the no-shuffle case.
    The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent noise
    model.
    The manipulation of the resource state is done using the all X measurement pattern.
    The followed protocol is specified as one of the parameters.

    Parameters
    ----------
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    limit : int
        Number that chooses where the data is taken until.
    path : str
        Specification on where to take the data from and where to save the plot.
        Default is 'scenarios/custom_1d/setting/'.
    save : True or False
        If True, the plot is saved in a .pdf file. If False, the plot is shown. Default is False.
    """
    plt.rcParams["font.size"] = 12
    legend_label_spacing = 0.4

    data = np.load(path + f"desired_shuffle_{protocol}_100.npy")
    hops_shuffled = data[0]
    shuffled = data[1]
    std_shuffled = data[2]
    hops_not_shuffled = data[3]
    not_shuffled = data[4]
    std_not_shuffled = data[5]
    # Cut the data lists.
    hops_shuffled = hops_shuffled[:limit]
    shuffled = shuffled[:limit]
    std_shuffled = std_shuffled[:limit]
    hops_not_shuffled = hops_not_shuffled[:limit]
    not_shuffled = not_shuffled[:limit]
    std_not_shuffled = std_not_shuffled[:limit]
    # Plot the data with error bars.
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(
        hops_not_shuffled,
        not_shuffled,
        label="Basic",
        color="#F18B14",
        marker="s",
        markersize=5.8,
        linestyle="None",
    )
    plt.fill_between(
        hops_not_shuffled,
        not_shuffled - std_not_shuffled,
        not_shuffled + std_not_shuffled,
        facecolor="#F18B1433",
        edgecolor="#F18B144D",
        linewidth=0.6,
    )
    plt.plot(
        hops_shuffled,
        shuffled,
        label="Custom",
        color="#2D004B",
        marker="o",
        markersize=6.5,
        linestyle="None",
    )
    plt.fill_between(
        hops_shuffled,
        shuffled - std_shuffled,
        shuffled + std_shuffled,
        facecolor="#2D004B33",
        edgecolor="#2D004B4D",
        linewidth=0.6,
    )
    plt.xlim(left=0.6, right=limit - 0.4)
    if protocol == "central":
        plt.ylim(bottom=0.59, top=0.96)
        plt.yticks([0.6, 0.7, 0.8, 0.9])
    elif protocol == "local":
        plt.ylim(bottom=0.3, top=0.93)
        plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    else:
        raise ValueError("protocol must be central or local.")
    plt.xticks([i for i in range(0, limit, 5)][1:])
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tick_params(axis="both", which="major", length=4)
    plt.tick_params(axis="x", which="both", pad=6)
    plt.xlabel("Hops (custom case)")  # Label the x-axis.
    plt.ylabel("Average fidelity")  # Label the y-axis.
    plt.legend(
        labelspacing=legend_label_spacing,
        borderpad=0.4,
        handlelength=2,
        handletextpad=0.1,
    )
    # Save or show the plot.
    if save is True:
        plt.savefig(path + f"Custom_vs_basic_{protocol}.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


def plot_usable_fidelity(N, hops, path="scenarios/custom_1d/setting/", save=False):
    """This function plots the percentage of pairs with a fidelity greater or equal to 0.5 for a set of number of hops
    for both the custom shuffle and the no-shuffle cases and the two protocols, local and central.
    The target is a Bell pair between two nodes of the network, achieved by connecting two qubits inside a 1D cluster
    which are placed in two nodes that are likely to require a pair, such that for an optimized resource state these two
    nodes would be spaced by a certain number of entanglement hops or edges of the optimized resource state.
    We consider that the nodes of the network are distributed in a 2D plane following the structure of a 1D chain.
    In the shuffled scenario the ordering of the qubits and the ordering of the nodes of the network do not coincide,
    whereas in the not-shuffled scenario the ordering of the qubits and the ordering of the nodes coincide.
    However, for the latter we compute for a certain number of hops based of the custom shuffle case.
    The noise considered is some initial Pauli noise acting on each cluster qubit and a dephasing time-dependent noise
    model.
    The manipulation of the resource state is done using the all X measurement pattern.

    Parameters
    ----------
    N : int
        Number of qubits in the 1D cluster.
    hops : int
        Number of hops between the two target qubits in the custom structure.
    path : str
        Specification on where to take the data from and where to save the plot.
        Default is 'scenarios/custom_1d/setting/'.
    save : True or False
        If True, the plot is saved in a .pdf file. If False, the plot is shown. Default is False.
    """
    # Create empty lists to save data. Make a list for each protocol and case.
    total_shuffle_central = []
    total_no_shuffle_central = []
    total_shuffle_local = []
    total_no_shuffle_local = []
    # Iterate over the possible hops.
    for hop in hops:
        # Import the corresponding data.
        shuffle_central = np.load(path + f"shuffle_desired_central_{N}_hops={hop}.npy")
        no_shuffle_central = np.load(
            path + f"no_shuffle_desired_central_{N}_hops={hop}.npy"
        )
        shuffle_local = np.load(path + f"shuffle_desired_local_{N}_hops={hop}.npy")
        no_shuffle_local = np.load(
            path + f"no_shuffle_desired_local_{N}_hops={hop}.npy"
        )
        # Create empty counters for each data set.
        usable_shuffle_central = 0
        usable_no_shuffle_central = 0
        usable_shuffle_local = 0
        usable_no_shuffle_local = 0
        # Iterate over all the values and count how many are greater or equal to 0.5.
        for f in shuffle_central:
            if f >= 0.5:
                usable_shuffle_central += 1
        for f in no_shuffle_central:
            if f >= 0.5:
                usable_no_shuffle_central += 1
        for f in shuffle_local:
            if f >= 0.5:
                usable_shuffle_local += 1
        for f in no_shuffle_local:
            if f >= 0.5:
                usable_no_shuffle_local += 1
        # Save the percentage of fidelities greater or equal to 0.5.
        total_shuffle_central.append(
            usable_shuffle_central / len(shuffle_central) * 100
        )
        total_no_shuffle_central.append(
            usable_no_shuffle_central / len(no_shuffle_central) * 100
        )
        total_shuffle_local.append(usable_shuffle_local / len(shuffle_local) * 100)
        total_no_shuffle_local.append(
            usable_no_shuffle_local / len(no_shuffle_local) * 100
        )
    # Plot the 4 data sets, plot separately the points and the lines, so lines can be a bit transparent.
    plt.rcParams["font.size"] = 12
    line_width = 2.7
    colors = ["#DE9151", "#A41623", "#7BC5BD", "#297481"]
    marker_size_square = 5.8
    marker_size_triangle = 6.5
    alpha = 0.3
    plt.plot(
        hops,
        total_no_shuffle_central,
        alpha=alpha,
        linewidth=line_width,
        color=colors[0],
    )
    plt.plot(
        hops, total_shuffle_central, alpha=alpha, linewidth=line_width, color=colors[1]
    )
    plt.plot(
        hops, total_no_shuffle_local, alpha=alpha, linewidth=line_width, color=colors[2]
    )
    plt.plot(
        hops, total_shuffle_local, alpha=alpha, linewidth=line_width, color=colors[3]
    )
    (p1,) = plt.plot(
        hops,
        total_no_shuffle_central,
        marker="D",
        markersize=marker_size_square,
        linestyle="None",
        color=colors[0],
    )
    (p2,) = plt.plot(
        hops,
        total_shuffle_central,
        marker="v",
        markersize=marker_size_triangle,
        linestyle="None",
        color=colors[1],
    )
    (p3,) = plt.plot(
        hops,
        total_no_shuffle_local,
        marker="s",
        markersize=marker_size_square,
        linestyle="None",
        color=colors[2],
    )
    (p4,) = plt.plot(
        hops,
        total_shuffle_local,
        marker="^",
        markersize=marker_size_triangle,
        linestyle="None",
        color=colors[3],
    )
    # Define this to create subtitles in the legend.
    class LegendTitle(object):
        def __init__(self, text_props=None):
            self.text_props = text_props or {}
            super(LegendTitle, self).__init__()

        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            title = mtext.Text(
                x0,
                y0,
                r"\underline{" + orig_handle + "}",
                usetex=True,
                **self.text_props,
            )
            handlebox.add_artist(title)
            return title

    plt.legend(
        ["Central protocol", p1, p2, "Local protocol", p3, p4],
        ["", "Basic", "Custom", "", "Basic", "Custom"],
        handler_map={str: LegendTitle()},
        ncol=2,
        columnspacing=3,
        markerscale=1.3,
    )
    plt.xlabel("Hops (custom case)")  # Label the x-axis.
    plt.ylabel(r"Pairs with fidelity $\geq 0.5$ [\%]")  # Label the y-axis.
    plt.xlim(left=0.6, right=limit - 0.4)
    plt.ylim(top=102, bottom=-2)
    plt.xticks([i for i in range(0, limit, 5)][1:])
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tick_params(axis="both", which="major", length=4)
    plt.tick_params(axis="x", which="both", pad=6)
    # Save or show the plot.
    if save is True:
        plt.savefig(path + f"Usable_pairs.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


if __name__ == "__main__":
    import pickle

    # Here we use the functions of this file to produce the data and the plots presented in the paper.
    N = 100
    hops = [i for i in range(1, int(N / 2))]
    path = "scenarios/custom_1d/setting/"
    ### Generate the data ###
    # First we import the random setting we have already generated.
    # Set the shuffling.
    with open(path + "shuffle_100", "rb") as fp:
        shuffle = pickle.load(fp)
    # Set depolarizing noise as the initial noise.
    p = 0.99
    c = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    coefficients = [c] * N
    # Set the processing times.
    processing_times = [0.000001] * N
    # Set the node positioning.
    positions = np.load(path + "positions_100.npy")
    # Set the dephasing times.
    dephasing_times = np.load(path + "dephasing_times_100.npy")
    for protocol in ["central", "local"]:
        shuffle_vs_no_shuffle_desired(
            hops,
            protocol,
            positions,
            shuffle,
            coefficients,
            dephasing_times,
            processing_times,
            save=False,
            path=path,
        )

    ### Plot the data ###
    limit = 41
    # Plot the average fidelity.
    for protocol in ["central", "local"]:
        plot_shuffle_vs_no_shuffle(protocol, limit, save=False)

    # Plot the percentage of pairs.
    plot_usable_fidelity(N, hops[:limit], path, save=False)
