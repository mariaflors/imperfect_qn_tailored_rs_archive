"""Collection of functions to analyze, search and plot the optimal measurement pattern for the implementation of the
central and local protocols using a 1D cluster topology with a Bell pair between the end nodes as the target state in a
symmetric scenario.
"""


import numpy as np
import graphepp as gg
from matplotlib import patches
from matplotlib import colors
import matplotlib.pyplot as plt
from noisy_graph_states import Strategy
from symmetric_1d import symmetric_1d_local
from symmetric_1d import symmetric_1d_central
from libs.aux_functions import sequence_to_pattern
from libs.aux_functions import pattern_to_sequence
from libs.strategies_1d import all_strategies_1d_pair


def optimal_pattern(
    N, protocol, distance, coefficients, dephasing_time, processing_time
):
    """Find best and worst fidelities, strategies, and measurement patterns for the specified protocol in a symmetric 1D
    cluster.
    For the central protocol, this function assumes that all nodes are placed in a line and the coordinator is the one
    in the center, the one corresponding to the label int(N / 2).

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    distance : scalar
        Distance to the central node in meters.
    coefficients : list of scalars
        List of the coefficients of the initial noise model.
    dephasing_time : scalar
        Dephasing time of the memories in seconds.
    processing_time : scalar
        Processing time or time required to perform a measurement in seconds.

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
            outcome = symmetric_1d_central(
                N,
                int(N / 2),
                distance,
                coefficients,
                s,
                dephasing_time,
                processing_time,
            )
        elif protocol == "local":
            outcome = symmetric_1d_local(
                N, distance, coefficients, s, dephasing_time, processing_time
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


def generate_difference_optimal(N, protocol, distance, num=50, save=False, path=""):
    """Generate the data for a 2D plot of the difference between fidelities from the two best measurement patterns for
    the manipulation of a symmetric N-qubit 1D cluster as a resource state using a central or local protocol.
    The difference is computed for different values of initial depolarizing noise and the dephasing time which
    determines the time-dependent dephasing noise the involved qubits are subject to. The processing times are set to
    10^{-6} s. For the central protocol, we consider that all nodes are placed in a line and the coordinator is the one
    corresponding to the label int(N / 2). Also generate the data of the best achievable fidelity.

    Parameters
    ----------
    N : int
        Number of qubits, which include the inner neighbours and the target qubits.
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    distance : scalar
        Distance between the nodes of the network.
    num : int
        Number of points per axis to be plotted. Default is 50.
    save : True or False
        If True, the outcome is saved in a .npy file. Default is False.
    path : str
        Specification on where to save the .npy file. Default is ''.
    """
    if N % 2 != 0:
        raise ValueError("N must be even")
    if N < 4:
        raise ValueError("N must be larger than 3")
    # Define the range of the strength of the initial depolarizing noise.
    ps = np.linspace(0, 0.1, num=num)
    # Define the range of the dephasing time.
    dts = np.logspace(-3, -1, num=num)
    linear_cluster = gg.Graph(N=N, E=[(i, i + 1) for i in range(N - 1)])
    # Define the optimal patterns and corresponding strategies.
    pattern_depolarizing = ["."] + ["x"] * (N - 2) + ["."]
    sequence_depolarizing = pattern_to_sequence(pattern_depolarizing, linear_cluster)
    strategy_depolarizing = Strategy(
        sequence=sequence_depolarizing, graph=linear_cluster
    )
    if protocol == "central":
        if N == 4:
            pattern_dephasing = [".", "y", "y", "."]
        else:
            pattern_dephasing = (
                ["."]
                + ["x"] * (int(N / 2) - 2)
                + ["y"] * 3
                + ["x"] * (int(N / 2) - 3)
                + ["."]
            )
    elif protocol == "local":
        pattern_dephasing = ["."] + ["y"] * 2 + ["x"] * (N - 4) + ["."]
    else:
        raise ValueError("protocol must be central or local.")
    sequence_dephasing = pattern_to_sequence(pattern_dephasing, linear_cluster)
    strategy_dephasing = Strategy(sequence=sequence_dephasing, graph=linear_cluster)
    # Compute the difference between the fidelities resulting of the two strategies, such that if both fidelities are
    # below 0.5 (Bell pair is not entangled) there is no output.
    def difference(p, dt):
        # Define the noise coefficients of the initial depolarizing noise.
        coefficients = [1 - p + p / 4, p / 4, p / 4, p / 4]
        # Set the processing time.
        pt = 0.000001
        if protocol == "central":
            outcome_depolarizing = symmetric_1d_central(
                N,
                int(N / 2),
                distance,
                coefficients,
                strategy_depolarizing,
                dt,
                pt,
            )["fidelity"]
            outcome_dephasing = symmetric_1d_central(
                N,
                int(N / 2),
                distance,
                coefficients,
                strategy_dephasing,
                dt,
                pt,
            )["fidelity"]
        elif protocol == "local":
            outcome_depolarizing = symmetric_1d_local(
                N, distance, coefficients, strategy_depolarizing, dt, pt
            )["fidelity"]
            outcome_dephasing = symmetric_1d_local(
                N, distance, coefficients, strategy_dephasing, dt, pt
            )["fidelity"]
        else:
            raise ValueError("protocol must be central or local.")
        if (
            np.real_if_close(outcome_depolarizing) < 0.5
            and np.real_if_close(outcome_dephasing) < 0.5
        ):
            difference = None
        else:
            difference = np.real_if_close(outcome_depolarizing) - np.real_if_close(
                outcome_dephasing
            )
        return difference

    # Compute the best fidelity between the two strategies.
    def best_fidelity(p, dt):
        # Define the noise coefficients of the initial depolarizing noise.
        coefficients = [1 - p + p / 4, p / 4, p / 4, p / 4]
        # Set the processing time.
        pt = 0.000001
        if protocol == "central":
            outcome_depolarizing = symmetric_1d_central(
                N,
                int(N / 2),
                distance,
                coefficients,
                strategy_depolarizing,
                dt,
                pt,
            )["fidelity"]
            outcome_dephasing = symmetric_1d_central(
                N,
                int(N / 2),
                distance,
                coefficients,
                strategy_dephasing,
                dt,
                pt,
            )["fidelity"]
        elif protocol == "local":
            outcome_depolarizing = symmetric_1d_local(
                N, distance, coefficients, strategy_depolarizing, dt, pt
            )["fidelity"]
            outcome_dephasing = symmetric_1d_local(
                N, distance, coefficients, strategy_dephasing, dt, pt
            )["fidelity"]
        else:
            raise ValueError("protocol must be central or local.")
        if np.real_if_close(outcome_depolarizing) < np.real_if_close(outcome_dephasing):
            best_fidelity = np.real_if_close(outcome_dephasing)
        elif np.real_if_close(outcome_depolarizing) > np.real_if_close(
            outcome_dephasing
        ):
            best_fidelity = np.real_if_close(outcome_depolarizing)
        else:
            best_fidelity = np.real_if_close(outcome_depolarizing)
        return best_fidelity

    # Vectorize function that computes the difference.
    vectorized_difference = np.vectorize(difference, otypes=[float])
    # Vectorize function that computes the best achievable fidelity.
    vectorized_best_fidelity = np.vectorize(best_fidelity, otypes=[float])
    # Vectorize strength of the depolarizing noise and dephasing time.
    PS, DTS = np.meshgrid(ps, dts)
    # Compute the data.
    DIFF = vectorized_difference(PS, DTS)
    BEST = vectorized_best_fidelity(PS, DTS)
    # Save the data.
    if save is True:
        np.save(path + f"optimal_{protocol}_{N}", DIFF)
        np.save(path + f"best_fidelity_{protocol}_{N}", BEST)
    return


# Let us define some parameters for the plotting function.
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "cm"
plt.rcParams["font.size"] = 16


def plot_difference_optimal(protocol, num=50, save=False, path=""):
    """Generate a 2D plot of difference between fidelities from the two best measurement patterns for
    the manipulation of a symmetric N-qubit 1D cluster as a resource state using a central or local protocol.
    The difference is computed for different values of initial depolarizing noise and the dephasing time which
    determines the time-dependent dephasing noise the involved qubits are subject to. The processing times are set to
    10^{-6} s. For the central protocol, we consider that all nodes are placed in a line and the coordinator is the one
    corresponding to the label int(N / 2).

    Parameters
    ----------
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    num : int
        Number of points per axis to be plotted. Default is 50.
    save : True or False
        If True, the plot is saved in a .pdf file. If False, the plot is shown. Default is False.
    path : str
        Specification on where to save the plot. Default is ''.
    """
    differences = []
    NS = [6, 8, 10, 12]
    for N in NS:
        differences.append(np.load(path + f"optimal_{protocol}_{N}.npy"))
    # Define the range of the strength of the initial depolarizing noise.
    ps = np.linspace(0, 0.1, num=num)
    # Define the range of the dephasing time.
    dts = np.logspace(-3, -1, num=num)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    # For each protocol define the limits of the color-bar.
    if protocol == "central":
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=-0.026, vmax=0.026)
        ticks = [0.02, 0.01, 0, -0.01, -0.02]
    elif protocol == "local":
        norm = colors.TwoSlopeNorm(vcenter=0, vmin=-0.016, vmax=0.016)
        ticks = [0.01, 0, -0.01]
    else:
        raise ValueError("protocol must be central or local.")
    # Make each subplot
    for ax, DIFF, N in zip(axes.flat, differences, NS):
        # Plot some lines below to indicate the area where the fidelities are below 0.5.
        b = ax.fill_between(
            [0, 0.1],
            0.001,
            0.1,
            hatch=2 * "//",
            color="none",
            cmap=colors.ListedColormap(["none"]),
            edgecolor="#A0A0A0",
        )
        # Plot the data for fidelities above 0.5.
        c = ax.pcolormesh(ps, dts, DIFF, shading="auto", cmap="PuOr", norm=norm)
        ax.set_yscale("log")  # Dephasing time in log scale.
        ax.set_box_aspect(1)  # Squared ratio.
        # Set labels and titles.
        ax.set_ylabel(r"$T$ [s]", labelpad=3)
        ax.set_xlabel(r"$p$", labelpad=-3)
        ax.set_title(rf"$N={N}$", x=0.2, y=1.0, pad=-18)  # Put title inside plot.
        # Set limits and ticks.
        ax.set_xlim(left=0, right=0.1)
        ax.set_ylim(bottom=0.001, top=0.1)
        ax.tick_params(which="both", direction="in")
        ax.set_xticks([0.0, 0.05, 0.1])
    fig.subplots_adjust(right=0.8, wspace=0.35)
    # Set color bar for data for fidelities above 0.5.
    percentage = lambda x, _: f"${x * 100:g}$"  # Data in percentage.
    ax_cb1 = fig.add_axes([0.83, 0.3, 0.03, 0.58])  # Position and size of the bar.
    color_bar = fig.colorbar(
        c, cax=ax_cb1, orientation="vertical", format=percentage, ticks=ticks
    )
    color_bar.set_label(r"$\Delta F = F_{\mathcal{D}} - F_{\mathcal{Z}}$ [\%]")
    # Set color bar for data for fidelities below 0.5.
    ax_cb2 = fig.add_axes([0.83, 0.11, 0.03, 0.17])  # Position and size of the bar.
    color_bar_2 = fig.colorbar(b, cax=ax_cb2, orientation="vertical", ticks=[])
    color_bar_2.ax.add_patch(
        patches.Rectangle(
            (0, -0.04),
            1,
            10,
            hatch=2 * "//",
            edgecolor="#A0A0A0",
            lw=0,
            zorder=2,
            fill=False,
            snap=True,
        )
    )
    color_bar_2.set_label(r"$F_{\mathcal{D}}, F_{\mathcal{Z}} < 0.5$", labelpad=30)
    # Save or show the plot.
    if save is True:
        fig.savefig(path + f"Difference_{protocol}.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


def plot_best_fidelity(protocol, num=50, save=False, path=""):
    """Generate a 2D plot of the best achievable fidelity of a Bell pair achieved via the manipulation of a symmetric
    N-qubit 1D cluster using a central or local protocol.
    The fidelity is computed for different values of initial depolarizing noise and the dephasing time which
    determines the time-dependent dephasing noise the involved qubits are subject to. The processing times are set to
    10^{-6} s. For the central protocol, we consider that all nodes are placed in a line and the coordinator is the one
    corresponding to the label int(N / 2).

    Parameters
    ----------
    protocol : str
        Takes values "central" or "local". Determines if the protocol used is a local or central one.
    num : int
        Number of points per axis to be plotted. Default is 50.
    save : True or False
        If True, the plot is saved in a .pdf file. If False, the plot is shown. Default is False.
    path : str
        Specification on where to save the plot. Default is ''.
    """
    fidelities = []
    NS = [6, 8, 10, 12]
    for N in NS:
        fidelities.append(np.load(path + f"best_fidelity_{protocol}_{N}.npy"))
    # Define the range of the strength of the initial depolarizing noise.
    ps = np.linspace(0, 0.1, num=num)
    # Define the range of the dephasing time.
    dts = np.logspace(-3, -1, num=num)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    # Make each subplot
    norm = colors.TwoSlopeNorm(vcenter=0.75, vmin=0.5, vmax=1)
    for ax, FID, N in zip(axes.flat, fidelities, NS):
        c = ax.pcolormesh(ps, dts, FID, shading="auto", cmap="Purples", norm=norm)
        ax.set_yscale("log")  # Dephasing time in log scale.
        ax.set_box_aspect(1)  # Squared ratio.
        # Set labels and titles.
        ax.set_ylabel(r"$T$ [s]", labelpad=3)
        ax.set_xlabel(r"$p$", labelpad=-3)
        ax.set_title(rf"$N={N}$", x=0.83, y=1.0, pad=-18)  # Put title inside plot.
        # Set limits and ticks.
        ax.set_xlim(left=0, right=0.1)
        ax.set_ylim(bottom=0.001, top=0.1)
        ax.tick_params(which="both", direction="in")
        ax.set_xticks([0.0, 0.05, 0.1])
    fig.subplots_adjust(right=0.8, wspace=0.35)
    color_bar = fig.colorbar(c, ax=axes, orientation="vertical", fraction=0.04)
    color_bar.set_label(r"Best achievable fidelity")
    # Save or show the plot.
    if save is True:
        fig.savefig(path + f"Fidelities_{protocol}.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


if __name__ == "__main__":
    ### Here there is some code to investigate the optimal measurement pattern:
    processing_time = 0.000001  # Set the processing time.
    distance = 15000.0  # Set the internode distance.
    N = 6  # Set the size of the network.
    ## Investigate the optimal measurement patter, changing the dephasing times ###
    # Set depolarizing noise as the initial noise.
    p = 0.9
    coefficients = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
    for dt in np.logspace(-3, -1, num=10):
        print(
            "local",
            dt,
            optimal_pattern(N, "local", distance, coefficients, dt, processing_time)[
                "best_pattern"
            ],
        )
        print(
            "central",
            dt,
            optimal_pattern(N, "central", distance, coefficients, dt, processing_time)[
                "best_pattern"
            ],
        )
    ### Investigate the optimal measurement pattern, changing the depolarizing noise ###
    # Set the dephasing time.
    dephasing_time = 1
    # Range of the strength of the initial depolarizing noise.
    ps = np.linspace(0, 1, num=10)
    for p in ps:
        c = [p + (1 - p) / 4, (1 - p) / 4, (1 - p) / 4, (1 - p) / 4]
        print(
            "local",
            p,
            optimal_pattern(N, "local", distance, c, dephasing_time, processing_time)[
                "best_strategy"
            ],
        )
        print(
            "central",
            p,
            optimal_pattern(N, "central", distance, c, dephasing_time, processing_time)[
                "best_strategy"
            ],
        )
    ### Here we use the functions of this file to produce the data and the plots presented in the paper.
    distance = 15000.0
    num = 100
    for protocol in ["local", "central"]:
        for N in [6, 8, 10, 12]:
            generate_difference_optimal(N, protocol, distance, num=num, save=True)
        plot_difference_optimal(protocol, num=num)
        plot_best_fidelity(protocol, num=num)
