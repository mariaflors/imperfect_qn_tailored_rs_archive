"""A collection of functions to determine a random setting.
"""

import math
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt


def random_positions(N, save=False, path="scenarios/custom_1d/"):
    """This function gives a set of random positions in a 2D plane of a 1D cluster. The positions are such that the
    distance between two neighbouring nodes ranges from 5km to 25km.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    save : True or False
        If True, the outcome is saved in a .npy file. Default is False.
    path : str
        Specification on where to save the .npy file. Default is 'scenarios/custom_1d/'.

    Returns
    -------
    positions : list
        List of the positions of all the nodes. Each position should specify an x and a y coordinate in a numpy.ndarray.

    """
    # Create a list of np.arrays and add the first position in the (0,0).
    positions = [np.array([0.0, 0.0])]
    # Iterate to add the remaining N-1 positions
    for i in range(N - 1):
        # In polar coordinates, choose a radius between 5000 and 30000.
        r = random.uniform(5000.0, 25000.0)
        # In polar coordinates, choose a random angle and convert it to radians.
        theta = random.uniform(0, 360) * np.pi / 180
        # Transform these radius and angle into the x and y coordinates.
        x = positions[i][0] + r * math.cos(theta)
        y = positions[i][1] + r * math.sin(theta)
        # Add these x and y as a np.array into the list.
        positions.append(np.array([x, y]))
    # Choose the minimum values of the x and y coordinates.
    min_x, min_y = min(p[0] for p in positions), min(p[1] for p in positions)
    # Normalize positions so all values of x and y are positive.
    positions = [p - np.array([min_x, min_y]) for p in positions]
    if save is True:
        # Save the positions in a .npy file.
        np.save(path + f"positions_{N}", positions)
    # Return the list of positions
    return positions


def random_dephasing_times(N, save=False, path="scenarios/custom_1d/"):
    """This function gives a set of random dephasing times that range from 0.01s to 0.1s.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    save : True or False
        If True, the outcome is saved in a .npy file. Default is False.
    path : str
        Specification on where to save the .npy file. Default is 'scenarios/custom_1d/'.

    Returns
    -------
    dephasing_times : List of scalar
        Dephasing times of the memories in seconds. The length of this list is N.
    """
    # Create an empty list.
    dephasing_times = []
    # Iterate over all the possible qubits.
    for i in range(N):
        # Choose at random a dephasing time between 0.1 and 0.01.
        dt = random.uniform(0.01, 0.1)
        # Save the dephasing time into in the list.
        dephasing_times.append(dt)
    if save is True:
        # Save the data in a .npy file.
        np.save(path + f"dephasing_times_{N}", dephasing_times)
    # Return the list of dephasing times.
    return dephasing_times


def random_shuffle(N, save=False, path="scenarios/custom_1d/"):
    """This function produces a list that indicates the positions of the qubits in the physical structure for the random
    shuffling.

    Parameters
    ----------
    N : int
        Number of qubits in the initial 1D cluster.
    save : True or False
        If True, the outcome is saved in a pickle file.
    path : str
        Specification on where to save the data. Default is 'scenarios/custom_1d/'.

    Returns
    -------
    shuffle : list
        List which indicates the positions of the qubits in the physical structure for the optimized 1D cluster.
        Such that 'shuffle[i]' corresponds to the label of the qubit which is in position 'i'.
    """
    # Create a list with a random permutation of the numbers ranging from 0 to N-1.
    shuffle = random.sample([i for i in range(N)], N)
    if save is True:
        # Save the data in a pickle file.
        with open(path + f"shuffle_{N}", "wb") as fp:
            pickle.dump(shuffle, fp)
    # Return the list of shuffled qubits.
    return shuffle


def plot_positions(positions, save=False, path=""):
    """This function plots the distribution of node of the networks.

    Parameters
    ---------
    positions : list
        List of the positions of all the nodes. Each position should specify an x and a y coordinate in a numpy.ndarray.
    save : True or False
        If True, the plot is saved in a .pdf file. If False, the plot is shown. Default is False.
    path : str
        Specification on where to save the plot. Default is ''.

    """
    # Define the use of LaTeX font and size 12.
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12

    # Create two lists with the x and y coordinates.
    x = [p[0] / 1000 for p in positions]
    y = [p[1] / 1000 for p in positions]
    fig, ax = plt.subplots()
    ax.axis("equal")
    # Plot the x and y coordinates with dots and a faded line connecting the consequent points.
    ax.plot(x, y, color="#8EA604", marker="s", markersize=5.8, linestyle="None")
    ax.plot(x, y, color="#8EA604", alpha=0.3, linewidth=2.7)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.tick_params(axis="both", which="major", length=4)
    plt.tick_params(axis="x", which="both", pad=6)

    plt.xlabel(r"Position in the $x$ axis [km]")  # Label the x-axis.
    plt.ylabel(r"Position in the $y$ axis [km]")  # Label the y-axis.
    # Save or show the plot.
    if save is True:
        plt.savefig(path + f"Positions.pdf", bbox_inches="tight")
    else:
        plt.show()
    return


if __name__ == "__main__":
    # Here we make an example on how to use the functions of this file.
    path = "scenarios/custom_1d/setting/"
    ### Generate the setting ###
    # # Run this to generate a random setting for N=100
    # save = False
    # random_positions(100, save=save, path=path)
    # random_dephasing_times(100, save=save, path=path)
    # random_shuffle(100, save=save, path=path)

    ### Plot the positions ###
    positions = np.load(path + "positions_100.npy")
    plot_positions(positions, save=False, path=path)
