# Imperfect quantum networks with tailored resource states

This code corresponds to the scenarios and settings described in

> Imperfect quantum networks with tailored resource states <br>
> Maria Flors Mor-Ruiz, Julius Wallnöfer and Wolfgang Dür <br>
    > Preprint: [arxiv num [quant-ph]](arxiv link);


## Setting up the environment
Setting up a virtual environment to run the scenarios is recommended.
All the needed dependencies are listed in `requirements.txt`.
The following assumes `pip` and `virtualenv` are available on your system.
First, create a new virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate
```
Then, install the dependencies like this:
```
pip install -r requirements.txt
```

If you encounter any problems, you can try replicate the
environment that was used to develop the code. You will need `Python3.12`
and `pipenv` installed:
```
pipenv sync --dev
```
and then `pipenv shell` to activate the virtual environment.

## Structure of the code

There are three main folders where the code is stored.

### Tools

In `tools` there are general functions that are prepared to be installed as a Python package for the convenience of the user.
In particular, in the folder `libs`, you can find the file `aux_functions.py`, which includes a set of useful functions
used throughout the rest of the code. There is also file `strategies_1d.py`, which includes a collection of functions
that describe measurement patterns and the corresponding strategies used for the manipulation of a one-dimensional cluster into a Bell pair.

### Tests
In `tests` there are some testing functions for the translation between strategies and measurement patterns.

### Scenarios
The folder `scenarios` is separated in several folders that relate to a subsection of Section V of the paper.
Note that throughout the code we refer to the basic entanglement topology as the _no-shuffle case_ and to the custom entanglement topology as the _shuffle case_.

#### symmetric_1d
First, there is the folder `symmetric_1d` which includes the necessary functions to analyze and plot results of the scenario presented in Section V A of the paper.
In a few words, the scenario considers an N-qubit 1D cluster as a resource state and the target is a Bell pair between the two end nodes.
The qubits of the cluster are distributed in a network with a symmetric 1D chain geometry following the basic entanglement topology.
The nodes of the network are placed following a line in a 2D surface, so they all have the same y-coordinate and different x-coordinates.
The initial noise model is the same for all qubits in the cluster and the dephasing time is also the same for all the qubits.

#### asymmetric_1d
Next, in the folder `asymmetric_1d` there are the necessary functions to analyze and plot results of the scenario presented in Section V B of the paper.
In a few words, the scenario considers an N-qubit 1D cluster as a resource state and the target is a Bell pair between the two end nodes.
The qubits of the cluster are distributed in a network with a 1D chain geometry following the basic entanglement topology.
The functions in this folder allow to input the positions in a 2D plane, such that each node has x and y coordinates.
The strength and type of the initial noise model can be different for different qubits, as well as the dephasing times.

#### custom_1d
Lastly, in the folder `custom_1d` there are the functions to analyze and plot results of the scenario presented in Section V C of the paper.
In a few words, the scenario considers an N-qubit 1D cluster as a resource state and the target is a Bell pair between any two selected nodes.
The qubits of the cluster are distributed in a network with a 1D chain geometry which can follow both the basic and the custom entanglement topologies.
The functions in this folder allow to input the positions in a 2D plane, such that each node has x and y coordinates.
The strength and type of the initial noise model can be different for different qubits, as well as the dephasing times.

In this folder one can also find the folder `setting`, which includes the data corresponding to the random setting we use in Section V C.
This data is sufficient to produce the same results in Section V C.

## Usage of the code
Here a brief instruction on how to use the functions to replicate the results of the paper is presented.
The code presented here uses the `noisy_graph_states` an open-source Python package, which is an implementation of the results presented in the following paper:
> Noisy stabilizer formalism <br>
> Maria Flors Mor-Ruiz and Wolfgang Dür <br>
> Published version: [Phys. Rev. A 107, 032424](https://doi.org/10.1103/PhysRevA.107.032424).

This tool allows tracking of how noisy graph states transform under operations and measurements.
This is publicly available on GitHub under an MIT License:
> noisy_graph_states <br>
> Julius Wallnöfer <br>
> GitHub repository: [noisy_graph_states](https://doi.org/10.5281/zenodo.10625617)

Note that the functions in the `tools` folder are just imported as a package to use, so the user does not need to touch that.
The testing functions in `tests` are also not needed by the user.

The instructions are structured by folders inside the `scenarios` folder as they correspond to a specific section of the paper.
Nevertheless, each defined function has a detailed description of its task, inputs and outputs.

### symmetric_1d
File `symmetric_1d.py` has two functions `symmetric_1d_central` and `symmetric_1d_local`, which one can run as shown in the `if __name__ == "__main__"` part of the file.
These functions are a specific instance of the implementation of each of the protocols for the symmetric scenario, which compute the fidelity of the target state.
Notably, both functions take as inputs the coefficients of the initial noise model, which can be any single-qubit noise model, not only the depolarizing one.

File `pattern_symmetric_1d.py` has the function `optimal_pattern`, which performs an exhaustive search of all possible measurement patterns to find the ones that lead to the best and the worst fidelities.
This function uses the functions defined in `symmetric_1d.py` for the corresponding protocols, and computes the best and worst fidelities of the target state using all possible strategies.
Notably, the function takes as inputs the coefficients of the initial noise model, which can be any single-qubit noise model, not only the depolarizing one.
This file includes two more functions to generate the data and plot corresponding to the figures in Sec. V A and Appendix C.
In particular, one can run the `if __name__ == "__main__"` to generate the same figures.



### asymmetric_1d
File `asymmetric_1d.py` has two functions `asymmetric_1d_central` and `asymmetric_1d_local`, which one can run as shown in the `if __name__ == "__main__"` part of the file.
These functions are a specific instance of the implementation of each of the protocols for an asymmetric scenario, which compute the fidelity of the target state.
Notably, both functions take as inputs the coefficients of the initial noise model for each single qubit, which can be any single-qubit noise model, not only the depolarizing one.
Moreover, since these functions are for an asymmetric scenario, they take as inputs a list of dephasing times and processing times, which can be different for each qubit of the resource state.
Regarding the positioning of the nodes, in the asymmetric case, we give as input a list of positions in a 2D surface.

File `memory_asymmetry.py` includes the necessary functions to plot the results of considering a symmetric scenario with a faulty memory.
Meaning that the dephasing time of one of the qubits is different from the rest.
These use of the functions defined in `asymmetric_1d.py`.
This file includes a function to generate the data and plot corresponding to the figures in Sec. V B 1.
In particular, one can run the `if __name__ == "__main__"` to generate the same figures.

File `position_asymmetry.py` includes the necessary functions to plot the results of considering a symmetric scenario with one node with a shifted position.
Meaning that the internode distances of one of the qubits with the previous and next qubits are different from the rest.
These use of the functions defined in `asymmetric_1d.py`.
This file includes a function to generate the data and plot corresponding to the figures in Sec. V B 2.
In particular, one can run the `if __name__ == "__main__"` to generate the same figures.

File `pattern_asymmetric_1d.py` has the necessary functions to perform an exhaustive search of all possible measurement patterns to find the ones that lead to the best and the worst fidelities.
For both the scenario with memory asymmetries and the scenario with position asymmetries.

### custom_1d
File `random_setting.py` has the corresponding functions to establish a randomized setting, including positions, dephasing times and distribution of the resource state.
It also includes a plotting function for the randomized node positioning.
The randomized functions have the option to save the data, such that the same randomized scenario can be reused.
These functions can be run as shown in the `if __name__ == "__main__"` part of the file.

File `protocols.py` has two functions `local_protocol` and `central_protocol`, which one can run as shown in the `if __name__ == "__main__"` part of the file.
These functions are a specific instance of the implementation of each of the protocols for the asymmetric scenario, and they compute the fidelity of the target state.
Notably, both functions take as inputs the coefficients of the initial noise model for each single qubit, which can be any single-qubit noise model, not only the depolarizing one.
These functions are build such that one can later use them to compute the fidelity of a sub-cluster, as it includes the Z measurement of the outer neighbours of the target qubits.
These functions can be run as shown in the `if __name__ == "__main__"` part of the file.

File `shuffle.py` has a function `shuffle_average_outcome`, which one can run as shown in the `if __name__ == "__main__"` part of the file.
This function takes a network setting and uses of the functions defined in `protocol.py` to compute the fidelity of all the pairs achieved via the manipulation of the sub-clusters of the same size in the large cluster.
Moreover, it has the option to save the data, such that it allows an easy processing of the outcomes.

File `no_shuffle.py` has a function `no_shuffle_desired_average`, which one can run as shown in the `if __name__ == "__main__"` part of the file.
This function takes a network setting and uses of the functions defined in `protocol.py` to compute the fidelity of all the pairs specified in the parameter `shuffle`,
the manipulated sub-clusters might not have the same size.
Moreover, it has the option to save the data, such that it allows an easy processing of the outcomes.

File `plot_shuffle_vs_no_shuffle.py` has three functions. The first is `shuffle_vs_no_shuffle_desired`,
which computes the data of a specific scenario for the two networks cases (shuffle and no-shuffle) using the functions from
`shuffle.py` and `no_shuffle.py`.
Moreover, it has the option to save the data, such that it allows an easy processing of the outcomes.
The other two functions, create the figured presented in Sec. V C.
One can use the data available in the folder `setting` and the functions of this file to replicate the results of Section V C.
In particular, one can run the `if __name__ == "__main__"` to generate the same figures.
