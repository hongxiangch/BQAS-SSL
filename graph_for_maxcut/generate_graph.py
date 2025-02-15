import networkx as nx
import random


def generate_graph(random_seed, n_qubits):
    random.seed(random_seed)
    graph = nx.erdos_renyi_graph(n=n_qubits, p=0.75)

    return graph
