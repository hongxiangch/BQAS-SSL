import pickle
import numpy as np
import networkx as nx
import utils
from tqdm import tqdm
import pennylane as qml
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


class RandomArchitectureSearcherContraint:
    """
    Generation method with two-qubit gate qubit constraints: Selects the positions where two-qubit gates can act based on edge_index.
    """
    def __init__(self, gate_type, num_of_single_qubit_gate, max_gate_num, num_layers, num_qubits,
             no_parameter_gate, max_two_qubit_gates_rate, edge_index, start_with_h):
        self.mean = 0  # Mean value
        self.standard_deviation = 1.35  # Standard deviation
        self.gate_type = gate_type  # Types of gates
        self.num_of_single_qubit_gate = num_of_single_qubit_gate  # Number of single-qubit gate types
        self.p1 = 0.3  # Probability of being the same as the previous gate
        self.nt = max_gate_num  # Maximum number of quantum gates to generate
        self.D = num_layers  # Maximum number of layers
        self.N = num_qubits  # Number of qubits
        self.start_with_h = start_with_h
        if start_with_h:  # If starting with H gates, exclude the row occupied by H gates and the number of gates consumed by H gates
            self.nt = self.nt - self.N
            self.D = self.D - 1
        self.no_parameter_gate = no_parameter_gate  # Index of parameter-free gates
        self.edge_index = edge_index  # Positions where two-qubit gates can act
        self.max_two_qubit_gates_rate = max_two_qubit_gates_rate  # Ratio of two-qubit gates (excluding H gates at the beginning of the circuit, only considering the ratio in the generated gates)

    def generate_circuit(self, generation_type, start_with_h):  # Generate a single circuit
        # np.random.normal generates a random array following a normal distribution
        if self.start_with_h:
            normal = np.random.normal(self.mean, self.standard_deviation, len(self.gate_type) - 1)
        else:
            normal = np.random.normal(self.mean, self.standard_deviation, len(self.gate_type))
        log_it_list = normal  # Without preference
        log_it_list = np.exp(log_it_list) / sum(np.exp(log_it_list))  # Convert to a probability distribution (7 values)
        ciru = []  # Used to store the generated circuit

        # Place H gates in the first row. For all circuits, -1 fixedly represents the H gate, regardless of whether the H gate exists in gate_type
        if start_with_h:
            for i in range(0, self.N):
                ciru.append([0, i, i])

        if generation_type == 0:
            last = self.add_gate(0, log_it_list)  # Add the first gate
            ciru.append(last)
            list_qubit = np.zeros(self.N)  # Create a list to track whether the gate is the same as the previous one
            if last[1] == last[2]:
                list_qubit[last[1]] = last[0] + 1
            else:
                list_qubit[last[1]] = last[0] + 1
                list_qubit[last[2]] = -1
            for i in range(1, self.nt):  # Add gates iteratively, gate_wise approach to add nt-1 gates
                last = self.add_gate(last, log_it_list)
                # Regenerate if the gate is the same as the adjacent previous gate
                while list_qubit[last[1]] == last[0] + 1 and (
                    list_qubit[last[2]] == last[0] + 1 or list_qubit[last[2]] == -1):
                    last = self.add_gate(last, log_it_list)
                if last[1] == last[2]:
                    list_qubit[last[1]] = last[0] + 1
                else:
                    list_qubit[last[1]] = last[0] + 1
                    list_qubit[last[2]] = -1
                ciru.append(last)

        else:
            print('Invalid generation type, supposed to be gate_wise only')

        return ciru

    def add_gate(self, last_one, log_it_list):  # Add a gate
        # res = []  # The representation of the gate to be returned
        if last_one == 0:  # If it is the first gate
            if self.start_with_h:
                gate = np.random.choice(a=[1, 2, 3], size=1, p=log_it_list).item()  # Choose a gate type
            else:
                gate = np.random.choice(a=len(self.gate_type), size=1, p=log_it_list).item()
        else:  # If it is not the first gate
            if np.random.uniform() < self.p1:  # Determine whether the generated gate is the same as the previous one
                gate = last_one[0]
            else:  # Otherwise, generate a new gate based on the probability
                if self.start_with_h:
                    gate = np.random.choice(a=[1, 2, 3], size=1, p=log_it_list).item()  # Choose a gate type
                else:
                    gate = np.random.choice(a=len(self.gate_type), size=1, p=log_it_list).item()
        # After considering qubit connectivity, there is no longer the concept of adjacency; positions are chosen randomly
        if gate > self.num_of_single_qubit_gate - 1:  # If it is a two-qubit gate
            # Randomly select a qubit pair from the available combinations
            position = np.random.choice(a=len(self.edge_index), size=1).item()  # Excluding itself
            position = self.edge_index[position]
            res = [gate, position[0], position[1]]

        else:  # If it is a single-qubit gate
            # Directly select a qubit for the single-qubit gate
            position = np.random.choice(a=self.N, size=1).item()
            res = [gate, position, position]

        return res

    def check(self, cir):  # Check if the generated circuit meets the requirements

        res = [0] * self.N  # Record depth, a list of length N
        no_para = 0  # Number of parameter-free gates
        num_two_qubit_gates = 0  # Number of two-qubit gates
        keep = True  # Flag to determine if the circuit is valid

        for i in range(len(cir)):
            if cir[i][0] in self.no_parameter_gate:  # If the gate is parameter-free
                no_para += 1
            if cir[i][1] != cir[i][2]:  # If it is a two-qubit gate
                num_two_qubit_gates += 1
                if res[cir[i][1]] > res[cir[i][2]]:  # Update depth for qubit pairs
                    res[cir[i][1]] += 1
                    res[cir[i][2]] = res[cir[i][1]]
                else:
                    res[cir[i][2]] += 1
                    res[cir[i][1]] = res[cir[i][2]]
            else:  # If it is a single-qubit gate
                res[cir[i][1]] += 1  # Update depth for the qubit

        for i in res:
            if i > self.D:  # If the depth exceeds the limit, discard the circuit immediately
                keep = False
                # print('bad candidate circuit')
                break

        if no_para >= len(cir):  # If the circuit consists entirely of parameter-free gates, discard it
            keep = False

        if num_two_qubit_gates > int(len(cir) * self.max_two_qubit_gates_rate):  # If the number of two-qubit gates exceeds the limit, discard the circuit
            keep = False

        return keep  # Return whether the circuit is valid

    # num_architecture: Number of architectures to generate
    # generate_type: Generation method (0: gatewise, 1: layerwise)
    def get_architectures(self, num_architecture, generate_type, start_with_h):
        cirs = []  # List to store generated circuits
        num = 0  # Counter for the number of valid circuits generated
        pbar = tqdm(total=num_architecture,
                    desc='Randomly generating circuits')  # Create a progress bar to track circuit generation progress
        while num < num_architecture:

            # if not start_with_h:  # Determine if this batch of circuits starts with H gates
            temp = self.generate_circuit(generate_type, start_with_h)  # Generate a circuit with 35 gates
            # else:
            #     temp = self.generate_circuit_start_with_h(generation_type=generate_type)

            # 2023.04.19 Check if the number of gates in the circuit exceeds the allowed limit
            if start_with_h:
                if len(temp) > self.nt + self.N:
                    del temp[self.nt: len(temp)]  # Remove excess gates
            else:
                if len(temp) > self.nt:
                    del temp[self.nt: len(temp)]  # Remove excess gates

            # Check if the circuit exceeds the maximum depth, contains parameterized gates, and adheres to the two-qubit gate limit
            keep = self.check(temp)

            if keep:  # If the circuit is valid
                cirs.append(temp)  # Add it to the list
                num += 1  # Increment the counter
                pbar.update(1)  # Update the progress bar
        return cirs  # Return the list of valid circuits

    def list_to_adj(self, data):
        res = []  # Store the results
        for i, list_arc in tqdm(enumerate(data),
                                desc='list to adj'):  # Iterate through the list of circuits with a progress bar

            # self.draw_plot(list_arc)
            list_arc = self.make_it_unique(list_arc,
                                           self.N)  # Ensure the circuit is unique (list_arc has 35 gates, why?)
            # self.draw_plot(list_arc)
            temp_op = []  # Temporary list to store node encodings
            graph = nx.DiGraph()  # Create a directed graph (35 gates + 2 nodes for start and end)

            graph.add_node('start', label='start')  # Add the start node
            for j in range(0, len(list_arc)):  # Add nodes for each gate
                graph.add_node(j, label=list_arc[j])
            graph.add_node('end', label='end')  # Add the end node

            # Track the last gate operation on each qubit, initialized to 'start'
            last = ['start' for _ in range(self.N)]  # Last gate to be connected on each qubit

            for k in range(0, len(list_arc)):  # Add edges between gates
                if list_arc[k][1] == list_arc[k][2]:  # If it is a single-qubit gate
                    graph.add_edge(last[list_arc[k][1]], k)  # Connect to the last gate on the same qubit
                    last[list_arc[k][1]] = k  # Update the last gate for the qubit
                else:  # If it is a two-qubit gate
                    graph.add_edge(last[list_arc[k][1]], k)  # Connect to the last gate on the first qubit
                    graph.add_edge(last[list_arc[k][2]], k)  # Connect to the last gate on the second qubit
                    last[list_arc[k][1]] = k  # Update the last gate for the first qubit
                    last[list_arc[k][2]] = k  # Update the last gate for the second qubit

            for _ in last:  # Connect all last gates to the end node
                graph.add_edge(_, 'end')

            # Encoding
            for node in graph.nodes:  # Encode node types (node type and qubit information concatenated)
                if node == 'start':
                    t1 = [0 for _ in range(len(self.gate_type) + 2)]  # Start node encoding
                    t2 = [1 for _ in range(self.N)]  # Qubit encoding
                    t1[0] = 1  # Mark as start node
                    t1.extend(t2)  # Concatenate qubit encoding
                    temp_op.append(t1)
                elif node == 'end':
                    t1 = [0 for _ in range(len(self.gate_type) + 2)]  # End node encoding
                    t2 = [1 for _ in range(self.N)]  # Qubit encoding
                    t1[-1] = 1  # Mark as end node
                    t1.extend(t2)  # Concatenate qubit encoding
                    temp_op.append(t1)
                else:
                    t1 = [0 for _ in range(len(self.gate_type) + 2)]  # Gate node encoding
                    t2 = [0 for _ in range(self.N)]  # Qubit encoding
                    t1[int(graph.nodes[node]['label'][0]) + 1] = 1  # Mark the gate type (offset by 1 for start)
                    t2[int(graph.nodes[node]['label'][1])] = 1  # Mark the first qubit
                    t2[int(graph.nodes[node]['label'][2])] = 1  # Mark the second qubit
                    t1.extend(t2)  # Concatenate qubit encoding
                    temp_op.append(t1)

            # Generate the adjacency matrix from the graph using nx
            temp_adj = nx.adjacency_matrix(graph).todense()
            # temp_adj = temp_adj.getA()

            temp_op = np.array(temp_op)  # Convert to a numpy array

            # Represent the circuit as a directed acyclic graph (DAG)
            res.append([temp_adj, temp_op])

        return res  # Return the list of encoded circuits

    def make_it_unique(self, arc, num_qubit):
        lists = []  # List to store gates for each qubit
        final_list = []  # Final list to store the unique circuit

        for i in range(0, num_qubit):  # Number of qubits (e.g., 5)
            lists.append([])  # Initialize empty lists for each qubit: [[], [], ..., []]

        # Generate a unique circuit
        for gate in arc:  # gate = [gate_type, qubit1, qubit2]
            if gate[2] != gate[1]:  # If it is a two-qubit gate
                if len(lists[gate[1]]) >= len(lists[gate[2]]):  # If the number of gates on qubit1 is greater than or equal to qubit2
                    lists[gate[1]].append(gate)  # Add the gate to qubit1's list
                    while len(lists[gate[1]]) > len(lists[gate[2]]):  # Pad qubit2's list with 0s to match qubit1's length
                        lists[gate[2]].append(0)
                else:  # If the number of gates on qubit1 is less than qubit2
                    while len(lists[gate[1]]) < len(lists[gate[2]]):  # Pad qubit1's list with 0s to match qubit2's length
                        lists[gate[1]].append(0)
                    lists[gate[1]].append(gate)  # Add the gate to qubit1's list
                    lists[gate[2]].append(0)  # Pad qubit2's list with a 0
            else:  # If it is a single-qubit gate
                lists[gate[1]].append(gate)  # Add the gate to the corresponding qubit's list

        depth = []  # List to store the depth of each qubit's gate list
        for i in range(0, num_qubit):
            depth.append(len(lists[i]))  # Calculate the depth for each qubit
        max_depth = max(depth)  # Find the maximum depth among all qubits

        # Reconstruct the circuit in a unique order
        for i in range(max_depth):  # Iterate through each depth level
            for j in range(num_qubit):  # Iterate through each qubit
                if depth[j] - 1 < i:  # Skip if the qubit has no gate at this depth
                    continue
                if lists[j][i] != 0:  # If the gate is not a padding 0
                    final_list.append(lists[j][i])  # Add the gate to the final list

        return final_list  # Return the unique circuit


class RandomArchitectureSearcher:

    def __init__(self, mean, standard_deviation, gate_type, num_of_single_qubit_gate,
                 p1, p2, preference_num, max_gate_num, num_layers, num_qubits):
        self.mean = mean  # Mean
        self.standard_deviation = standard_deviation  # Standard deviation
        self.gate_type = gate_type  # Gate type
        self.num_of_single_qubit_gate = num_of_single_qubit_gate  # Number of single-qubit gates
        self.p1 = p1  # Probability of being the same as the previous gate
        self.p2 = p2  # Probability of being adjacent to the previous gate
        self.nt = max_gate_num  # Maximum number of quantum gates
        self.D = num_layers  # Maximum number of layers
        self.N = num_qubits  # Number of qubits
        self.preference_num = preference_num  # Preference value for single-qubit gates

    def generate_circuit(self, generation_type):  # Generate a single circuit
        normal = np.random.normal(self.mean, self.standard_deviation, len(self.gate_type))
        preference = [0 for _ in range(len(self.gate_type))]
        for i in range(0, self.num_of_single_qubit_gate):  # By default, single-qubit gates are placed before two-qubit gates, and preference is added to single-qubit gates
            preference[i] = self.preference_num
        log_it_list = normal + preference
        log_it_list = np.exp(log_it_list) / sum(np.exp(log_it_list))

        circuit = []

        if generation_type == 0:
            last = self.add_gate(0, log_it_list)
            circuit.append(last)

            for _ in range(1, self.nt):  # Add nt-1 gates in a gate-wise manner
                last = self.add_gate(last, log_it_list)
                circuit.append(last)

        elif generation_type == 1:  # Layer-wise (add half_layer each time, repeat self.D times)
            while len(circuit) < self.nt:
                t = self.add_layer(log_it_list)
                circuit.extend(t)

        else:
            print('Invalid generation type, expected to be 0 or 1, 0 for gate-wise pipeline and 1 for layer-wise pipeline')

        return circuit

    def generate_circuit_start_with_h(self,
                                      generation_type):  # Generate a single circuit (place H gates in the first row)
        normal = np.random.normal(self.mean, self.standard_deviation, len(self.gate_type))
        preference = [0 for _ in range(len(self.gate_type))]

        # Assign preferences to single-qubit gates (default is before two-qubit gates)
        for i in range(0, self.num_of_single_qubit_gate):
            preference[i] = self.preference_num

        log_it_list = normal + preference
        log_it_list = np.exp(log_it_list) / sum(np.exp(log_it_list))  # Softmax

        ciru = []  # Circuit

        # Place H gates in the first row
        for i in range(0, self.N):
            ciru.append([0, i, i])

        if generation_type == 0:
            last = self.add_gate(0, log_it_list)
            ciru.append(last)

            while len(ciru) < self.nt:  # Add nt gates using gate_wise method
                last = self.add_gate(last, log_it_list)
                ciru.append(last)

        elif generation_type == 1:  # layer_wise (add gates by adding half-layer each time)
            while len(ciru) < self.nt:
                t = self.add_layer(log_it_list)
                ciru.extend(t)

        else:
            print('Invalid generation type, supposed to be 0 or 1')

        return ciru

    def add_gate(self, last_one, log_it_list):  # Add a gate
        res = []  # Representation of the gate to be returned

        if last_one == 0:  # If it's the first gate
            gate = np.random.choice(a=len(self.gate_type), size=1, p=log_it_list).item()  # Choose a gate
            position = np.random.choice(a=self.N, size=1).item()  # Choose a position

            if gate > self.num_of_single_qubit_gate - 1:  # If it's a two-qubit gate
                # Since the quantum device used for classification tasks is a ring topology,
                # if it's the last qubit, connect it to the first one, i.e., [gate, 9, 0]
                if position == self.N - 1:
                    res = [gate, position, 0]
                else:
                    res = [gate, position, position + 1]

            else:  # If it's a single-qubit gate
                res = [gate, position, position]

        else:  # If it's not the first gate
            if np.random.uniform() < self.p1:  # Check if the generated gate is the same as the previous one
                gate = last_one[0]
            else:  # Otherwise, generate a new gate based on probability
                gate = np.random.choice(a=len(self.gate_type), size=1, p=log_it_list).item()  # Choose a gate

            if np.random.uniform() < self.p2:  # Check if the generated gate is adjacent to the previous gate
                if last_one[0] > self.num_of_single_qubit_gate - 1:
                    position = (last_one[1] + 2) % self.N
                else:
                    position = (last_one[1] + 1) % self.N
            else:  # Otherwise, randomly generate a position
                position = np.random.choice(a=self.N, size=1).item()  # Choose a position

            if gate > self.num_of_single_qubit_gate - 1:  # If it's a two-qubit gate
                # Since it's a ring topology, if it's the last qubit, connect it to qubit 0, i.e., [gate, 9, 0]
                if position == self.N - 1:
                    res = [gate, position, 0]
                else:
                    res = [gate, position, position + 1]

            else:  # If it's a single-qubit gate
                res = [gate, position, position]

        return res

    def add_layer(self,
                  log_it_list):  # Generate gates using layerwise method (N must be even, otherwise the generated circuit may not meet the requirements)
        res = []
        gate = np.random.choice(a=len(self.gate_type), size=1, p=log_it_list).item()
        position = np.random.choice(a=self.N, size=1).item()

        if gate > self.num_of_single_qubit_gate - 1:
            if position % 2 == 0:
                for i in range(0, int(self.N / 2)):
                    res.append([gate, i * 2, i * 2 + 1])
            else:
                res.append([gate, self.N - 1, 0])
                for i in range(1, int(self.N / 2)):
                    res.append([gate, i * 2 - 1, i * 2])
        else:
            if position % 2 == 0:
                for i in range(0, int(self.N / 2)):
                    res.append([gate, i * 2, i * 2])
            else:
                for i in range(0, int(self.N / 2)):
                    res.append([gate, i * 2 + 1, i * 2 + 1])

        return res

    def check(self, cir):  # Check if the depth of the generated circuit exceeds D
        res = [0] * self.N
        for i in range(len(cir)):
            if cir[i][0] > self.num_of_single_qubit_gate - 1:
                if res[cir[i][1]] > res[cir[i][2]]:
                    res[cir[i][1]] += 1
                    res[cir[i][2]] = res[cir[i][1]]
                else:
                    res[cir[i][2]] += 1
                    res[cir[i][1]] = res[cir[i][2]]
            else:
                res[cir[i][1]] += 1

        return res

    # num_architecture specifies the number of architectures to generate, generate_type specifies the generation method (0 for gatewise, 1 for layerwise)
    def get_architectures(self, num_architecture, generate_type, start_with_h):
        cirs = []
        num = 0
        pace = 0

        while num < num_architecture:
            flag = True
            if num % 5000 == 0:
                if not pace == num:
                    print('Progress of new circuit generation: %d/%d' % (num, num_architecture))
                    pace = num

            if not start_with_h:  # Check if this batch includes circuits starting with H gates
                temp = self.generate_circuit(generation_type=generate_type)
            else:
                temp = self.generate_circuit_start_with_h(generation_type=generate_type)
            che = self.check(temp)  # Check if the circuit exceeds the maximum depth

            for i in che:
                if i > self.D:
                    flag = False
                    # print('bad candidate circuit')
                    break

            if flag:
                cirs.append(temp)
                num += 1

        print('%d circuits generated successfully' % num_architecture)

        return cirs

    def list_to_adj(self, data, num_qubit, num_gate_type):
        res = []  # Store the results
        for i, list_arc in enumerate(data):
            list_arc = self.make_it_unique(list_arc, num_qubit)
            temp_op = []
            graph = nx.DiGraph()

            graph.add_node('start', label='start')  # Add a start node first
            for j in range(0, len(list_arc)):  # Add nodes one by one
                graph.add_node(j, label=list_arc[j])
            graph.add_node('end', label='end')  # Add an end node

            last = ['start' for _ in range(num_qubit)]  # The gate to be connected for each qubit

            for k in range(0, len(list_arc)):  # Add edges one by one
                if list_arc[k][1] == list_arc[k][2]:  # If it's a single-qubit gate
                    graph.add_edge(last[list_arc[k][1]], k)
                    last[list_arc[k][1]] = k
                else:  # If it's a two-qubit gate
                    graph.add_edge(last[list_arc[k][1]], k)
                    graph.add_edge(last[list_arc[k][2]], k)
                    last[list_arc[k][1]] = k
                    last[list_arc[k][2]] = k

            for _ in last:  # Connect all the last nodes to the end node
                graph.add_edge(_, 'end')

            # Encoding
            for node in graph.nodes:  # Mark node types (concatenate node types and qubit information)
                if node == 'start':
                    t1 = [0 for _ in range(num_gate_type + 2)]
                    t2 = [1 for _ in range(num_qubit)]
                    t1[0] = 1
                    t1.extend(t2)
                    temp_op.append(t1)
                elif node == 'end':
                    t1 = [0 for _ in range(num_gate_type + 2)]
                    t2 = [1 for _ in range(num_qubit)]
                    t1[-1] = 1
                    t1.extend(t2)
                    temp_op.append(t1)
                else:
                    t1 = [0 for _ in range(num_gate_type + 2)]
                    t2 = [0 for _ in range(num_qubit)]
                    t1[int(graph.nodes[node]['label'][0]) + 1] = 1
                    t2[int(graph.nodes[node]['label'][1])] = 1
                    t2[int(graph.nodes[node]['label'][2])] = 1
                    t1.extend(t2)
                    temp_op.append(t1)

            # Generate the adjacency matrix using the graph object and nx
            temp_adj = nx.adjacency_matrix(graph).todense()
            temp_adj = temp_adj.getA()

            temp_op = np.array(temp_op)

            # Directed acyclic graph (DAG) representation of the circuit
            res.append([temp_adj, temp_op])

            if i % 5000 == 0:
                print('Completed %d/%d' % (i, len(data)))

        return res

    def make_it_unique(self, arc, num_qubit):
        lists = []
        final_list = []

        for i in range(0, num_qubit):
            lists.append([])

        for gate in arc:  # gate = [gate_type, qubit1, qubit2]
            if gate[2] != gate[1]:  # Two-qubit gate
                if len(lists[gate[1]]) >= len(
                    lists[gate[2]]):  # If the number of gates on qubit 1 is greater than or equal to that on qubit 2
                    lists[gate[1]].append(gate)  # Add the gate to qubit 1
                    while len(lists[gate[1]]) > len(lists[gate[2]]):  # Fill qubit 2 with 0s
                        lists[gate[2]].append(0)
                else:  # If the number of gates on qubit 1 is less than that on qubit 2
                    while len(lists[gate[1]]) < len(lists[gate[2]]):
                        lists[gate[1]].append(0)  # Fill qubit 1 with 0s
                    lists[gate[1]].append(gate)  # Add the gate to qubit 1
                    lists[gate[2]].append(0)  # Fill qubit 2 with 0

            else:  # Single-qubit gate
                lists[gate[1]].append(gate)

        depth = []
        for i in range(0, num_qubit):
            depth.append(len(lists[i]))
        max_depth = max(depth)
        for i in range(max_depth):
            for j in range(num_qubit):
                if depth[j] - 1 < i:
                    continue
                if lists[j][i] != 0:
                    final_list.append(lists[j][i])

        return final_list