#  list形式的架构转成邻接矩阵及one-hot节点类型矩阵
import utils
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#  ops_dim:type*qubit(qubit+1)/2
def list_to_adj(data, num_qubit, num_gate_type):
    res = []  # Store the results
    for i, list_arc in enumerate(data):
        temp_op = []  # Temporary list to store node encodings
        graph = nx.DiGraph()  # Create a directed graph

        graph.add_node('start', label='start')  # Add the start node
        for j in range(0, len(list_arc)):  # Add nodes for each gate
            graph.add_node(j, label=list_arc[j])
        graph.add_node('end', label='end')  # Add the end node

        last = ['start' for _ in range(num_qubit)]  # Track the last gate connected to each qubit

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

        for node in graph.nodes:  # Encode node types (node type and qubit information concatenated)
            if node == 'start':
                t1 = [0 for _ in range(num_gate_type + 2)]  # Start node encoding
                t2 = [0 for _ in range(int(num_qubit * (num_qubit + 1) / 2 + 1))]  # Qubit matrix encoding (start and end nodes are set to the last type: [0..., 1])
                t1[0] = 1  # Mark as start node
                t2[-1] = 1  # Mark as the last type
                t1.extend(t2)  # Concatenate qubit encoding
                temp_op.append(t1)
            elif node == 'end':
                t1 = [0 for _ in range(num_gate_type + 2)]  # End node encoding
                t2 = [0 for _ in range(int(num_qubit * (num_qubit + 1) / 2 + 1))]
                t1[-1] = 1  # Mark as end node
                t2[-1] = 1  # Mark as the last type
                t1.extend(t2)  # Concatenate qubit encoding
                temp_op.append(t1)
            else:  # Gate node encoding
                t1 = [0 for _ in range(num_gate_type + 2)]
                t2 = [0 for _ in range(int(num_qubit * (num_qubit + 1) / 2 + 1))]
                t1[int(graph.nodes[node]['label'][0]) + 1] = 1  # Mark the gate type (offset by 1 for start)
                if int(graph.nodes[node]['label'][1]) > int(graph.nodes[node]['label'][2]):
                    x = int(graph.nodes[node]['label'][2])
                    y = int(graph.nodes[node]['label'][1])
                else:
                    x = int(graph.nodes[node]['label'][1])
                    y = int(graph.nodes[node]['label'][2])
                t2[int(num_qubit * x + y - x / 2 * (x + 1))] = 1  # Mark the qubit pair
                t1.extend(t2)  # Concatenate qubit encoding
                temp_op.append(t1)

        # nx.draw_networkx(graph)
        # plt.show()
        temp_adj = nx.adjacency_matrix(graph).todense()  # Generate the adjacency matrix
        temp_adj = temp_adj.getA()
        temp_op = np.array(temp_op)  # Convert to a numpy array

        res.append([temp_adj, temp_op])  # Store the adjacency matrix and node encodings

        if i % 5000 == 0:
            print('Completed %d/%d' % (i, len(data)))  # Print progress every 5000 iterations

    return res  # Return the list of encoded circuits


# ops: type concatenated with qubit information
def list_to_adj_new(data, num_qubit, num_gate_type):
    res = []  # Store the results
    for i, list_arc in enumerate(data):
        list_arc = make_it_unique(list_arc, num_qubit)  # Ensure the circuit is unique
        temp_op = []  # Temporary list to store node encodings
        graph = nx.DiGraph()  # Create a directed graph

        graph.add_node('start', label='start')  # Add the start node
        for j in range(0, len(list_arc)):  # Add nodes for each gate
            graph.add_node(j, label=list_arc[j])
        graph.add_node('end', label='end')  # Add the end node

        last = ['start' for _ in range(num_qubit)]  # Track the last gate connected to each qubit

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

        for node in graph.nodes:  # Encode node types (node type and qubit information concatenated)
            if node == 'start':
                t1 = [0 for _ in range(num_gate_type + 2)]  # Start node encoding
                t2 = [1 for _ in range(num_qubit)]  # Qubit encoding
                t1[0] = 1  # Mark as start node
                t1.extend(t2)  # Concatenate qubit encoding
                temp_op.append(t1)
            elif node == 'end':
                t1 = [0 for _ in range(num_gate_type + 2)]  # End node encoding
                t2 = [1 for _ in range(num_qubit)]  # Qubit encoding
                t1[-1] = 1  # Mark as end node
                t1.extend(t2)  # Concatenate qubit encoding
                temp_op.append(t1)
            else:  # Gate node encoding
                t1 = [0 for _ in range(num_gate_type + 2)]
                t2 = [0 for _ in range(num_qubit)]
                t1[int(graph.nodes[node]['label'][0]) + 1] = 1  # Mark the gate type (offset by 1 for start)
                t2[int(graph.nodes[node]['label'][1])] = 1  # Mark the first qubit
                t2[int(graph.nodes[node]['label'][2])] = 1  # Mark the second qubit
                t1.extend(t2)  # Concatenate qubit encoding
                temp_op.append(t1)

        # nx.draw_networkx(graph)
        # plt.show()
        temp_adj = nx.adjacency_matrix(graph).todense()  # Generate the adjacency matrix
        temp_adj = temp_adj.getA()
        temp_op = np.array(temp_op)  # Convert to a numpy array

        res.append([temp_adj, temp_op])  # Store the adjacency matrix and node encodings

        if i % 5000 == 0:
            print('Completed %d/%d' % (i, len(data)))  # Print progress every 5000 iterations

    return res  # Return the list of encoded circuits


def make_it_unique(arc, num_qubit):
    lists = []  # List to store gates for each qubit
    final_list = []  # Final list to store the unique circuit

    for i in range(0, num_qubit):  # Initialize empty lists for each qubit
        lists.append([])

    for gate in arc:  # gate = [gate_type, qubit1, qubit2]
        if gate[2] != gate[1]:  # If it is a two-qubit gate
            if len(lists[gate[1]]) >= len(lists[gate[2]]):  # If qubit1 has more or equal gates than qubit2
                lists[gate[1]].append(gate)  # Add the gate to qubit1's list
                while len(lists[gate[1]]) > len(lists[gate[2]]):  # Pad qubit2's list with 0s
                    lists[gate[2]].append(0)
            else:  # If qubit1 has fewer gates than qubit2
                while len(lists[gate[1]]) < len(lists[gate[2]]):  # Pad qubit1's list with 0s
                    lists[gate[1]].append(0)
                lists[gate[1]].append(gate)  # Add the gate to qubit1's list
                lists[gate[2]].append(0)  # Pad qubit2's list with a 0
        else:  # If it is a single-qubit gate
            lists[gate[1]].append(gate)  # Add the gate to the corresponding qubit's list

    depth = []  # List to store the depth of each qubit's gate list
    for i in range(0, num_qubit):
        depth.append(len(lists[i]))  # Calculate the depth for each qubit
    max_depth = max(depth)  # Find the maximum depth among all qubits

    for i in range(max_depth):  # Iterate through each depth level
        for j in range(num_qubit):  # Iterate through each qubit
            if depth[j] - 1 < i:  # Skip if the qubit has no gate at this depth
                continue
            if lists[j][i] != 0:  # If the gate is not a padding 0
                final_list.append(lists[j][i])  # Add the gate to the final list

    return final_list  # Return the unique circuit


if __name__ == '__main__':
    list_arc = utils.load_pkl('datasets/state_classification/list_arcs_for_training.pkl')  # Load the list of circuits
    adj_arc = list_to_adj_new(list_arc, 8, 7)  # Generate adjacency matrix representations of the architectures
    utils.save_pkl(adj_arc, 'datasets/state_classification/arc_for_predictor.pkl')  # Save the results