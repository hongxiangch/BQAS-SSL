import numpy as np
import torch
import utils
from quantum_circuits.search_architecture import RandomArchitectureSearcher, RandomArchitectureSearcherContraint
from settings import get_parser


def get_datasets():
    args = get_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    list_arcs_for_testing = []
    arc_for_embedding = []
    if args.task == "vqe_36gate_h_head_new":
        args.gate_type = ['H', 'Rx', 'Ry', 'Rz', 'XX', 'YY', 'ZZ']
        random_searcher = RandomArchitectureSearcher(args.mean, args.standard_deviation, args.gate_type,
                                                     args.num_of_single_qubit_gate, args.p1, args.p2,
                                                     args.preference_num, args.max_gate_num,
                                                     args.num_layers, args.num_qubits)

        list_arcs_for_testing = random_searcher.get_architectures(args.num_test_set, args.generate_type, args.start_with_h)
        utils.save_pkl(list_arcs_for_testing, f'datasets/{args.task}/list_arcs_for_testing.pkl')
        arc_for_embedding = random_searcher.list_to_adj(list_arcs_for_testing, args.num_qubits, len(args.gate_type))
        utils.save_pkl(arc_for_embedding, f'datasets/{args.task}/arc_for_embedding.pkl')

    elif args.task == "vqe_5_constraint":
        args.gate_type = ['Rx', 'Ry', 'Rz', 'XX', 'YY', 'ZZ']
        random_searcher = RandomArchitectureSearcherContraint(args.gate_type, args.num_of_single_qubit_gate,
                                                              args.max_gate_num,
                                                              args.num_layers, args.num_qubits, args.no_parameter_gate,
                                                              args.max_two_qubit_gates_rate, args.edge_index,
                                                              args.start_with_h)

        list_arcs_for_testing = random_searcher.get_architectures(args.num_test_set, args.generate_type, args.start_with_h)
        utils.save_pkl(list_arcs_for_testing, f'datasets/{args.task}/list_arcs_for_testing.pkl')
        arc_for_embedding = random_searcher.list_to_adj(list_arcs_for_testing)
        utils.save_pkl(arc_for_embedding, f'datasets/{args.task}/arc_for_embedding.pkl')

    elif args.task == "Maxcut_12":
        args.gate_type = ['H', 'Ry', 'Rz', 'CNOT']
        args.edge_index = [[0, 1], [1, 2], [1, 4], [2, 3], [3, 5], [4, 7], [5, 8], [6, 7], [8, 9], [7, 10], [8, 11]]
        random_searcher = RandomArchitectureSearcherContraint(args.gate_type, args.num_of_single_qubit_gate,
                                                              args.max_gate_num,
                                                              args.num_layers, args.num_qubits, args.no_parameter_gate,
                                                              args.max_two_qubit_gates_rate, args.edge_index,
                                                              args.start_with_h)

        num_tasks = 5

        for i in range(num_tasks):
            args.seed = i
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            list_arcs_for_testing = random_searcher.get_architectures(args.num_test_set, args.generate_type, args.start_with_h)
            utils.save_pkl(list_arcs_for_testing, f'datasets/{args.task}/list_arcs_for_testing_seed{args.seed}.pkl')
            arc_for_embedding = random_searcher.list_to_adj(list_arcs_for_testing)
            utils.save_pkl(arc_for_embedding, f'datasets/{args.task}/arc_for_embedding_seed{args.seed}.pkl')

    else:
        print('invalid task')
        exit(0)

    return list_arcs_for_testing, arc_for_embedding

list_arcs_for_testing, arc_for_embedding = get_datasets()
print("Dataset preparation completed.")

