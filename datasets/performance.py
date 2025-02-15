import numpy as np
import torch

import utils
from quantum_circuits.vqe_task_paralell import VqeTrainer, MaxCutTrainer
from settings import get_parser
from graph_for_maxcut.generate_graph import generate_graph

def get_performance(list):
    args = get_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #  Random Architecture Generator
    noise_param = None
    if args.noise == True:
        noise_param = {
            'two_qubit_depolarizing_p': args.two_qubit_depolarizing_p,
            'single_qubit_depolarizing_p': args.single_qubit_depolarizing_p,
            'bit_flip_p': args.bit_flip_p
        }

    if args.task == "vqe_36gate_h_head_new":
        args.gate_type = ['H', 'Rx', 'Ry', 'Rz', 'XX', 'YY', 'ZZ']

        trainer = VqeTrainer(n_cir_parallel=args.parallel, n_qubit=args.num_qubits, gate_number=args.max_gate_num,
                                n_depth=args.num_layers, n_runs=args.n_runs,
                                single_qubit_gate=args.num_of_single_qubit_gate,
                                h=args.Hamiltonian_vqe1, max_iteration=args.max_iteration, seed=args.seed,
                                noise_param=noise_param, task=args.task
                                )

        energy, params = trainer.process(list)
        energy = np.array(energy)
        utils.save_pkl(energy, f'datasets/{args.task}/label_for_embedding.pkl')

    elif args.task == "vqe_5_constraint":
        args.gate_type = ['Rx', 'Ry', 'Rz', 'XX', 'YY', 'ZZ']

        trainer = VqeTrainer(n_cir_parallel=args.parallel, n_qubit=args.num_qubits, gate_number=args.max_gate_num,
                                n_depth=args.num_layers, n_runs=args.n_runs,
                                single_qubit_gate=args.num_of_single_qubit_gate,
                                h=args.Hamiltonian_vqe5, max_iteration=args.max_iteration, seed=args.seed,
                                noise_param=noise_param, task=args.task
                                )

        energy, params = trainer.process(list)
        energy = np.array(energy)

    elif args.task == "Maxcut_12":
        args.gate_type = ['H', 'Ry', 'Rz', 'CNOT']

        graph_ = generate_graph(args.seed, args.num_qubits)

        trainer = MaxCutTrainer(gate_type=args.gate_type, num_qubits=args.num_qubits, num_layers=args.num_layers,
                                graph=graph_, n_cir_parallel=args.parallel, n_runs=args.n_runs,
                                max_iteration=args.max_iteration, seed=args.seed, theoretical=None,
                                noise_param=noise_param)
        theoretical = trainer.compute_theoretical()
        trainer.theoretical = theoretical
        energy, energy_epoch, pro_epoch, cut_edges = trainer.process(list)
        energy = np.array(energy)

    else:
        print('invalid task')
        exit(0)

    return energy
