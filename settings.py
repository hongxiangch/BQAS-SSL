import argparse



def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='vqe_5_constraint', help='vqe_5_constraint, vqe_36gate_h_head_new')  # Task type
    # Parameters for generating quantum circuits
    parser.add_argument('--generate_type', type=int, default=0, help='1 for layerwise, 0 for gatewise')  # Generation type, vqe_36gate_h_head_new:1
    parser.add_argument('--num_qubits', type=int, default=2)  # Number of qubits, vqe_36gate_h_head_new:6
    parser.add_argument('--num_layers', type=int, default=20)  # Number of layers, vqe_36gate_h_head_new:10
    parser.add_argument('--max_gate_num', type=int, default=38)  # Maximum number of gates, vqe_36gate_h_head_new:36
    parser.add_argument('--gate_type', type=list, default=['Rx', 'Ry', 'Rz', 'XX', 'YY', 'ZZ'])
    parser.add_argument('--num_of_single_qubit_gate', type=int, default=3)
    parser.add_argument('--no_parameter_gate', type=list, default=[])
    parser.add_argument('--max_two_qubit_gates_rate', type=float, default=0.35)
    parser.add_argument('--edge_index', type=list, default=[[0, 1], [1, 2], [1, 3], [3, 4]])
    parser.add_argument('--start_with_h', type=bool, default=False)
    parser.add_argument('--preference_num', type=float, default=0.5)  # Preference value for single-qubit gates
    parser.add_argument('--mean', type=float, default=0)
    parser.add_argument('--standard_deviation', type=float, default=1.35)
    parser.add_argument('--p1', type=float, default=0.7)
    parser.add_argument('--p2', type=float, default=0.65)

    # Parameters for calculating expressiveness
    parser.add_argument("--proxy", type=str, default='EX', help="Proxy file")
    parser.add_argument('--search_space', type=str, default='block', help='Search space')
    parser.add_argument("--num_random_initial", type=int, default=2000, help="Number of random initial for fidelities calculation")
    parser.add_argument("--haarPoints", type=int, default=100, help="KL_haar points")
    parser.add_argument("--gaussian_kernel_sigma", type=float, default=0.01, help="MMD Gaussian kernel sigma")

    # Parameters of noise
    parser.add_argument("--noise", type=int, default=False, help="Noise")
    parser.add_argument("--two_qubit_depolarizing_p", type=float, default=0.01,
                        help="Two-qubit depolarizing noise level")
    parser.add_argument("--single_qubit_depolarizing_p", type=float, default=0.001,
                        help="Single-qubit depolarizing noise level")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="Bit-flip noise")

    # Energy calculation parameters
    parser.add_argument("--max_iteration", type=int, default=5000, help="Max iterations")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel processing")
    parser.add_argument("--Hamiltonian_vqe1", type=dict, default={'pbc': True, 'hzz': 1, 'hxx': 0, 'hyy': 0, 'hx': 1, 'hy': 0, 'hz': 0, 'sparse': False}, help="TFIM Hamiltonian")
    parser.add_argument('--Hamiltonian_vqe5', type=dict, default={'pbc': True, 'hzz': 1, 'hxx': 1, 'hyy': 1, 'hx': 0, 'hy': 0, 'hz': 1, 'sparse': False})
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs")

    # Pretraining parameters
    parser.add_argument('--seed2', type=int, default=0, help='random seed for data')
    parser.add_argument('--pretraining_num', type=int, default=5000)
    parser.add_argument('--cuda_for_embedding', type=bool, default=False, help='use gpu')
    parser.add_argument('--pretraining', type=bool, default=False, help='Whether to generate a new embedding')
    parser.add_argument('--training_vae', type=bool, default=False, help='Whether to train a new VAE model')
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument('--cfg', type=int, default=4, help='Configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=32, help='Batch size (default: vqe-32, classification-256)')
    parser.add_argument('--epochs', type=int, default=1, help='Training epochs (default: vqe-1, classification-1)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Decoder implicit regularization (default: 0.3)')
    parser.add_argument('--hidden_dim', type=int, default=128)  # 128
    parser.add_argument('--dim', type=int, default=128, help='Feature dimension (default: vqe-15, classification-17)')
    parser.add_argument('--gmp_dim', type=int, default=96)
    parser.add_argument('--hops', type=int, default=5)

    # Parameters for training the predictor
    parser.add_argument('--print_info', type=bool, default=False, help='Whether to print detailed information')
    parser.add_argument('--altering_parameter', type=bool, default=False, help='Parameter tuning')
    parser.add_argument('--predictor', type=bool, default=False, help='Whether to train and test a predictor')
    # parser.add_argument('--E0', type=float, default=-7.7274067, help='Ground state energy')
    parser.add_argument('--E0', type=float, default=-7.7274067, help='Ground state energy')
    parser.add_argument('--num_test_set', type=int, default=50000, help='Number of testing data')
    parser.add_argument('--original_label', type=bool, default=False, help='Whether to use original label')
    parser.add_argument('--num_training_sample', type=int, default=200, help='Number of training samples')
    parser.add_argument('--epochs_for_predictor', type=int, default=100, help='Epochs when training predictor')
    parser.add_argument('--lr_for_predictor', type=float, default=0.001, help='Learning rate for predictor')
    parser.add_argument('--train_bs_for_predictor', type=int, default=16, help='Train batch size for predictor')
    parser.add_argument('--val_bs_for_predictor', type=int, default=100, help='Validation batch size for predictor')
    parser.add_argument('--test_bs_for_predictor', type=int, default=150, help='Test batch size for predictor')

    # Downstream task parameters
    parser.add_argument('--epoch_stage', type=int, default=300, help='Epochs for stage')
    parser.add_argument('--lr_for_stage', type=float, default=0.0001, help='Learning rate for stage')

    # Data smoothing parameters
    parser.add_argument('--dealing_imbalance', type=bool, default=False, help='Whether to use a smoothed version of the dataset')
    parser.add_argument('--lds', type=bool, default=False, help='Whether to use LDS: vqe-True, classification-False')
    parser.add_argument('--bins', type=int, default=500, help='Number of bins for labels')  # 500 for vqe, 10 for state classification
    parser.add_argument('--kernel', type=str, default='triang', help='Kernel function, options: gaussian, triang, laplace')
    parser.add_argument('--fds', type=bool, default=False, help='Whether to use FDS')
    parser.add_argument('--bucket_num', type=int, default=280, help='Number of buckets for FDS labels')
    parser.add_argument('--start_update', type=int, default=0, help='Which epoch to start FDS updating')
    parser.add_argument('--start_smooth', type=int, default=1, help='Which epoch to start using FDS for feature smoothing')
    parser.add_argument('--special_weight', type=float, default=1.0, help='Special weight for the best architecture in the training set')

    # self_supervise fine-tuning
    parser.add_argument('--self_supervise', type=bool, default=False, help='whether to use pf method')
    parser.add_argument('--pure_supervise', type=bool, default=False, help='randomly set the initial param of encoder')
    parser.add_argument('--fine_tune', type=bool, default=False, help='fine-tuning')
    parser.add_argument('--fine_tune_stage', type=int, default=100, help='only do one stage')

    # BQAS parameters
    parser.add_argument('--BANAQAS', type=bool, default=False, help='whether to use weakQAS to search architecture')
    parser.add_argument('--N', type=int, default=200, help='n')
    parser.add_argument('--K', type=int, default=20, help='top k')
    parser.add_argument('--num_init_sample', type=int, default=20, help='number of initial training data')
    parser.add_argument('--num_ensemble', type=int, default=3, help='number of iterations')
    parser.add_argument('--bs_for_train_bo', type=int, default=16, help='batch size for training')
    parser.add_argument('--test_bo_bs_for_predictor', type=int, default=150, help='batch size for training')
    parser.add_argument('--epochs_for_init', type=int, default=100, help='epochs per iteration')
    parser.add_argument('--lr_for_bo', type=float, default=0.0001, help='learning rate for predictor')

    args = parser.parse_args()
    return args

