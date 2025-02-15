import os
import sys

from arc2vec.BANAQAS import BANAQAS
from datasets.performance import get_performance

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'  # !(important)It makes rnn_based model deterministic
sys.path.append('arc2vec')
from settings import get_parser
import utils
from quantum_circuits.search_architecture import RandomArchitectureSearcher, RandomArchitectureSearcherContraint
from quantum_circuits.vqe_task_paralell import VqeTrainer
from quantum_circuits.Expressbility import ExpressibilityCalculator, ExpressibilityCalculatorlayer
from arc2vec.pretraining import Pretraining
import numpy as np
import torch
from models.supervised_model import SSL_Stage
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
import lds.smooth_label_space as smooth_func
from lds.loss import weighted_mse_loss


def supervised_test(model, y_train, test_arcs, test_lists, args, result_path):
    result_path = result_path + f'/supervised_result-hops5/pretraining_{args.pretraining_num}/train-{args.num_training_sample}/stage-{args.epoch_stage}/fine_tune-{args.fine_tune_stage}/model-{args.seed2}/seed-{args.seed}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if args.task == 'vqe_36gate_h_head_new':
        min_energy = -7.7
        param = 14
    elif args.task == 'vqe_5_constraint':
        min_energy = -8.47
        param = 20
    elif args.task == 'Maxcut_12':
        if args.seed2 < 3:
            min_energy = -13.99
            param = 14
        elif args.seed2 == 3:
            min_energy = -17.99
            param = 14
        elif args.seed2 == 4:
            min_energy = -16.99
            param = 14
    else:
        print('Invalid task.')
        exit(0)

    test_arcs_ops = np.array([test_arcs[i][1] for i in range(0, len(test_arcs))])
    test_arcs_adj = np.array([test_arcs[i][0] for i in range(0, len(test_arcs))])

    X_test_adj = torch.Tensor(test_arcs_adj)
    X_test_ops = torch.Tensor(test_arcs_ops)
    test_data = TensorDataset(X_test_adj, X_test_ops)
    test_loader = DataLoader(test_data, batch_size=args.test_bs_for_predictor, shuffle=False, drop_last=False)

    print('Energy predicting...')
    pred = []
    model.eval()
    for step, (b_x_adj, b_x_ops) in enumerate(test_loader):
        with torch.no_grad():
            p = model(b_x_adj, b_x_ops)[0].detach().cpu().tolist()
            pred.extend(p)
    print('give up stage1...')
    pred = np.array(pred)

    #  ranking
    print('ranking...')
    rank = np.argsort(pred)
    picked_index = rank
    print('ranking complete')
    picked_best = []
    count_best = 0
    first_rank = len(rank) - 1
    lowest = 0
    runs_history, rank_history, best_history = [], [], []
    temp_history = []
    for i in range(0, len(picked_index)):
        label = get_performance([test_lists[rank[i]]])
        picked_best.append(label)
        if label < min_energy:
            if i < first_rank:
                first_rank = i
            count_best += 1
            print('rank:%d pred:%f label:%f' % (i, pred[rank[i]] * param + args.E0, label[rank[i]]))
            temp_history.append([i, pred[rank[i]] * param + args.E0, label])
            lowest = label
            break

        if label < lowest:
            lowest = label

    print('found best:%d first rank:%d lowest energy:%f' % (count_best, first_rank, lowest))
    runs_history.append(temp_history)
    rank_history.append([count_best, first_rank, lowest])
    best_history.append(picked_best)
    utils.save_pkl(y_train, result_path + '/train_label.pkl')
    utils.save_pkl(first_rank, result_path + '/queries.pkl')

    print('------------------------------')

    return 0

def BANAQAS_search(list_cir, matrix_cir, args, result_path):
    result_path = result_path + f'/BANAQAS_result/BPR/train-{args.num_init_sample}/epoch-{args.epochs_for_init}/model-{args.seed2}/seed-{args.seed}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    visited = np.array([False for _ in range(0, len(matrix_cir))])

    visited_index_history = []

    # All test set
    energy_result = np.array([None for i in range(0, len(matrix_cir))])
    adj = np.array([matrix_cir[i][0] for i in range(len(matrix_cir))])
    ops = np.array([matrix_cir[i][1] for i in range(len(matrix_cir))])

    # Create a randomly shuffled index array
    indices = np.random.permutation(range(args.num_test_set))

    # Randomly select num_init_sample samples from the test set for training the proxy model
    initial_train_index = indices[:args.num_init_sample]

    # Get the true performance of the training samples
    for i in range(0, len(initial_train_index)):
        energy_result[initial_train_index[i]] = get_performance([list_cir[initial_train_index[i]]])

    visited[initial_train_index] = True  # Mark the selected circuits as visited
    visited_index_history.extend(initial_train_index)

    encoder_loc = f'pretrained/{args.task}/pretraining_{args.pretraining_num}/seed-{args.seed2}/dim-{args.dim}/model.pt'

    # Before starting the iteration, initialize the model and perform initial training
    model = SSL_Stage(args)

    if not args.pure_supervise:
        model.encoder.load_state_dict(torch.load(encoder_loc)['encoder_state'])

    # Save the model state dictionary
    torch.save(model.state_dict(), 'models/model_for_BANAQAS/init_model.pt')

    print(f'Initial, train_data_len: {len(adj[visited])}')

    BO = BANAQAS(args)
    queries = args.num_test_set
    # Initial training process
    num = 0
    picked_best = []
    while len(visited[visited == True]) <= args.N:
        print(len(visited[visited == True]))
        # Temporarily use other data from the test set as training samples, generative part to be updated
        # Prepare the initial training data

        train_loader = BO.prepare_data(adj, ops, energy_result, visited)
        candidate_predictions = []
        for i in range(0, args.num_ensemble):
            # Test set preparation
            predictor = SSL_Stage(args)
            predictor.encoder.load_state_dict(torch.load(encoder_loc)['encoder_state'])
            predictor.load_state_dict(torch.load('models/model_for_BANAQAS/init_model.pt'))
            encoder, predictor = BO.train(predictor, train_loader, args)

            x_test_adj = torch.Tensor(adj[~visited])  # Select unvisited data for prediction
            x_test_ops = torch.Tensor(ops[~visited])  # Select unvisited data for prediction

            test_data = TensorDataset(x_test_adj, x_test_ops)
            test_loader = DataLoader(test_data, batch_size=args.test_bo_bs_for_predictor, shuffle=False,
                                     drop_last=False)

            candidate_predictions.append(BO.predict(predictor, test_loader))

        test_index_in_space = np.argwhere(visited == False).reshape(-1)  # This line retrieves the unvisited data indices in the original dataset
        candidate_indices = BO.acq_fn(predictions=candidate_predictions, explore_type='its')

        n_index_in_space = test_index_in_space[candidate_indices[:args.K]]  # The indices corresponding to the remaining unselected circuits

        energy_path = f'result/{args.task}/BANAQAS_result/BPR/energy/train-{args.num_init_sample}/epoch-{args.epochs_for_init}/seed2-{args.seed2}/'
        if not os.path.exists(energy_path):
            os.makedirs(energy_path)

        if len(visited[visited == True]) == args.N:
            print("The number of training iterations has reached the upper limit.")
            exit(0)

        visited_index_history.extend(n_index_in_space)  # Record the visited circuits
        visited[visited_index_history] = True

        if args.task == 'vqe_36gate_h_head_new':
            args.E0 = -7.7274067
            min_energy = -7.7
        elif args.task == 'vqe_5_constraint':
            args.E0 = -8.472136
            min_energy = -8.47
        elif args.task == 'Maxcut_12':
            if args.seed2 < 3:
                args.E0 = -14
                min_energy = -13.99
            elif args.seed2 == 3:
                args.E0 = -18
                min_energy = -17.99
            elif args.seed2 == 4:
                args.E0 = -17
                min_energy = -16.99
        else:
            print('Invalid task.')
            exit(0)

        # Get the true performance of the selected data
        for r in range(0, len(n_index_in_space)):
            energy_result[n_index_in_space[r]] = get_performance([list_cir[n_index_in_space[r]]])
            if energy_result[n_index_in_space[r]][0] < min_energy:
                queries = num * args.K + r
                print('already found the best\n')
                print(
                    f'iteration{num} have found the best, the energy: {energy_result[n_index_in_space[r]]}')
                print(f'rank: {queries}')
                break
        picked_best.extend(energy_result[n_index_in_space])

        if energy_result[n_index_in_space[r]] < min_energy:
            utils.save_pkl(picked_best, energy_path + f'energy_{args.seed}.pkl')
            break

        print(f'interation{num} found lowest energy: {np.min(energy_result[n_index_in_space])}')

        num += 1

    utils.save_pkl(queries, result_path + '/queries.pkl')
    utils.save_pkl(visited_index_history, result_path + '/visited_history.pkl')
    utils.save_pkl(energy_result, result_path + '/label.pkl')
    return 0

def self_supervise_training(arcs, original_label, args, result_path):
    print(f'seed{args.seed}-stage_epoch:{args.epoch_stage}-fine_tune_stage:{args.fine_tune_stage}')
    original_label = np.array(original_label)
    adj = np.array([arcs[i][0] for i in range(len(arcs))])
    ops = np.array([arcs[i][1] for i in range(len(arcs))])

    encoder_loc = f'pretrained/{args.task}/pretraining_{args.pretraining_num}/seed-{args.seed2}/dim-{args.dim}/model.pt'

    # stage2
    if args.original_label:
        stage_label = original_label
    else:
        if args.task == 'vqe_36gate_h_head_new':
            args.E0 = -7.7274067
            stage_label = (original_label - args.E0) / 14
        elif args.task == 'vqe_5_constraint':
            args.E0 = -8.472136
            stage_label = (original_label - args.E0) / 20
        elif args.task == 'Maxcut_12':
            if args.seed2 < 3:
                args.E0 = -14
                stage_label = (original_label - args.E0) / 28
            elif args.seed2 == 3:
                args.E0 = -18
                stage_label = (original_label - args.E0) / 36
            elif args.seed2 == 4:
                args.E0 = -17
                stage_label = (original_label - args.E0) / 34
        else:
            print('Invalid task.')

    X_adj_train = torch.Tensor(adj)
    X_ops_train = torch.Tensor(ops)
    y_train = torch.Tensor(stage_label)

    if args.lds:
        eff_label_dist, bin_index_per_label = smooth_func.smooth(y_train, args)
        eff_label_dist = eff_label_dist.astype(float)
        eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
        weights = [np.float32(1.0 / x) for x in eff_num_per_label]
        weights = torch.Tensor(weights)
        train_data_stage = TensorDataset(X_adj_train, X_ops_train, y_train, weights)
    else:
        train_data_stage = TensorDataset(X_adj_train, X_ops_train, y_train)

    train_loader_stage = DataLoader(train_data_stage, batch_size=args.train_bs_for_predictor, shuffle=True,
                                     drop_last=False)

    model = SSL_Stage(args)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'stage trainable parameter:{trainable_num}')
    if not args.pure_supervise:
        model.encoder.load_state_dict(torch.load(encoder_loc)['encoder_state'])
    loss_func = weighted_mse_loss()

    overall_train_loss = []

    # First, freeze the encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False

    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'freeze the encoder parameters, stage trainable parameter:{trainable_num}')

    # Set the optimizer to only update the predictor
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_for_stage)

    # Training
    all_stage_loss = []
    for i in range(args.epoch_stage):

        model.train()

        train_loss = []
        for step, batch in enumerate(train_loader_stage):
            if args.lds:
                (b_x_adj, b_x_ops, b_y, w) = batch
            else:
                (b_x_adj, b_x_ops, b_y) = batch
            optimizer.zero_grad()
            x, _ = model(b_x_adj, b_x_ops)
            if args.lds:
                loss = loss_func.compute_loss(x, b_y, w)
            else:
                loss = loss_func.compute_loss(x, b_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        all_stage_loss.append(train_loss)
        temp = sum(train_loss) / len(train_loss)
        if args.print_info:
            print('train_loss:%f' % (temp))
        if (i + 1) % 10 == 0:
            overall_train_loss.append(temp)

    # Clear gradients
    optimizer.zero_grad()

    if args.fine_tune:
        print('fine tune stage...')
        for param in model.encoder.parameters():
            param.requires_grad = True

        # Set the optimizer to update both encoder and predictor
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_for_stage)
        # Fine-tuning
        for i in range(args.fine_tune_stage):

            model.train()
            train_loss = []
            for step, batch in enumerate(train_loader_stage):
                if args.lds:
                    (b_x_adj, b_x_ops, b_y, w) = batch
                else:
                    (b_x_adj, b_x_ops, b_y) = batch

                optimizer.zero_grad()
                x, _ = model(b_x_adj, b_x_ops)
                if args.lds:
                    loss = loss_func.compute_loss(x, b_y, w)
                else:
                    loss = loss_func.compute_loss(x, b_y)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            temp = sum(train_loss) / len(train_loss)
            if args.print_info:
                print('train_loss:%f' % (temp))
            if (i + 1) % 10 == 0:
                overall_train_loss.append(temp)

            if args.altering_parameter:
                if (i+1) % 10 == 0:
                    checkpoint = {
                        'epoch': i,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }
                    if not os.path.exists(result_path+'/model_checkpoints'):
                        os.makedirs(result_path+'/model_checkpoints')
                    torch.save(checkpoint, result_path+f'/model_checkpoints/stage2-model-{i+1}.pt')

    y_train = original_label

    return model, y_train


def main():
    args = get_parser()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.task == "vqe_36gate_h_head_new":
        args.gate_type = ['H', 'Rx', 'Ry', 'Rz', 'XX', 'YY', 'ZZ']
        random_searcher = RandomArchitectureSearcher(args.mean, args.standard_deviation, args.gate_type,
                                                     args.num_of_single_qubit_gate, args.p1, args.p2,
                                                     args.preference_num, args.max_gate_num,
                                                     args.num_layers, args.num_qubits)
        Ex = ExpressibilityCalculatorlayer(args.num_random_initial, args.num_qubits, args.num_layers,
                                                args.num_of_single_qubit_gate, args.task, args.seed)

    elif args.task == "vqe_5_constraint":
        args.gate_type = ['Rx', 'Ry', 'Rz', 'XX', 'YY', 'ZZ']
        random_searcher = RandomArchitectureSearcherContraint(args.gate_type, args.num_of_single_qubit_gate,
                                                              args.max_gate_num,
                                                              args.num_layers, args.num_qubits, args.no_parameter_gate,
                                                              args.max_two_qubit_gates_rate, args.edge_index,
                                                              args.start_with_h)
        Ex = ExpressibilityCalculator(
            args.num_random_initial, args.num_qubits, args.parallel, None,
            args.gaussian_kernel_sigma, args.seed, args.task
        )

    elif args.task == "Maxcut_12":
        args.gate_type = ['H', 'Ry', 'Rz', 'CNOT']
        random_searcher = RandomArchitectureSearcherContraint(args.gate_type, args.num_of_single_qubit_gate,
                                                              args.max_gate_num,
                                                              args.num_layers, args.num_qubits, args.no_parameter_gate,
                                                              args.max_two_qubit_gates_rate, args.edge_index,
                                                              args.start_with_h)
        Ex = ExpressibilityCalculator(
            args.num_random_initial, args.num_qubits, args.parallel, None,
            args.gaussian_kernel_sigma, args.seed, args.task
        )
    else:
        print('invalid task')
        exit(0)

    if args.task == "vqe_36gate_h_head_new" or args.task == "vqe_5_constraint":
        if os.path.exists(f'datasets/{args.task}/arc_for_embedding.pkl'):
            print("Load datasets...")
            list_arcs_for_testing = utils.load_pkl(f'datasets/{args.task}/list_arcs_for_testing.pkl')
            arc_for_embedding = utils.load_pkl(f'datasets/{args.task}/arc_for_embedding.pkl')
        else:
            print('No test circuits found, generating fist...')
            exit(0)
    elif args.task == "Maxcut_12":
        if os.path.exists(f'datasets/{args.task}/arc_for_embedding_seed{args.seed2}.pkl'):
            print("Load datasets...")
            list_arcs_for_testing = utils.load_pkl(f'datasets/{args.task}/list_arcs_for_testing_seed{args.seed2}.pkl')
            arc_for_embedding = utils.load_pkl(f'datasets/{args.task}/arc_for_embedding_seed{args.seed2}.pkl')
        else:
            print('No test circuits found, generating fist...')
            exit(0)
    else:
        print('invalid task')
        exit(0)

    result_path = f'result/{args.task}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    argsDict = args.__dict__  # save setting
    with open(result_path + '/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    print('Data loading completed.')
    # embedding
    if args.pretraining:

        pretrainer = Pretraining(args)

        if os.path.exists(
            f'datasets/{args.task}/arc_for_training_encoder.pkl'):  # Check if architectures for training the encoder exist
            print('Existing data for training the encoder found, loading...')
            list_arcs_for_encoder = utils.load_pkl(f'datasets/{args.task}/list_arcs_for_encoder.pkl')
            matrix_arc_for_training_encoder = utils.load_pkl(f'datasets/{args.task}/arc_for_training_encoder.pkl')
            list_arcs_for_encoder = list_arcs_for_encoder[0:args.pretraining_num]
            matrix_arc_for_training_encoder = matrix_arc_for_training_encoder[0:args.pretraining_num]
            print('Loading completed.')

        else:
            # Depending on the task, different types of architectures may need to be generated
            print(f'No data available for training the encoder, generating {args.task} dataset...')

            list_arcs_for_encoder = random_searcher.get_architectures(
                args.pretraining_num, args.generate_type, start_with_h=args.start_with_h
            )
            utils.save_pkl(list_arcs_for_encoder, f'datasets/{args.task}/list_arcs_for_encoder.pkl')
            if args.generate_type == 0:
                matrix_arc_for_training_encoder = random_searcher.list_to_adj(list_arcs_for_encoder)
            else:
                matrix_arc_for_training_encoder = random_searcher.list_to_adj(list_arcs_for_encoder, args.num_qubits, len(args.gate_type))
            utils.save_pkl(matrix_arc_for_training_encoder, f'datasets/{args.task}/arc_for_training_encoder.pkl')

        if os.path.exists(f'datasets/{args.task}/Expressibility.pkl'):
            print('Loading Expressibility datasets...')
            expressibilities = utils.load_pkl(f'datasets/{args.task}/Expressibility.pkl')
            expressibilities = expressibilities[0:args.pretraining_num]
        else:
            print('There is no quantum circuit expressibility data, generating it now...')
            expressibilities = Ex.get_expressibility(list_arcs_for_encoder)
            utils.save_pkl(expressibilities, f'datasets/{args.task}/Expressibility.pkl')

        # Compute the ranking of each data point
        ranks = np.argsort(np.argsort(expressibilities))
        # Compute the quantile of each data point
        expressibilities = (ranks + 1) / (len(expressibilities) + 1)
        print('Data preparation completed.')
        print('Pretraining started.')
        pretrain_model = pretrainer.pretraining(list_arcs_for_encoder, expressibilities, matrix_arc_for_training_encoder,
                                           args)
        pretrain_model.eval()
        print('Pretraining finished.')
        exit(0)
    else:
        # Randomly sample training data from the testing set
        # Shuffle the order
        indices = np.array(np.random.permutation(range(args.num_test_set)))
        # Select training set
        temp1 = []
        temp2 = []
        list1 = []
        list2 = []
        for i in range(len(arc_for_embedding)):
            if i < args.num_training_sample:
                list1.append(list_arcs_for_testing[indices[i]])
                temp1.append(arc_for_embedding[indices[i]])
            if i >= args.num_training_sample:
                list2.append(list_arcs_for_testing[indices[i]])
                temp2.append(arc_for_embedding[indices[i]])

        list_for_predictor = list1
        list_for_testing = list2
        arc_for_predictor = temp1
        arc_for_testing = temp2

        label_for_predictor = get_performance(list_for_predictor)

        if args.self_supervise:
            predictor, y_train = self_supervise_training(arc_for_predictor, label_for_predictor, args,
                                                                      result_path)
            supervised_test(predictor, y_train, arc_for_testing, list_for_testing, args, result_path)
        elif args.BANAQAS:
            BANAQAS_search(list_arcs_for_testing, arc_for_embedding, args, result_path)
        else:
            print('invalid search method.')
            exit(0)

if __name__ == '__main__':
    main()
