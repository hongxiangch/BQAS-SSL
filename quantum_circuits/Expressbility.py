import os
import numpy as np
from tqdm import tqdm
import tensorcircuit as tc
import tensorflow as tf
from scipy import stats
from utils import load_pkl, save_pkl
import random
import argparse
from multiprocessing import Pool
import torch
import time
tc.set_dtype("complex128")
tc.set_backend("tensorflow")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ExpressibilityCalculator:
    def __init__(self, num_random_initial, qubits, n_cir_parallel, noise_param, gaussian_kernel_sigma, seed, task) :
        self.task = task
        self.pi_2 = 2 * np.pi
        self.num_random_initial = num_random_initial
        self.n_qubit = qubits
        self.n_cir_parallel = n_cir_parallel
        self.noise = False
        self.seed = seed
        self.gaussian_kernel_sigma = gaussian_kernel_sigma
        if noise_param is None:
            self.noise = False
        else:
            self.noise = True
        self.two_qubit_channel_depolarizing_p = None
        self.single_qubit_channel_depolarizing_p = None
        self.bit_flip_p = None
        self.zz = np.kron(tc.gates._z_matrix, tc.gates._z_matrix)
        self.xx = np.kron(tc.gates._x_matrix, tc.gates._x_matrix)
        self.yy = np.kron(tc.gates._y_matrix, tc.gates._y_matrix)
        if self.noise:
            print(f'noise: invalid')
            exit(0)

    def quantum_circuit(self, structure, param):
        """
        :param param:
        :param structure:
        :return:
        """
        if self.noise:
            print(f'noise: invalid')
            exit(0)
        else:
            c = tc.Circuit(self.n_qubit)
            if self.task == 'vqe_5_constraint':
                for i in range(0, len(structure)):
                    if structure[i][0] == 0:
                        c.rx(structure[i][1], theta=param[i])  # 前面是作用比特的位置，后面旋转角度
                    elif structure[i][0] == 1:
                        c.ry(structure[i][1], theta=param[i])
                    elif structure[i][0] == 2:
                        c.rz(structure[i][1], theta=param[i])
                    elif structure[i][0] == 3:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.xx, theta=param[i])
                    elif structure[i][0] == 4:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.yy, theta=param[i])
                    elif structure[i][0] == 5:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.zz, theta=param[i])
                    else:
                        print("invalid gate!")
                        exit(0)

            elif self.task == 'Maxcut_12':
                for i in range(0, len(structure)):
                    if structure[i][0] == 1:
                        c.ry(structure[i][1], theta=param[i])
                    elif structure[i][0] == 2:
                        c.rz(structure[i][1], theta=param[i])
                    elif structure[i][0] == 3:
                        c.cnot(structure[i][1], structure[i][2])
                    else:
                        print("invalid gate!")
                        exit(0)
        st = c.state()
        return st

    def get_parallel(self):
        parallel = tc.backend.vmap(self.quantum_circuit, vectorized_argnums=1)
        parallel = tc.backend.jit(parallel, static_argnums=(0))
        return parallel

    def train_circuit(self, circuit):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)
        parallel = self.get_parallel()
        par = np.random.uniform(0, 1, (self.num_random_initial, len(circuit), 1)) * self.pi_2
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )
        output_states = parallel(circuit, param)
        return output_states.numpy()

    def process(self, circuits):
        pool = Pool(processes=self.n_cir_parallel)
        quantum_state = pool.map(self.train_circuit, circuits)
        pool.close()
        pool.join()
        return quantum_state

    def fidelity_calculator(self, circuits):
        fidelities = []
        output_states = self.process(circuits)
        for i in range(len(output_states)):
            output_states1 = output_states[i][0:int(self.num_random_initial / 2)]
            output_states2 = output_states[i][int(self.num_random_initial / 2):]
            fidelity = (output_states1 * output_states2.conjugate()).sum(-1)
            fidelity = np.power(np.absolute(fidelity), 2)
            fidelities.append(fidelity)
        return fidelities

    def compute_expressibility_KL(self, circuits):
        N = self.n_qubit
        points = 100
        space = 1 / points
        x = [space * (i + 1) for i in range(-1, points)]
        haar_points = [] #shuz
        for i in range(1, len(x)):
            temp1 = -1 * np.power((1 - x[i]), np.power(2, N) - 1)
            temp0 = -1 * np.power((1 - x[i - 1]), np.power(2, N) - 1)
            haar_points.append(temp1 - temp0)
        haar_points = np.array(haar_points)
        fidelities = self.fidelity_calculator(circuits)
        expressivity = []
        for inner in tqdm(fidelities, desc='Computing expressivity'):
            bin_index = np.floor(inner * points).astype(int)
            num = []
            for i in range(0, points):
                num.append(len(bin_index[bin_index == i]))
            num = np.array(num) / sum(num)
            output = stats.entropy(num, haar_points)
            expressivity.append(output)
        expressivity = np.array(expressivity)
        expressivity = -1 * expressivity
        return expressivity

    def get_expressibility(self, circuits):
        expressibility = self.compute_expressibility_KL(circuits)
        return expressibility

class ExpressibilityCalculatorlayer:
    def __init__(self, num_random_initial, qubits, n_depth, num_single_qubit_gate, task, seed):
        self.pi_2 = 2 * np.pi
        self.num_random_initial = num_random_initial
        self.qubits = qubits
        self.n_depth = n_depth
        self.num_single_qubit_gate = num_single_qubit_gate
        self.K = tc.set_backend("tensorflow")
        self.swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex64)
        self.task = task
        self.seed = seed
        self.processors_for_fidelity = 1

    def gate_list_vqe(self, param):  # H RX RY RZ XX YY ZZ
        l = [
            tc.gates.Gate(self.K.eye(4)),
            tc.gates.Gate(self.K.kron(tc.gates._h_matrix.astype('complex128'), self.K.eye(2))),
            tc.gates.Gate(self.K.kron(tc.gates.rx_gate(theta=param).tensor, self.K.eye(2))),
            tc.gates.Gate(self.K.kron(tc.gates.ry_gate(theta=param).tensor, self.K.eye(2))),
            tc.gates.Gate(self.K.kron(tc.gates.rz_gate(theta=param).tensor, self.K.eye(2))),
            tc.gates.exp1_gate(theta=param, unitary=tc.gates._xx_matrix),
            tc.gates.exp1_gate(theta=param, unitary=tc.gates._yy_matrix),
            tc.gates.exp1_gate(theta=param, unitary=tc.gates._zz_matrix)]
        return [tc.backend.reshape2(m.tensor) for m in l if isinstance(m, tc.gates.Gate)]

    def gate_list_vqe_4(self, param):  # RX RY RZ XX YY ZZ
        l = [
            tc.gates.Gate(self.K.eye(4)),
            tc.gates.Gate(self.K.kron(tc.gates.rx_gate(theta=param).tensor, self.K.eye(2))),
            tc.gates.Gate(self.K.kron(tc.gates.ry_gate(theta=param).tensor, self.K.eye(2))),
            tc.gates.Gate(self.K.kron(tc.gates.rz_gate(theta=param).tensor, self.K.eye(2))),
            tc.gates.exp1_gate(theta=param, unitary=tc.gates._xx_matrix),
            tc.gates.exp1_gate(theta=param, unitary=tc.gates._yy_matrix),
            tc.gates.exp1_gate(theta=param, unitary=tc.gates._zz_matrix)]
        return [tc.backend.reshape2(m.tensor) for m in l if isinstance(m, tc.gates.Gate)]

    def quantum_circuit_vqe(self, structure, num_qubit, rotational_angle):
        print('compiling...')
        c = tc.Circuit(num_qubit)
        for i in range(structure.shape[0]):
            for j in range(num_qubit):
                c.select_gate(structure[i, j], self.gate_list_vqe(rotational_angle[i, j]), j, ((j + 1) % num_qubit))
        s = c.state()
        return s

    def quantum_circuit_vqe_4(self, structure, num_qubit, rotational_angle):
        print('compiling...')
        c = tc.Circuit(num_qubit)
        for i in range(structure.shape[0]):
            for j in range(num_qubit):
                c.select_gate(structure[i, j], self.gate_list_vqe_4(rotational_angle[i, j]), j, ((j + 1) % num_qubit))
        s = c.state()
        return s

    def fidelity_calculator(self, circuits):
        fidelities = []
        if self.task == 'vqe_36gate_h_head_new':
            parallel = tc.backend.jit(
                tc.backend.vmap(self.quantum_circuit_vqe, vectorized_argnums=(2,)))
        elif (self.task == 'vqe_5') or (self.task == 'BeH2'):
            parallel = tc.backend.jit(
                tc.backend.vmap(self.quantum_circuit_vqe_4, vectorized_argnums=(2,)))
        else:
            parallel = None
            print('invalid task')
            exit(0)

        for index in tqdm(range(0, len(circuits)), desc='Computing fidelity'):
            structure = np.zeros((self.n_depth, self.qubits), dtype=np.int64)
            count = [0 for _ in range(self.qubits)]
            for i in range(0, len(circuits[index])):
                if circuits[index][i][0] > (self.num_single_qubit_gate - 1):
                    if count[circuits[index][i][1]] > count[circuits[index][i][2]]:
                        structure[count[circuits[index][i][1]]][circuits[index][i][1]] = circuits[index][i][0] + 1
                        count[circuits[index][i][1]] += 1
                        count[circuits[index][i][2]] = count[circuits[index][i][1]]
                    else:
                        structure[count[circuits[index][i][2]]][circuits[index][i][1]] = circuits[index][i][0] + 1
                        count[circuits[index][i][2]] += 1
                        count[circuits[index][i][1]] = count[circuits[index][i][2]]
                else:
                    structure[count[circuits[index][i][1]]][circuits[index][i][1]] = circuits[index][i][0] + 1
                    count[circuits[index][i][1]] += 1

            para = np.random.uniform(-1, 1, self.num_random_initial * self.n_depth * self.qubits)
            para = para.reshape((self.num_random_initial, self.n_depth, self.qubits)) * self.pi_2
            para = tf.Variable(
                initial_value=tf.convert_to_tensor(para, dtype=getattr(tf, tc.rdtypestr))
            )
            structure = tc.backend.cast(structure, "int64")
            output_states = parallel(structure, self.qubits, para)
            output_states = output_states.numpy()
            output_states1 = output_states[0:int(self.num_random_initial/2)]
            output_states2 = output_states[int(self.num_random_initial/2):]
            fidelity = (output_states1*output_states2.conjugate()).sum(-1)
            fidelity = np.power(np.absolute(fidelity), 2)
            fidelities.append(fidelity)

        return fidelities

    def compute_expressibility(self, circuits):
        N = self.qubits
        points = 100
        space = 1 / points
        x = [space * (i + 1) for i in range(-1, points)]
        haar_points = []
        for i in range(1, len(x)):
            temp1 = -1 * np.power((1 - x[i]), np.power(2, N) - 1)
            temp0 = -1 * np.power((1 - x[i - 1]), np.power(2, N) - 1)
            haar_points.append(temp1 - temp0)
        haar_points = np.array(haar_points)

        fidelities = self.fidelity_calculator(circuits)
        expressivity = []
        for inner in tqdm(fidelities, desc='Computing expressivity'):
            bin_index = np.floor(inner * points).astype(int)
            num = []
            for i in range(0, points):
                num.append(len(bin_index[bin_index == i]))
            num = np.array(num) / sum(num)
            output = stats.entropy(num, haar_points)
            expressivity.append(output)
        expressivity = np.array(expressivity)
        expressivity = -1 *expressivity
        # expressivity = np.log10(expressivity)

        return expressivity

    def get_expressibility(self, circuits):
        expressibility = self.compute_expressibility(circuits)
        return expressibility

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--parallel", type=int, default=5, help="parallel processing")
    parser.add_argument("--proxy", type=str, default='EX', help="proxy file")
    parser.add_argument("--task", type=str, default='Maxcut_12', help="task")
    parser.add_argument('--search_space', type=str, default='block', help='')
    parser.add_argument("--num_random_initial", type=int, default=2000, help="number of random initial for fidelities calcualtion")
    parser.add_argument("--qubits", type=int, default=12, help="qubit")
    parser.add_argument("--noise", type=int, default=False, help="noise")
    parser.add_argument("--two_qubit_channel_depolarizing_p", type=float, default=0.01, help="two_qubit_noise")
    parser.add_argument("--single_qubit_channel_depolarizing_p", type=float, default=0.001, help="single_qubit_noise")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="bit_flip_noise")
    parser.add_argument("--haarPoints", type=int, default=100, help="KL_haar Points")
    parser.add_argument("--gaussian_kernel_sigma", type=float, default=0.01, help="MMD_gaussian_kernel_sigma")
    args = parser.parse_args()
    start_time = time.time()
    # noise = False
    noise_param = None
    if args.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': args.two_qubit_channel_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': args.single_qubit_channel_depolarizing_p,
                       'bit_flip_p': args.bit_flip_p}

    list_cir = load_pkl(f'list_arcs_for_vae.pkl')
    list_cir = list_cir
    Ex = ExpressibilityCalculator(args.num_random_initial, args.num_qubits, args.parallel, noise_param, args.gaussian_kernel_sigma, args.seed)
    # save_path = f'result/{args.task}/{args.search_space}_seed{args.seed}_pretrained/EX/'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    for i in range(15, 25):
        expressibility = Ex.get_expressibility(list_cir[i*200: (i+1)*200])
        save_pkl(expressibility, f'Express/expressibility_{(i+1)*200}.pkl')
    end_time = time.time()
    print(f'Time:{end_time - start_time}s')
    # save_pkl([end_time - start_time], f'result/{args.task}/{args.search_space}_seed{args.seed}_pretrained/running_time_EX.pkl')
    print('------------------------')
