import os
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import utils
from tqdm import tqdm
from multiprocessing import Pool

class VqeTrainer:
    def __init__(self, n_cir_parallel, n_qubit, gate_number, n_depth, n_runs, single_qubit_gate, h, max_iteration,
                 seed,noise_param, task):
        self.task = task
        self.K = tc.set_backend("tensorflow")
        self.seed = seed
        self.n_qubit = n_qubit
        self.n_depth = n_depth
        self.n_runs = n_runs
        self.n_cir_parallel = n_cir_parallel
        self.max_iteration = max_iteration
        self.num_gate = gate_number
        self.single_qubit_num = single_qubit_gate
        self.hamiltonian_ = h
        self.lattice = tc.templates.graphs.Line1D(self.n_qubit, pbc=self.hamiltonian_['pbc'])
        self.h = tc.quantum.heisenberg_hamiltonian(self.lattice, hzz=self.hamiltonian_['hzz'],
                                                   hxx=self.hamiltonian_['hxx'], hyy=self.hamiltonian_['hyy'],
                                                   hx=self.hamiltonian_['hx'], hy=self.hamiltonian_['hy'],
                                                   hz=self.hamiltonian_['hz'], sparse=self.hamiltonian_['sparse'])
        self.give_up_rest = False
        self.solution = None

        self.zz = np.kron(tc.gates._z_matrix, tc.gates._z_matrix)
        self.xx = np.kron(tc.gates._x_matrix, tc.gates._x_matrix)
        self.yy = np.kron(tc.gates._y_matrix, tc.gates._y_matrix)

        if noise_param is None:
            self.noise = False
        else:
            self.noise = True
        self.two_qubit_channel_depolarizing_p = None
        self.single_qubit_channel_depolarizing_p = None
        self.bit_flip_p = None
        if self.noise:
            self.two_qubit_channel_depolarizing_p = noise_param['two_qubit_depolarizing_p']
            self.single_qubit_channel_depolarizing_p = noise_param['single_qubit_depolarizing_p']
            self.bit_flip_p = noise_param['bit_flip_p']
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(
                self.two_qubit_channel_depolarizing_p / 15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(
                self.single_qubit_channel_depolarizing_p / 3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

    def vqef(self, param, structure):
        """
        :param param:
        :param structure:
        :return:
        """
        if self.noise:
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)

            c = tc.DMCircuit(self.n_qubit)
            if self.task == 'vqe_36gate_h_head_new':
                for i in range(0, len(structure)):
                    if structure[i][0] == 0:
                        c.h(structure[i][1])
                        c.general_kraus(self.single_qubit_dep_channel, structure[i][1])
                    elif structure[i][0] == 1:
                        c.rx(structure[i][1], theta=param[i])
                        c.general_kraus(self.single_qubit_dep_channel,structure[i][1] )
                    elif structure[i][0] == 2:
                        c.ry(structure[i][1], theta=param[i])
                        c.general_kraus(self.single_qubit_dep_channel, structure[i][1])
                    elif structure[i][0] == 3:
                        c.rz(structure[i][1], theta=param[i])
                        c.general_kraus(self.single_qubit_dep_channel, structure[i][1])
                    elif structure[i][0] == 4:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.xx, theta=param[i])
                        c.general_kraus(self.two_qubit_dep_channel, structure[i][1], structure[i][2])
                    elif structure[i][0] == 5:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.yy, theta=param[i])
                        c.general_kraus(self.two_qubit_dep_channel, structure[i][1], structure[i][2])
                    elif structure[i][0] == 6:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.zz, theta=param[i])
                        c.general_kraus(self.two_qubit_dep_channel, structure[i][1], structure[i][2])
            elif self.task == 'vqe_5_constraint':
                for i in range(0, len(structure)):
                    if structure[i][0] == 0:
                        c.rx(structure[i][1], theta=param[i])
                        c.general_kraus(self.single_qubit_dep_channel,structure[i][1] )
                    elif structure[i][0] == 1:
                        c.ry(structure[i][1], theta=param[i])
                        c.general_kraus(self.single_qubit_dep_channel, structure[i][1])
                    elif structure[i][0] == 2:
                        c.rz(structure[i][1], theta=param[i])
                        c.general_kraus(self.single_qubit_dep_channel, structure[i][1])
                    elif structure[i][0] == 3:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.xx, theta=param[i])
                        c.general_kraus(self.two_qubit_dep_channel, structure[i][1], structure[i][2])
                    elif structure[i][0] == 4:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.yy, theta=param[i])
                        c.general_kraus(self.two_qubit_dep_channel, structure[i][1], structure[i][2])
                    elif structure[i][0] == 5:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.zz, theta=param[i])
                        c.general_kraus(self.two_qubit_dep_channel, structure[i][1], structure[i][2])
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)

            st = c.state()
            x = tf.matmul(st, self.h)
            e = tf.linalg.trace(x)
            e = self.K.real(e)

        else:
            c = tc.Circuit(self.n_qubit)
            if self.task == 'vqe_36gate_h_head_new':
                for i in range(0, len(structure)):
                    if structure[i][0] == 0:
                        c.h(structure[i][1])
                    elif structure[i][0] == 1:
                        c.rx(structure[i][1], theta=param[i])
                    elif structure[i][0] == 2:
                        c.ry(structure[i][1], theta=param[i])
                    elif structure[i][0] == 3:
                        c.rz(structure[i][1], theta=param[i])
                    elif structure[i][0] == 4:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.xx, theta=param[i])
                    elif structure[i][0] == 5:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.yy, theta=param[i])
                    elif structure[i][0] == 6:
                        c.exp1(structure[i][1], structure[i][2], unitary=self.zz, theta=param[i])
            elif self.task == 'vqe_5_constraint':
                for i in range(0, len(structure)):
                    if structure[i][0] == 0:
                        c.rx(structure[i][1], theta=param[i])
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
            e = tc.templates.measurements.operator_expectation(c, self.h)
        return e

    def train_circuit(self, cirs):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        tc.set_backend("tensorflow")
        result = []
        params = []
        for index, arc in tqdm(enumerate(cirs), desc='Optimizing circuits'):

            energy, param = self.batched_train_step_tf(self.n_runs, arc, self.max_iteration)
            energy = energy.numpy()
            result.append(energy.min())
            params.append(param[1][np.argsort(energy)[0]])
            if self.give_up_rest:
                if result[-1] <= self.solution:
                    break
        return [result, params]

    def process(self, arcs):

        if self.n_cir_parallel == 1:
            result = self.train_circuit(arcs)
            energy_result = result[0]
            param_result = result[1]
        else:
            pool = Pool(processes=self.n_cir_parallel)
            input_list = []
            cir_per_process = int(np.ceil(len(arcs) / self.n_cir_parallel))
            temp = []
            for i in range(0, len(arcs)):
                if len(temp) < cir_per_process:
                    temp.append(arcs[i])
                else:
                    input_list.append(temp)
                    temp = []
                    temp.append(arcs[i])
            input_list.append(temp)

            result = pool.map(self.train_circuit, input_list)
            pool.close()
            pool.join()
            energy_result = []
            param_result = []
            for part in range(0, len(result)):
                energy_result.extend(result[part][0])
                param_result.extend(result[part][1])

        return np.array(energy_result), np.array(param_result)

    def batched_train_step_tf(self, batch, structure, maxiter=10000):
        trainer = tc.backend.jit(tc.backend.vvag(self.vqef, argnums=0, vectorized_argnums=0), )
        pi = 2 * np.pi
        par = np.random.uniform(-1, 1, batch * len(structure))

        par = par.reshape((batch, len(structure))) * pi

        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )
        param_initial = param.numpy()
        opt = tf.keras.optimizers.Adam(0.01)
        e_last = np.full((batch, ), 1000)
        for i in range(maxiter):
            e, grad = trainer(param, structure)
            opt.apply_gradients([(grad, param)])
            if i % 100 == 0:
                distance = abs(e_last - e.numpy())
                if distance.max() < 0.0001:
                    # print(distance.max())
                    break
                else:
                    e_last = e.numpy()
        param_end = param.numpy()
        return e, [param_initial, param_end]


class MaxCutTrainer:
    def __init__(self, gate_type, num_qubits, num_layers, graph, n_cir_parallel, n_runs, max_iteration, seed, theoretical=None, noise_param=None):
        self.K = tc.set_backend("tensorflow")
        self.gate_pool = gate_type
        self.n_qubit = num_qubits
        self.n_depth = num_layers
        self.graph = graph
        self.seed = seed
        self.theoretical = theoretical
        self.max_iteration = max_iteration
        self.n_cir_parallel = n_cir_parallel
        self.n_runs = n_runs
        self.give_up_rest = False

        if noise_param is None:
            self.noise = False
        else:
            self.noise = True
        self.two_qubit_channel_depolarizing_p = None
        self.single_qubit_channel_depolarizing_p = None
        self.bit_flip_p = None
        if self.noise:
            self.two_qubit_channel_depolarizing_p = noise_param['two_qubit_channel_depolarizing_p']
            self.single_qubit_channel_depolarizing_p = noise_param['single_qubit_channel_depolarizing_p']
            self.bit_flip_p = noise_param['bit_flip_p']
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.two_qubit_channel_depolarizing_p/15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.single_qubit_channel_depolarizing_p/3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

    def compute_theoretical(self):
        """
            Calculate the theoretical optimal solution
            :param graph: The graph corresponding to the problem
            :return: The energy of the optimal solution and the optimal solution itself
        """
        m = np.zeros((2**self.n_qubit, 2**self.n_qubit), dtype='complex128')
        for e in self.graph.edges:
            c = tc.Circuit(self.n_qubit)
            c.z(e[0])
            c.z(e[1])
            m += c.matrix()
        value_h, vector_h = np.linalg.eig(m)
        value_h = value_h.real
        min_value = np.min(value_h)
        solution = []
        min_index = np.where(value_h <= min_value)
        for m_i in min_index[0]:
            # solution.append()
            found_solution = bin(int(m_i))[2:]
            res = []
            for _ in range(0, self.n_qubit - len(found_solution)):
                res.append(0)
            for i in range(0, len(found_solution)):
                if found_solution[i] == '0':
                    res.append(0)
                elif found_solution[i] == '1':
                    res.append(1)
                else:
                    print('invalid number')
                    exit(0)
            solution.append(res)
        cutted_edge = 0
        for edge in self.graph.edges:
            if solution[0][edge[0]] != solution[0][edge[1]]:
                cutted_edge += 1

        output = {'min_value': min_value, 'cutted_edge': cutted_edge, 'solution': solution}
        return output

    def tfi_energy(self, c):
        loss = 0.0
        for e in self.graph.edges:
            loss += c.expectation_ps(z=[e[0], e[1]])

        return self.K.real(loss)

    def compute_energy(self, param, structure):
        """
        :param param:
        :param structure:
        :return:
        """
        if self.noise:
            print('noise: ', self.noise)
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)

            c = tc.DMCircuit(self.n_qubit)
            for i in range(0, len(structure)):
                if structure[i][0] == 0:
                    c.h(structure[i][1])
                    c.general_kraus(self.single_qubit_dep_channel, structure[i][1])
                elif structure[i][0] == 1:
                    c.ry(structure[i][1], theta=param[i])
                    c.general_kraus(self.single_qubit_dep_channel, structure[i][1])
                elif structure[i][0] == 2:
                    c.rz(structure[i][1], theta=param[i])
                    c.general_kraus(self.single_qubit_dep_channel, structure[i][1])
                elif structure[i][0] == 3:
                    c.cnot(structure[i][1], structure[i][2])
                    c.general_kraus(self.two_qubit_dep_channel, structure[i][1], structure[i][2])
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)
        else:
            c = tc.Circuit(self.n_qubit)
            for i in range(0, len(structure)):
                if structure[i][0] == 0:
                    c.h(structure[i][1])
                elif structure[i][0] == 1:
                    c.ry(structure[i][1], theta=param[i])
                elif structure[i][0] == 2:
                    c.rz(structure[i][1], theta=param[i])
                elif structure[i][0] == 3:
                    c.cnot(structure[i][1], structure[i][2])

        e = self.tfi_energy(c)
        state = c.state()
        return e, state

    def get_param_num(self, cir):
        param_num = 0
        for i in range(len(cir)):
            if cir[i].para_gate:
                param_num += 1
        return param_num

    def depth_count(self, cir, qubit):
        res = [0] * qubit
        # count = 0
        for gate in cir:
            if gate[0] > 2:
                depth_q = []
                depth_q.append(res[gate[1]])
                depth_q.append(res[gate[2]])
                max_depth = max(depth_q)
                max_depth += 1
                res[gate[1]] = max_depth
                res[gate[2]] = max_depth
            else:
                res[gate[0]] += 1

        depth = np.max(res)
        return depth

    def target_amplitude_square_sum(self, quantum_state, target_bases):

        total_prob = 0.0
        quantum_state = quantum_state.numpy()
        if self.noise:
            prob = quantum_state.diagonal().real
        else:
            conjugate_state = quantum_state.conjugate()
            prob = (quantum_state*conjugate_state).real

        for basis in target_bases:
            index = int("".join(map(str, basis)), 2)
            total_prob += prob[index]
        return total_prob

    def train_circuit(self, work_queue):
        tc.set_backend("tensorflow")
        tc.set_dtype("complex128")
        single_circuit = work_queue[0]
        c_index = work_queue[1]
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        trainer = tc.backend.jit(tc.backend.value_and_grad(self.compute_energy, argnums=0, has_aux=True))
        param_num = self.get_param_num(single_circuit)
        L = self.depth_count(single_circuit, self.n_qubit)
        par = np.random.normal(loc=0, scale=1 / (8 * (L + 2)), size=param_num)
        par = par.reshape((param_num, 1))
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )
        param_initial = param.numpy()
        e_last = 1000
        energy_epoch = []
        prob_epoch = []
        opt = tf.keras.optimizers.Adam(0.05)
        for i in range(self.max_iteration):
            (e, state), grad = trainer(param, single_circuit)
            energy_epoch.append(e.numpy())
            opt.apply_gradients([(grad, param)])
            prob = self.target_amplitude_square_sum(state, self.theoretical['solution'])
            prob_epoch.append(prob)

            if i % 100 == 0:
                distance = abs(e_last - e.numpy())
                if distance < 0.0001:
                    # print(distance.max())
                    break
                else:
                    e_last = e.numpy()

        print(f'circuit: {c_index}, epoch: {i}, energy: {e}, prob: {prob}')
        cut_edge = 1/2*(len(self.graph.edges) - e.numpy())
        return e, energy_epoch, prob_epoch, cut_edge


    def process(self, arcs):
        work_queue = []
        for i in range(0, len(arcs)):
            work_queue.append([arcs[i], i])
        # self.train_circuit(work_queue[56])
        pool = Pool(processes=self.n_cir_parallel)
        result = pool.map(self.train_circuit, work_queue)
        pool.close()
        pool.join()

        energy, energy_epoch, pro_epoch, cut_edges = [], [], [], []
        for part in result:
            energy.append(part[0])
            energy_epoch.append(part[1])
            pro_epoch.append(part[2])
            cut_edges.append(part[3])

        return energy, energy_epoch, pro_epoch, cut_edges

