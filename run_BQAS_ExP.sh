#TFIM
for (( i = 0; i < 20; i++ )); do
    python main.py --task 'vqe_36gate_h_head_new' --BANAQAS True --num_qubits 6 --max_gate_num 36 --num_of_single_qubit_gate 4 --epochs_for_init 100 --num_init_sample 30 --seed2 $i
done

#Heisenberg
for (( i = 0; i < 20; i++ )); do
    python main.py --task 'vqe_5_constraint' --BANAQAS True --num_qubits 5 --max_gate_num 38 --num_of_single_qubit_gate 3 --epochs_for_init 100 --num_init_sample 20 --seed2 $i
done

# Max-Cut
for (( i = 0; i < 5; i++ )); do
    for (( j = 0; j < 4; j++ )); do
      python main.py --task 'Maxcut_12' --BANAQAS True --num_qubits 12 --max_gate_num 24 --num_of_single_qubit_gate 3 --epochs_for_init 100 --num_init_sample 20  --k 20 --seed2 $i --seed $j
    done
done