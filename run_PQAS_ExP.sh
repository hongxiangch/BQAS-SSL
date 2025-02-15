#TFIM
for ((i=0; i<1; i++)); do
      python main.py --task 'vqe_36gate_h_head_new' --self_supervise True --num_qubits 6 --max_gate_num 36 --num_of_single_qubit_gate 4 --fine_tune True --epoch_stage 100 --fine_tune_stage 50 --num_training_sample 100 --seed2 $i
done

#Heisenberg
for ((i=0; i<20; i++)); do
      python main.py --task 'vqe_5_constraint' --self_supervise True --num_qubits 5 --max_gate_num 38 --num_of_single_qubit_gate 3 --fine_tune True --epoch_stage 600 --fine_tune_stage 300 --num_training_sample 200 --seed2 $i
done

# Max-Cut
for (( i = 0; i < 5; i++ )); do
    for (( j = 0; j < 4; j++ )); do
      python main.py --task 'Maxcut_12' --self_supervise True --num_qubits 12 --max_gate_num 24 --num_of_single_qubit_gate 3 --fine_tune True --epoch_stage 200 --fine_tune_stage 300 --num_training_sample 70 --seed2 $i --seed $j
    done
done