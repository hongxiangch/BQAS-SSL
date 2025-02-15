python datasets_prepare.py  --task 'vqe_36gate_h_head_new' \
                --num_qubits 6 \
                --max_gate_num 36 \
                --num_of_single_qubit_gate 4 \
                --max_two_qubit_gates_rate 0.35 \
                --generate_type 1 \
                --num_test_set 5 \
                --num_layers 10

python datasets_prepare.py  --task 'vqe_5_constraint' \
                --num_qubits 5 \
                --max_gate_num 38 \
                --num_of_single_qubit_gate 3 \
                --max_two_qubit_gates_rate 0.35 \
                --generate_type 0 \
                --num_test_set 5 \
                --num_layers 20
#
python datasets_prepare.py  --task 'Maxcut_12' \
                --num_qubits 12 \
                --max_gate_num 24 \
                --num_of_single_qubit_gate 3 \
                --max_two_qubit_gates_rate 0.25 \
                --generate_type 0 \
                --num_test_set 5 \
                --num_layers 20 \
                --start_with_h True

