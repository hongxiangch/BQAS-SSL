# Self-supervised representation learning for Bayesian quantum architecture search

The repository is built upon [tensorcircuit] (https://tensorcircuit.readthedocs.io/en/latest/index.html).

## Setup

- Python 3.11

- Install all required packages using pip:

  ```
  pip install -r requirements.txt
  ```

## Usage examples

This repository includes examples for generating candidate circuits for the target dataset using **gatewise**, 
and **layerwise** search spaces. Circuits in **gatewise** space are generated based on the 
topological constraints of the **ibmq_quito** and **ibmq_casablanca** quantum devices. 
To generate the quantum circuits, run:

```
generate_dataset.sh
```
The encoder is pre-trained using expressibility prediction by running:

```
run_pre_training.sh
```

Experiments for Predictor-based Quantum Architecture Search with SSL using Expressibility Prediction (PQAS-ExP) on the TFIM, Heisenberg, and Maxcut tasks can be executed with:

```
run_PQAS_ExP.sh
```

Experiments for Bayesian QAS framework with pre-trained predictors via Expressibility Prediction (BQAS-ExP) on the TFIM, Heisenberg, and Maxcut tasks can be executed with:


```
run_BQAS_ExP.sh
```






