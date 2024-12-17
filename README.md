# QEKTA: Quantum Efficient Kernel Target Alignment

This repo provides the code for implementing trainable quantum embedding kernels using the Nyström Approximation method (or not). This allows for efficient quantum kernel implementations that scale only linearly with the training dataset size ```N``` in its entire pipeline, from training to the generation of the training kernel matrix that is fed to the SVM.


In ```configs/example.yaml``` one finds an exemplary configuration for a run that analyses the capacities of the Nyström approximation method under coherent noise. The different directories inside ```configs``` contain the different configurations needed to replicate the results of the paper.

To run experiments, use:
```python main.py "path_to_config_file" ```