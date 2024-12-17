import pennylane as qml
import torch.nn as nn
import torch
from quantum.circuits import kernel_hwe, kernel_paper, kernel_uqc

class QuantumKernel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ansatz = config["ansatz"]
        self.num_qubits = config["num_qubits"]
        self.use_input_scaling = config["use_input_scaling"]
        self.use_data_reuploading = config["use_data_reuploading"]
        self.num_features = config["num_features"]
        self.wires = range(self.num_qubits)
        self.num_layers = config["num_layers"]
        all_params_random = config["all_params_random"]
        self.projector = torch.zeros((2**self.num_qubits,2**self.num_qubits))
        self.projector[0,0] = 1
        use_shots = config["use_shots"]
        n_shots = config["n_shots"]
        self.use_coherent_noise = config["use_coherent_noise"]
        self.std = config["std"]
        self.use_depolarizing_noise = config["use_depolarizing_noise"]
        self.depolarizing_strength = config["depolarizing_strength"]
        

        if self.ansatz == "hwe":
            if self.use_input_scaling:
                self.register_parameter(name="input_scaling", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=True))
            else:
                self.register_parameter(name="input_scaling", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=False))
            self.register_parameter(name="variational", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi, requires_grad=True))
        elif self.ansatz == "paper": # ansatz from the paper https://arxiv.org/pdf/2105.02276
            if self.use_input_scaling:
                self.register_parameter(name="input_scaling", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=True))
            else:
                self.register_parameter(name="input_scaling", param = nn.Parameter(torch.ones(self.num_layers,self.num_qubits), requires_grad=False))
            self.register_parameter(name="variational", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits * 2) * 2 * torch.pi, requires_grad=True))
        elif self.ansatz == "uqc":
            if all_params_random:
                self.register_parameter(name="w", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits, self.num_features) * 2* torch.pi, requires_grad = True))
                self.register_parameter(name="b", param = nn.Parameter(torch.rand(self.num_layers, self.num_qubits) * 2 * torch.pi, requires_grad = True))
            else:
                self.register_parameter(name="w", param = nn.Parameter(torch.randn(self.num_layers,self.num_qubits, self.num_features) *0.01, requires_grad = True))
                self.register_parameter(name="b", param = nn.Parameter(torch.zeros(self.num_layers, self.num_qubits), requires_grad = True))
            self.register_parameter(name="phi", param = nn.Parameter(torch.rand(self.num_layers,self.num_qubits) * 2 * torch.pi, requires_grad=True))
        
        if use_shots:
            dev = qml.device("lightning.qubit", wires = self.wires, shots = n_shots)
        elif self.use_depolarizing_noise:
            dev = qml.device("default.mixed", wires = self.wires)
        else:
            dev = qml.device("lightning.qubit", wires = self.wires)

        if self.ansatz == "hwe":
            self.kernel = qml.QNode(kernel_hwe, dev, diff_method="adjoint", interface = "torch")
        elif self.ansatz == "paper":
            if use_shots:
                self.kernel = qml.QNode(kernel_paper, dev, interface = "torch")
            elif self.use_depolarizing_noise:
                self.kernel = qml.QNode(kernel_paper, dev, diff_method="backprop", interface = "torch")
            else:
                self.kernel = qml.QNode(kernel_paper, dev, diff_method="adjoint", interface = "torch")
        elif self.ansatz == "uqc":
            self.kernel = qml.QNode(kernel_uqc, dev, diff_method="adjoint", interface = "torch")

    def forward(self,x,y):
        coherent_noises = {}
        if self.use_coherent_noise:
            # Need to create tensor containing all of the values to be added to the ansatz
            coherent_noises["input_scaling"] = torch.normal(mean = 0, std = self.std, size = (self.num_layers, self.num_qubits), requires_grad=False)
            coherent_noises["variational"] = torch.normal(mean = 0, std =self.std, size = (self.num_layers,self.num_qubits * 2), requires_grad=False)
        else:
            coherent_noises["input_scaling"] = torch.zeros(size = (self.num_layers, self.num_qubits), requires_grad=False)
            coherent_noises["variational"] = torch.zeros(size = (self.num_layers,self.num_qubits * 2), requires_grad=False)

        all_zero_state = self.kernel(x,y,self._parameters,self.wires,self.num_layers,self.projector, self.use_data_reuploading, coherent_noises, self.use_depolarizing_noise, self.depolarizing_strength)
        return all_zero_state

    