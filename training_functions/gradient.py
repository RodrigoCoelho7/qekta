from sklearn.datasets import make_classification
from quantum.model import QuantumKernel
import torch
from sklearn.svm import SVC
from utils.cost_functions import accuracy, kta, frobenian_kta, centered_kta, kernel_polarization
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import ray
import random

costs = {
    "trace_kta": kta,
    "frobenian_kta": frobenian_kta,
    "centered_kta": centered_kta,
    "kernel_polarization": kernel_polarization
}

def get_gradient(config):
    num_samples = config["num_samples"]
    num_features = config["num_features"]
    ansatz = config["ansatz"]
    num_layers = config["num_layers"]
    lr = config["lr"]
    feature_scaling = config["feature_scaling"]
    num_qubits_equal_num_features = config["num_qubits_equal_num_features"]
    full_kta = config["full_kta"]
    subsample_size = config["subsample_size"]
    cost_function = config["cost_function"]
    
    if num_qubits_equal_num_features:
        num_qubits = num_features
    else:
        num_qubits = config["num_qubits"]

    # Load the correct dataset
    if config["dataset"] == "make_classification":
        x,y = make_classification(n_samples = num_samples,n_features = num_features,
                                  n_informative = num_features, n_redundant = 0)
        num_points_test = int(0.2 * len(x))
        num_samples = num_samples - num_points_test
    elif config["dataset"] == "tc" or config["dataset"] == "ls" or config["dataset"] == "hm":
        dataset_npy = np.load("/home/users/coelho/quantum_kernels/efficient_kta/datasets/final_datasets.npy", allow_pickle=True).item()
        x_train = dataset_npy[config["dataset"]][num_features][num_samples]["x_train"]
        y_train = dataset_npy[config["dataset"]][num_features][num_samples]["y_train"]
        x_test = dataset_npy[config["dataset"]][num_features][num_samples]["x_test"]
        y_test = dataset_npy[config["dataset"]][num_features][num_samples]["y_test"]
        num_points_test = x_test.shape[0]
        x = np.concatenate((x_train,x_test))
        y = np.concatenate((y_train,y_test))
    elif config["dataset"] == "checkerboard" or config["dataset"] == "donuts" or config["dataset"] == "sine":
        if config["dataset"] == "checkerboard":
            dataset_npy = np.load("/home/users/coelho/quantum_kernels/efficient_kta/datasets/checkerboard_dataset.npy", allow_pickle=True).item()
        elif config["dataset"] == "donuts":
            dataset_npy = np.load("/home/users/coelho/quantum_kernels/efficient_kta/datasets/donuts.npy", allow_pickle=True).item()
        elif config["dataset"] == "sine":
            dataset_npy = np.load("/home/users/coelho/quantum_kernels/efficient_kta/datasets/small_sine_dataset.npy", allow_pickle=True).item()

        x_train = dataset_npy["x_train"]
        y_train = dataset_npy["y_train"]
        x_test = dataset_npy["x_test"]
        y_test = dataset_npy["y_test"]
        num_points_test = x_test.shape[0]
        x = np.concatenate((x_train,x_test))
        y = np.concatenate((y_train,y_test))
    elif config["dataset"] == "make_variable":
        dataset_npy = np.load("/home/users/coelho/quantum_kernels/efficient_kta/datasets/make_variable.npy", allow_pickle=True).item()

        x_train = dataset_npy[num_samples][num_features]["x_train"]
        x_test = dataset_npy[num_samples][num_features]["x_test"]
        y_train = dataset_npy[num_samples][num_features]["y_train"]
        y_test = dataset_npy[num_samples][num_features]["y_test"]
        num_points_test = x_test.shape[0]
        x = np.concatenate((x_train,x_test))
        y = np.concatenate((y_train,y_test))
    else:
        raise NotImplementedError
    
    if feature_scaling == "pi/2" or feature_scaling == "pi":
        if feature_scaling == "pi/2":
            min_val = - np.pi/2
            max_val = np.pi/2
        elif feature_scaling == "pi":
            min_val = - np.pi
            max_val = np.pi

        scaler = MinMaxScaler(feature_range=(min_val,max_val)).fit(x)
        x = scaler.transform(x)

    y = np.where(y==0,-1,y)
    
    x_train = x[num_points_test:]
    x_test = x[:num_points_test]
    y_train = y[num_points_test:]
    y_test = y[:num_points_test]

    train_dataset = TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))

    train_loader = DataLoader(train_dataset,batch_size=subsample_size,shuffle=True)

    idx = 0
    for i in range(1000):
        quantum_kernel_config = {
            "ansatz": ansatz,
            "num_qubits": num_qubits,
            "num_layers": num_layers,
            "use_input_scaling": config["use_input_scaling"],
            "use_data_reuploading": config["use_data_reuploading"],
            "all_params_random": True,
            "num_features": config["num_features"]
        }

        model = QuantumKernel(quantum_kernel_config)

        opt = optim.Adam(model.parameters(), lr=lr)

        model.train()

        if full_kta:
            x_batch,y_batch = torch.Tensor(x_train), torch.Tensor(y_train)
        else:
            x_batch, y_batch = random.choice(list(train_loader))

        has_negative_class = torch.any(y_batch == -1)
        has_positive_class = torch.any(y_batch == 1)
        has_both_classes = has_negative_class and has_positive_class

        if not has_both_classes:
            continue

        idx +=1
        opt.zero_grad()
    
        x_0 = x_batch.repeat(x_batch.shape[0],1)
        x_1 = x_batch.repeat_interleave(x_batch.shape[0], dim=0)

        output = model(x_0,x_1).to(torch.float32)

        loss = - costs[cost_function](output.reshape(x_batch.shape[0],x_batch.shape[0]),y_batch)
        loss.backward()

        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().numpy().flatten()

        opt.step()

        metrics = {
            "num_layers": num_layers,
            "grads": grads
            }

        ray.train.report(metrics = metrics)
    





    




    
