from sklearn.datasets import make_classification
from quantum.model import QuantumKernel
import torch
from sklearn.svm import SVC
from utils.cost_functions import accuracy, kta
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import ray

def random_init(config):
    num_samples = config["num_samples"]
    num_features = config["num_features"]
    ansatz = config["ansatz"]
    num_layers = config["num_layers"]
    lr = config["lr"]
    feature_scaling = config["feature_scaling"]
    num_qubits_equal_num_features = config["num_qubits_equal_num_features"]
    
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
    test_dataset = TensorDataset(torch.Tensor(x_test),torch.Tensor(y_test))

    quantum_kernel_config = {
        "ansatz": ansatz,
        "num_qubits": num_qubits,
        "num_layers": num_layers,
        "use_input_scaling": config["use_input_scaling"],
        "use_data_reuploading": config["use_data_reuploading"],
        "all_params_random": False
    }

    model = QuantumKernel(quantum_kernel_config)

    opt = optim.Adam(model.parameters(), lr=lr)

    "Computing the initial kernel matrix"
    x_0 = train_dataset.tensors[0].repeat(num_samples,1)
    x_1 = train_dataset.tensors[0].repeat_interleave(num_samples, dim=0)
    results = model(x_0,x_1).to(torch.float32)
    kernel_matrix = results.reshape(num_samples, num_samples).detach().numpy()

    "Fitting an SVM to the original kernel matrix"
    svm = SVC(kernel = "precomputed").fit(kernel_matrix,y_train)

    "Computing the accuracy of this initial kernel"
    accuracy_train_init = accuracy(svm, kernel_matrix, train_dataset.tensors[1].detach().numpy())

    "Computing the initial testing kernel matrix"
    x_0 = test_dataset.tensors[0].repeat_interleave(num_samples,dim=0)
    x_1 = train_dataset.tensors[0].repeat(num_points_test, 1)
    results_test = model(x_0,x_1).to(torch.float32)
    kernel_matrix_test = results_test.reshape(num_points_test, num_samples).detach().numpy()

    accuracy_test_init = accuracy(svm, kernel_matrix_test, test_dataset.tensors[1].detach().numpy())

    "Computing the training KTA of this initial kernel"
    kta_init_training = kta(results.reshape(num_samples,num_samples),train_dataset.tensors[1]).item()

    training_predictions_initial = svm.predict(kernel_matrix)

    testing_predictions_initial = svm.predict(kernel_matrix_test)

    metrics = {
        "num_layers": num_layers,
        "accuracy_train_init": accuracy_train_init,
        "accuracy_test_init": accuracy_test_init,
        "kta_train_init": kta_init_training,
        "training_predictions_initial": training_predictions_initial,
        "testing_predictions_initial": testing_predictions_initial    }

    ray.train.report(metrics = metrics)
    





    




    
