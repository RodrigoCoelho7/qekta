from sklearn.datasets import make_classification
from quantum.model import QuantumKernel
import torch
from sklearn.svm import SVC
from utils.cost_functions import accuracy, kta, frobenian_kta, centered_kta, kernel_polarization
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import ray
import copy
import os

costs = {
    "trace_kta": kta,
    "frobenian_kta": frobenian_kta,
    "centered_kta": centered_kta,
    "kernel_polarization": kernel_polarization
}

def run(config):
    num_samples = config["num_samples"]
    num_features = config["num_features"]
    subsample_size = config["subsample_size"]
    ansatz = config["ansatz"]
    num_layers = config["num_layers"]
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    feature_scaling = config["feature_scaling"]
    num_qubits_equal_num_features = config["num_qubits_equal_num_features"]
    full_kta = config["full_kta"]
    cost_function = config["cost_function"]
    use_nystrom_approx = config["use_nystrom_approx"]
    validate_every_epoch = config["validate_every_epoch"]

    if num_qubits_equal_num_features:
        num_qubits = num_features
    else:
        num_qubits = config["num_qubits"]

    relative_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")

    if config["dataset"] == "checkerboard" or config["dataset"] == "donuts" or config["dataset"] == "sine":
        if config["dataset"] == "checkerboard":
            dataset_npy = np.load(os.path.join(relative_path, "checkerboard_dataset.npy"), allow_pickle=True).item()
        elif config["dataset"] == "donuts":
            dataset_npy = np.load(os.path.join(relative_path, "donuts.npy"), allow_pickle=True).item()
        elif config["dataset"] == "sine":
            dataset_npy = np.load(os.path.join(relative_path, "small_sine_dataset.npy"), allow_pickle=True).item()

        x_train = dataset_npy["x_train"]
        y_train = dataset_npy["y_train"]
        x_test = dataset_npy["x_test"]
        y_test = dataset_npy["y_test"]
        num_points_test = x_test.shape[0]
        x = np.concatenate((x_train,x_test))
        y = np.concatenate((y_train,y_test))

    elif config["dataset"] == "spirals" or config["dataset"] == "corners":
        if config["dataset"] == "spirals":
            dataset_npy = np.load(os.path.join(relative_path, "spirals.npy"), allow_pickle=True).item()
        elif config["dataset"] == "corners":
            dataset_npy = np.load(os.path.join(relative_path, "corners.npy"), allow_pickle=True).item()

        x_train = dataset_npy[num_samples]["x_train"]
        x_test = dataset_npy[num_samples]["x_test"]
        y_train = dataset_npy[num_samples]["y_train"]
        y_test = dataset_npy[num_samples]["y_test"]
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

    train_loader = DataLoader(train_dataset,batch_size=subsample_size,shuffle=True)

    quantum_kernel_config = {
        "ansatz": ansatz,
        "num_qubits": num_qubits,
        "num_layers": num_layers,
        "use_input_scaling": config["use_input_scaling"],
        "use_data_reuploading": config["use_data_reuploading"],
        "num_features": config["num_features"],
        "all_params_random": False,
        "use_shots": config["use_shots"],
        "n_shots": config["n_shots"],
        "use_coherent_noise": config["use_coherent_noise"],
        "std": config["std"],
        "use_depolarizing_noise": config["use_depolarizing_noise"],
        "depolarizing_strength": config["depolarizing_strength"]
    }

    model = QuantumKernel(quantum_kernel_config)
    initial_weights_numpy = {name: np.copy(param.detach().cpu().numpy()) for name, param in model.state_dict().items()}

    opt = optim.Adam(model.parameters(), lr=lr)

    if use_nystrom_approx:

        nystrom_loader = DataLoader(train_dataset,batch_size=config["num_landmarks"],shuffle=True)
        x_batch_nystrom, y_batch_nystrom = random.choice(list(nystrom_loader))

        x_batch_nystrom_np = np.copy(x_batch_nystrom.detach().cpu().numpy())

        # Compute W
        with torch.no_grad():
            x_0 = x_batch_nystrom.repeat(x_batch_nystrom.shape[0],1)
            x_1 = x_batch_nystrom.repeat_interleave(x_batch_nystrom.shape[0], dim=0)
            W = model(x_0,x_1).to(torch.float32).reshape(x_batch_nystrom.shape[0],x_batch_nystrom.shape[0])

            # Compute C
            x_0 = train_dataset.tensors[0].repeat_interleave(x_batch_nystrom.shape[0], dim=0)
            x_1 = x_batch_nystrom.repeat(num_samples,1)
            C = model(x_0,x_1).to(torch.float32).reshape(num_samples, x_batch_nystrom.shape[0])

        epsilon = 1e-5
        W_inv = torch.inverse(W + epsilon * torch.eye(x_batch_nystrom.shape[0]))

        kernel_matrix_torch = C @ W_inv @ C.T
        kernel_matrix = kernel_matrix_torch.detach().numpy()

        svm = SVC(kernel = "precomputed").fit(kernel_matrix,y_train)
        accuracy_train_init = accuracy(svm, kernel_matrix, train_dataset.tensors[1].detach().numpy())

        with torch.no_grad():
            "Computing the initial testing kernel matrix"
            x_0 = test_dataset.tensors[0].repeat_interleave(x_batch_nystrom.shape[0],dim=0)
            x_1 = x_batch_nystrom.repeat(num_points_test, 1)
            C_test = model(x_0,x_1).to(torch.float32).reshape(num_points_test, x_batch_nystrom.shape[0])

        kernel_matrix_test = C_test @ W_inv @ C.T
        kernel_matrix_test = kernel_matrix_test.detach().numpy()

        accuracy_test_init = accuracy(svm, kernel_matrix_test, test_dataset.tensors[1].detach().numpy())

        "Computing the training KTA of this initial kernel"
        kta_init_training = costs[cost_function](kernel_matrix_torch,train_dataset.tensors[1]).item()

        training_predictions_initial = svm.predict(kernel_matrix)

        testing_predictions_initial = svm.predict(kernel_matrix_test)
    
    else:
        x_batch_nystrom_np = []

        "Computing the initial kernel matrix"
        with torch.no_grad():
            x_0 = train_dataset.tensors[0].repeat(num_samples,1)
            x_1 = train_dataset.tensors[0].repeat_interleave(num_samples, dim=0)
            results = model(x_0,x_1).to(torch.float32)
            kernel_matrix = results.reshape(num_samples, num_samples).detach().numpy()

        "Fitting an SVM to the original kernel matrix"
        svm = SVC(kernel = "precomputed").fit(kernel_matrix,y_train)

        "Computing the accuracy of this initial kernel"
        accuracy_train_init = accuracy(svm, kernel_matrix, train_dataset.tensors[1].detach().numpy())

        with torch.no_grad():
            "Computing the initial testing kernel matrix"
            x_0 = test_dataset.tensors[0].repeat_interleave(num_samples,dim=0)
            x_1 = train_dataset.tensors[0].repeat(num_points_test, 1)
            results_test = model(x_0,x_1).to(torch.float32)
            kernel_matrix_test = results_test.reshape(num_points_test, num_samples).detach().numpy()

        accuracy_test_init = accuracy(svm, kernel_matrix_test, test_dataset.tensors[1].detach().numpy())

        "Computing the training KTA of this initial kernel"
        kta_init_training = costs[cost_function](results.reshape(num_samples,num_samples),train_dataset.tensors[1]).item()

        training_predictions_initial = svm.predict(kernel_matrix)

        testing_predictions_initial = svm.predict(kernel_matrix_test)

    best_kernel_training = kernel_matrix
    best_kernel_testing = kernel_matrix_test
    best_kta_training = kta_init_training
    kta_training_epochs = []
    accuracies_training = []
    accuracies_testing = []
    total_circuit_executions_training = 0

    idx = 0
    while (idx < num_epochs):
        model.train()

        if config["use_nystrom_training"]:
            x_0 = x_batch_nystrom.repeat(x_batch_nystrom.shape[0],1)
            x_1 = x_batch_nystrom.repeat_interleave(x_batch_nystrom.shape[0], dim=0)
            W = model(x_0,x_1).to(torch.float32).reshape(x_batch_nystrom.shape[0],x_batch_nystrom.shape[0])

            total_circuit_executions_training += x_batch_nystrom.shape[0] ** 2

            x_0 = train_dataset.tensors[0].repeat_interleave(x_batch_nystrom.shape[0], dim=0)
            x_1 = x_batch_nystrom.repeat(num_samples,1)
            C = model(x_0,x_1).to(torch.float32).reshape(num_samples, x_batch_nystrom.shape[0])

            total_circuit_executions_training += num_samples * x_batch_nystrom.shape[0]

            epsilon = 1e-5
            W_inv = torch.inverse(W + epsilon * torch.eye(x_batch_nystrom.shape[0]))
            kernel_matrix_torch = C @ W_inv @ C.T

            opt.zero_grad()
            idx += 1

            loss = - costs[cost_function](kernel_matrix_torch,torch.Tensor(y_train))
            loss.backward()
            opt.step()

        else:
            x_batches, y_batches = [], []
            losses = []
            if full_kta:
                x_batch,y_batch = torch.Tensor(x_train), torch.Tensor(y_train)
            else:
                x_batch, y_batch = random.choice(list(train_loader))

                idx +=1
                opt.zero_grad()

                x_0 = x_batch.repeat(x_batch.shape[0],1)
                x_1 = x_batch.repeat_interleave(x_batch.shape[0], dim=0)

                output = model(x_0,x_1).to(torch.float32)
                total_circuit_executions_training += x_0.shape[0] ** 2

                loss = - costs[cost_function](output.reshape(x_batch.shape[0],x_batch.shape[0]),y_batch)
                loss.backward()
                opt.step()

        if idx % validate_every_epoch == 0:

            if use_nystrom_approx:
                with torch.no_grad():
                    # Compute W
                    x_0 = x_batch_nystrom.repeat(x_batch_nystrom.shape[0],1)
                    x_1 = x_batch_nystrom.repeat_interleave(x_batch_nystrom.shape[0], dim=0)
                    W = model(x_0,x_1).to(torch.float32).reshape(x_batch_nystrom.shape[0],x_batch_nystrom.shape[0])

                    # Compute C
                    x_0 = train_dataset.tensors[0].repeat_interleave(x_batch_nystrom.shape[0], dim=0)
                    x_1 = x_batch_nystrom.repeat(num_samples,1)
                    C = model(x_0,x_1).to(torch.float32).reshape(num_samples, x_batch_nystrom.shape[0])

                epsilon = 1e-5
                W_inv = torch.inverse(W + epsilon * torch.eye(x_batch_nystrom.shape[0]))

                kernel_matrix_torch = C @ W_inv @ C.T
                kernel_matrix_training = kernel_matrix_torch.detach().numpy()

                svm = SVC(kernel = "precomputed").fit(kernel_matrix_training,y_train)
                accuracies_training.append(accuracy(svm, kernel_matrix_training, train_dataset.tensors[1].detach().numpy()))

                "Computing the best testing kernel matrix"
                with torch.no_grad():
                    x_0 = test_dataset.tensors[0].repeat_interleave(x_batch_nystrom.shape[0],dim=0)
                    x_1 = x_batch_nystrom.repeat(num_points_test, 1)
                    C_test = model(x_0,x_1).to(torch.float32).reshape(num_points_test, x_batch_nystrom.shape[0])
                kernel_matrix_test = C_test @ W_inv @ C.T
                kernel_matrix_test = kernel_matrix_test.detach().numpy()

                accuracies_testing.append(accuracy(svm, kernel_matrix_test, test_dataset.tensors[1].detach().numpy()))

                "Computing the training KTA of this initial kernel"
                kta_epoch_training = costs[cost_function](kernel_matrix_torch,train_dataset.tensors[1]).item()
                kta_training_epochs.append(kta_epoch_training)
            
            else:

                # Calculate kernel target alignment full
                "Computing the training kernel matrix"
                with torch.no_grad():
                    x_0 = train_dataset.tensors[0].repeat(num_samples,1)
                    x_1 = train_dataset.tensors[0].repeat_interleave(num_samples, dim=0)
                    results = model(x_0,x_1).to(torch.float32)
                kernel_matrix_training = results.reshape(num_samples, num_samples).detach().numpy()
                kta_epoch_training = costs[cost_function](results.reshape(num_samples,num_samples),train_dataset.tensors[1]).item()
                kta_training_epochs.append(kta_epoch_training)

                "Compute the testing kernel matrix"
                with torch.no_grad():
                    x_0 = test_dataset.tensors[0].repeat_interleave(num_samples,dim=0)
                    x_1 = train_dataset.tensors[0].repeat(num_points_test, 1)
                    results_test = model(x_0,x_1).to(torch.float32)
                kernel_matrix_test = results_test.reshape(num_points_test, num_samples).detach().numpy()

                "Fitting an SVM to the kernel matrix"
                svm = SVC(kernel = "precomputed").fit(kernel_matrix_training,y_train)

                "Computing the accuracy of this kernel"
                accuracy_train = accuracy(svm, kernel_matrix_training, train_dataset.tensors[1].detach().numpy())

                accuracy_test = accuracy(svm, best_kernel_testing, test_dataset.tensors[1].detach().numpy())

                accuracies_training.append(accuracy_train)
                accuracies_testing.append(accuracy_test)

            if kta_epoch_training > best_kta_training:
                best_kta_training = kta_epoch_training
                best_kernel_training = kernel_matrix_training
                best_kernel_testing = kernel_matrix_test
                best_model_weights = copy.deepcopy(model.state_dict())

    if use_nystrom_approx:
        total_circuit_executions_kernel_matrix = x_batch_nystrom.shape[0] * x_batch_nystrom.shape[0] + num_samples * x_batch_nystrom.shape[0]
        total_circuit_executions_inference = num_points_test * x_batch_nystrom.shape[0]
    else:
        total_circuit_executions_kernel_matrix = num_samples ** 2
        total_circuit_executions_inference = num_points_test * num_samples

    training_predictions_best = svm.predict(best_kernel_training)
    testing_predictions_best = svm.predict(best_kernel_testing)

    "Fitting an SVM to the best kernel matrix"
    svm = SVC(kernel = "precomputed").fit(best_kernel_training,y_train)

    "Computing the accuracy of this kernel"
    accuracy_train_final = accuracy(svm, best_kernel_training, train_dataset.tensors[1].detach().numpy())
    accuracy_test_final = accuracy(svm,best_kernel_testing, test_dataset.tensors[1].detach().numpy())

    best_weights_numpy = {name: param.detach().cpu().numpy() for name, param in best_model_weights.items()}


    metrics = {
        "accuracy_train_init": accuracy_train_init,
        "accuracy_test_init": accuracy_test_init,
        "alignment_train_init": kta_init_training,
        "accuracy_train_final": accuracy_train_final,
        "accuracy_test_final": accuracy_test_final,
        "training_predictions_best": training_predictions_best,
        "testing_predictions_best": testing_predictions_best,
        "best_alignment_training": best_kta_training,
        "alignment_train_epochs": kta_training_epochs,
        "train_accuracies": accuracies_training,
        "test_accuracies": accuracies_testing,
        "circuit_executions": total_circuit_executions_training,
        "training_predictions_initial": training_predictions_initial,
        "testing_predictions_initial": testing_predictions_initial,
        "total_circuit_executions_kernel_matrix": total_circuit_executions_kernel_matrix,
        "total_circuit_executions_inference": total_circuit_executions_inference,
        "initial_model_weights": initial_weights_numpy,
        "best_model_weights": best_weights_numpy,
        "batch_nystrom": x_batch_nystrom_np
    }

    ray.train.report(metrics = metrics)
    





    




    
