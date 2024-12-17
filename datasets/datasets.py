import pennylane as qml
import numpy as np
from sklearn.utils import shuffle

#[tc] = qml.data.load("other", name="two-curves")
#[hm] = qml.data.load("other", name="hidden-manifold")
#[ls] = qml.data.load("other", name="linearly-separable")
#
#qubits = [2,4,6,8,10]
#
#datasets = {
#    "tc": {},
#    "hm": {},
#    "ls": {}
#}
#
#for num_qubits in qubits:
#    datasets["tc"][num_qubits] = {"train": {}, "test": {}}
#    datasets["tc"][num_qubits]["train"]["inputs"] = np.array(tc.train[f"{num_qubits}"]["inputs"])
#    datasets["tc"][num_qubits]["train"]["labels"] = np.array(tc.train[f"{num_qubits}"]["labels"])
#    datasets["tc"][num_qubits]["test"]["inputs"] = np.array(tc.test[f"{num_qubits}"]["inputs"])
#    datasets["tc"][num_qubits]["test"]["labels"] = np.array(tc.test[f"{num_qubits}"]["labels"])
#
#    datasets["ls"][num_qubits] = {"train": {}, "test": {}}
#    datasets["ls"][num_qubits]["train"]["inputs"] = np.array(ls.train[f"{num_qubits}"]["inputs"])
#    datasets["ls"][num_qubits]["train"]["labels"] = np.array(ls.train[f"{num_qubits}"]["labels"])
#    datasets["ls"][num_qubits]["test"]["inputs"] = np.array(ls.test[f"{num_qubits}"]["inputs"])
#    datasets["ls"][num_qubits]["test"]["labels"] = np.array(ls.test[f"{num_qubits}"]["labels"])
#
#    datasets["hm"][num_qubits] = {"train": {}, "test": {}}
#    datasets["hm"][num_qubits]["train"]["inputs"] = np.array(hm.diff_train[f"{num_qubits}"]["inputs"])
#    datasets["hm"][num_qubits]["train"]["labels"] = np.array(hm.diff_train[f"{num_qubits}"]["labels"])
#    datasets["hm"][num_qubits]["test"]["inputs"] = np.array(hm.diff_test[f"{num_qubits}"]["inputs"])
#    datasets["hm"][num_qubits]["test"]["labels"] = np.array(hm.diff_test[f"{num_qubits}"]["labels"])
#
#
#np.save("datasets/variable_features_datasets.npy", datasets)


final_dataset = {
    "tc": {},
    "hm": {},
    "ls": {}
}

datasets = ["ls", "tc", "hm"]
features = [2,4,6,8,10]
test_ratio = 0.25
dataset_sizes = [50,100,150,200]

for dataset in datasets:
    for num_features in features:
        final_dataset[dataset][num_features] = {}
        dataset_npy = np.load("/home/users/coelho/quantum_kernels/efficient_kta/datasets/variable_features_datasets.npy", allow_pickle=True).item()
        x_train = dataset_npy[dataset][num_features]["train"]["inputs"]
        y_train = dataset_npy[dataset][num_features]["train"]["labels"]
        x_test = dataset_npy[dataset][num_features]["test"]["inputs"]
        y_test = dataset_npy[dataset][num_features]["test"]["labels"]

        x = np.concatenate((x_train,x_test))
        y = np.concatenate((y_train, y_test))

        for size in dataset_sizes:
            final_dataset[dataset][num_features][size] = {}
            new_x, new_y = shuffle(x,y)
            num_test_samples = int(size * test_ratio)
            x_train, y_train = new_x[:size], new_y[:size]
            x_test, y_test = new_x[size:size+num_test_samples], new_y[size:size+num_test_samples]
    
            final_dataset[dataset][num_features][size]["x_train"] = x_train
            final_dataset[dataset][num_features][size]["x_test"] = x_test
            final_dataset[dataset][num_features][size]["y_train"] = y_train
            final_dataset[dataset][num_features][size]["y_test"] = y_test


np.save("datasets/final_datasets.npy", final_dataset)


