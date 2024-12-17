import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
import csv


class Plotter():
    """
    This class will be used to plot the results from the runs.
    """

    def __init__(self,run_path):
        self.run_path = run_path

    def load_run(self, gradients = True):
        all_dirs = os.listdir(self.run_path)
        self.runs = [dir for dir in all_dirs if dir.startswith("run") or dir.startswith("get_gradient") or dir.startswith("kernel")]

        if gradients:
            results = []
            for run in self.runs:
                aux = []
                result_file_path = os.path.join(self.run_path, run, 'result.json')
                with open(result_file_path, 'r') as file:
                    for line in file:
                        try:
                            aux.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON: {e}")
                    results.append(aux)
        else:
            results = []
            for run in self.runs:
                result_file_path = os.path.join(self.run_path, run, 'result.json')
                if os.path.isfile(result_file_path):
                    with open(result_file_path, 'r') as file:
                        results.append(json.load(file))
        
        params = []
        for run in self.runs:
            params_file_path = os.path.join(self.run_path, run, 'params.json')
            if os.path.isfile(params_file_path):
                with open(params_file_path, 'r') as file:
                    params.append(json.load(file))
        
        self.results = results
        self.params = params

    def performance_plots(self,params,save_path, figure_name,labels):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        results_plot = []
        for i, param in enumerate(params):
            aux = []
            for j,run in enumerate(self.runs):
                if param in run:
                    aux.append(self.results[j])
            results_plot.append(aux)
        
        # Plot KTA over epochs
        results_gathered_kta = []
        results_gathered_train = []
        results_gathered_test = []
        for result in results_plot:
            aux_kta = []
            aux_train = []
            aux_test = []
            for sub_result in result:
                aux_kta.append(sub_result["alignment_train_epochs"])
                aux_train.append(sub_result["train_accuracies"])
                aux_test.append(sub_result["test_accuracies"])
            results_gathered_kta.append(np.array(aux_kta))
            results_gathered_train.append(np.array(aux_train))
            results_gathered_test.append(np.array(aux_test))
        
        std_gathered_kta = []
        std_gathered_train = []
        std_gathered_test = []
        for i in range(len(results_gathered_kta)):
            aux_kta = np.std(results_gathered_kta[i], axis = 0)
            aux_train = np.std(results_gathered_train[i], axis = 0)
            aux_test = np.std(results_gathered_test[i], axis = 0)
            results_gathered_kta[i] = np.mean(results_gathered_kta[i],axis = 0)
            results_gathered_train[i] = np.mean(results_gathered_train[i],axis = 0)
            results_gathered_test[i] = np.mean(results_gathered_test[i],axis = 0)
            std_gathered_kta.append(aux_kta)
            std_gathered_train.append(aux_train)
            std_gathered_test.append(aux_test)
            
        results_gathered_kta = np.array(results_gathered_kta)
        results_gathered_train = np.array(results_gathered_train)
        results_gathered_test = np.array(results_gathered_test)
        std_gathered_kta = np.array(std_gathered_kta)
        std_gathered_train = np.array(std_gathered_train)
        std_gathered_test = np.array(std_gathered_test)

        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(131)
        for i in range(len(results_gathered_kta)):
            lower_kta = np.clip(results_gathered_kta[i] - std_gathered_kta[i],0,1)
            upper_kta = np.clip(results_gathered_kta[i] + std_gathered_kta[i],0,1)
            ax1.plot(results_gathered_kta[i], label = labels[i])
            ax1.fill_between(range(len(results_gathered_kta[i])),lower_kta, upper_kta, alpha = 0.3)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("KTA")
        
        ax2 = fig.add_subplot(132)
        for i in range(len(results_gathered_train)):
            lower_train = np.clip(results_gathered_train[i] - std_gathered_train[i],0,1)
            upper_train = np.clip(results_gathered_train[i] + std_gathered_train[i],0,1)
            ax2.plot(results_gathered_train[i], label = labels[i])
            ax2.fill_between(range(len(results_gathered_train[i])),lower_train, upper_train, alpha = 0.3)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Train Accuracies")
        
        ax3 = fig.add_subplot(133, sharey=ax2)
        for i in range(len(results_gathered_test)):
            lower_test = np.clip(results_gathered_test[i] - std_gathered_test[i],0,1)
            upper_test = np.clip(results_gathered_test[i] + std_gathered_test[i],0,1)
            ax3.plot(results_gathered_test[i], label = labels[i])
            ax3.fill_between(range(len(results_gathered_test[i])),lower_test, upper_test, alpha = 0.3)
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Test Accuracies")
            ax3.legend()
        
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f"{figure_name}.png"), dpi = 300)

    def plot_noise(self,params,save_path, figure_name,labels):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        fig = plt.figure(figsize=(15,5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, sharey=ax2)
        
        for k,param in enumerate(params):
            results_plot = []
            for i, p in enumerate(param):
                aux = []
                for j,run in enumerate(self.runs):
                    if p in run:
                        aux.append(self.results[j])
                results_plot.append(aux)

            # Plot KTA over epochs
            results_gathered_kta = []
            results_gathered_train = []
            results_gathered_test = []
            for result in results_plot:
                aux_kta = []
                aux_train = []
                aux_test = []
                for sub_result in result:
                    aux_kta.append(sub_result["best_alignment_training"])
                    aux_train.append(sub_result["accuracy_train_final"])
                    aux_test.append(sub_result["accuracy_test_final"])
                results_gathered_kta.append(np.array(aux_kta))
                results_gathered_train.append(np.array(aux_train))
                results_gathered_test.append(np.array(aux_test))

            std_gathered_kta = []
            std_gathered_train = []
            std_gathered_test = []
            for i in range(len(results_gathered_kta)):
                aux_kta = np.std(results_gathered_kta[i], axis = 0)
                aux_train = np.std(results_gathered_train[i], axis = 0)
                aux_test = np.std(results_gathered_test[i], axis = 0)
                results_gathered_kta[i] = np.mean(results_gathered_kta[i],axis = 0)
                results_gathered_train[i] = np.mean(results_gathered_train[i],axis = 0)
                results_gathered_test[i] = np.mean(results_gathered_test[i],axis = 0)
                std_gathered_kta.append(aux_kta)
                std_gathered_train.append(aux_train)
                std_gathered_test.append(aux_test)

            results_gathered_kta = np.array(results_gathered_kta)
            results_gathered_train = np.array(results_gathered_train)
            results_gathered_test = np.array(results_gathered_test)
            std_gathered_kta = np.array(std_gathered_kta)
            std_gathered_train = np.array(std_gathered_train)
            std_gathered_test = np.array(std_gathered_test)

            x = np.array([0,0.1,0.2,0.5])

            #ax1.plot(x, results_gathered_kta, marker = "o", label = labels[k])
            ax1.errorbar(x, results_gathered_kta, yerr=std_gathered_kta,fmt="o-", capsize=5, label = labels[k])
            ax1.set_xlabel("$\delta$")
            ax1.set_ylabel("$KTA$")
            ax1.legend()

            #ax2.plot(x, results_gathered_train, marker = "o", label = labels[k])
            ax2.errorbar(x, results_gathered_train, yerr=std_gathered_train,fmt="o-", capsize=5, label = labels[k])
            ax2.set_xlabel("$\delta$")
            ax2.set_ylabel("Train Accuracy")

            #ax3.plot(x, results_gathered_test, marker = "o", label = labels[k])
            ax3.errorbar(x, results_gathered_test, yerr=std_gathered_test,fmt="o-", capsize=5, label = labels[k])
            ax3.set_xlabel("$\delta$")
            ax3.set_ylabel("Test Accuracy")
        
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f"{figure_name}.png"), dpi = 300)


    def gradient_plots(self,params,save_path, figure_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        results_plot = []
        for i, param in enumerate(params):
            aux = []
            for j,run in enumerate(self.runs):
                if param in run:
                    aux.append(self.results[j])
            results_plot.append(aux)
        
        # Get initial and final KTA, train_acc, test_acc
        results_gathered_input_scaling = []
        results_gathered_variational = []
        for result in results_plot:
            aux_input_scaling = []
            aux_variational = []
            for sub_result in result:
                for sample in sub_result:
                    aux_input_scaling.append([sample["grads"]["input_scaling"]])
                    aux_variational.append([sample["grads"]["variational"]])
            results_gathered_input_scaling.append(np.array(aux_input_scaling))
            results_gathered_variational.append(np.array(aux_variational))
        
        vars_variational = []
        vars_input_scaling = []
        for i in range(len(results_gathered_input_scaling)):
            run_input_scaling = np.array(results_gathered_input_scaling[i])
            run_variational = np.array(results_gathered_variational[i])
            run_input_scaling = run_input_scaling.reshape(run_input_scaling.shape[0],run_input_scaling.shape[2])
            run_variational = run_variational.reshape(run_variational.shape[0],run_variational.shape[2])

            mean_input_scaling = np.mean(run_input_scaling, axis = 1)
            mean_variational = np.mean(run_variational, axis = 1)

            var_input_scaling = np.var(mean_input_scaling)
            var_variational = np.var(mean_variational)

            vars_variational.append(var_variational)
            vars_input_scaling.append(var_input_scaling)

        fig, axs = plt.subplots(1,2,figsize=(8,4), tight_layout=True)
        axs[0].scatter([2,4,6,8], vars_input_scaling)
        axs[0].set_xlabel("Number of Qubits")
        axs[0].set_ylabel("Variance of Gradient")
        axs[0].set_yscale("log")
        axs[0].set_title("Input Scaling")

        axs[1].scatter([2,4,6,8], vars_variational)
        axs[1].set_xlabel("Number of Qubits")
        axs[1].set_ylabel("Variance of Gradient")
        axs[1].set_yscale("log")
        axs[1].set_title("Variational")

        plt.savefig(os.path.join(save_path, f"{figure_name}.png"), dpi = 300)

    def bar_plots(self,params,save_path, figure_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        results_plot = []
        for i, param in enumerate(params):
            aux = []
            for j,run in enumerate(self.runs):
                if param in run:
                    aux.append(self.results[j])
            results_plot.append(aux)
        
        # Get initial and final KTA, train_acc, test_acc
        results_gathered_circuit_executions_training = []
        results_gathered_circuit_executions_kernel = []
        results_gathered_circuit_executions_inference = []
        for result in results_plot:
            aux_train = []
            aux_kernel = []
            aux_inference = []
            for sub_result in result:
                aux_train.append([sub_result["circuit_executions"]])
                aux_kernel.append([sub_result["total_circuit_executions_kernel_matrix"]])
                aux_inference.append([sub_result["total_circuit_executions_inference"]])
            results_gathered_circuit_executions_training.append(np.array(aux_train))
            results_gathered_circuit_executions_kernel.append(np.array(aux_kernel))
            results_gathered_circuit_executions_inference.append(np.array(aux_inference))
            
        results_gathered_circuit_executions_training = np.array(results_gathered_circuit_executions_training)
        results_gathered_circuit_executions_kernel = np.array(results_gathered_circuit_executions_kernel)
        results_gathered_circuit_executions_inference = np.array(results_gathered_circuit_executions_inference)

        results_gathered_circuit_executions_training = results_gathered_circuit_executions_training.reshape(results_gathered_circuit_executions_training.shape[0], results_gathered_circuit_executions_training.shape[1])
        results_gathered_circuit_executions_kernel = results_gathered_circuit_executions_kernel.reshape(results_gathered_circuit_executions_training.shape[0], results_gathered_circuit_executions_training.shape[1])
        results_gathered_circuit_executions_inference = results_gathered_circuit_executions_inference.reshape(results_gathered_circuit_executions_training.shape[0], results_gathered_circuit_executions_training.shape[1])

        results_gathered_circuit_executions_training = np.mean(results_gathered_circuit_executions_training,axis=1)
        results_gathered_circuit_executions_kernel = np.mean(results_gathered_circuit_executions_kernel,axis=1)
        results_gathered_circuit_executions_inference = np.mean(results_gathered_circuit_executions_inference,axis=1)

        data = [results_gathered_circuit_executions_training, results_gathered_circuit_executions_kernel, results_gathered_circuit_executions_inference]
        num_subcategories = len(data[0])
        category_labels = ["Training", "Kernel Matrix", "Inference"]
        colors = plt.cm.viridis(np.linspace(0, 1, num_subcategories))
        x = np.arange(len(data))
        bar_width = 0.8 / num_subcategories

        fig,ax = plt.subplots()
        for i in range(num_subcategories):
            ax.bar(x + i * bar_width, [d[i] for d in data], bar_width, label = params[i], color = colors[i])
        
        ax.set_ylabel("Circuit Executions")
        ax.set_xticks(x + bar_width * (num_subcategories - 1) / 2)
        ax.set_xticklabels(category_labels)
        ax.legend()
        plt.savefig(os.path.join(save_path, f"{figure_name}_bar.png"), dpi = 300)

            
    def performance_plots_static(self,params,save_path, figure_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        results_plot = []
        for i, param in enumerate(params):
            aux = []
            for j,run in enumerate(self.runs):
                if param in run:
                    aux.append(self.results[j])
            results_plot.append(aux)
        
        # Get initial and final KTA, train_acc, test_acc
        results_gathered_kta = []
        results_gathered_train = []
        results_gathered_test = []
        for result in results_plot:
            aux_kta = []
            aux_train = []
            aux_test = []
            for sub_result in result:
                aux_kta.append([sub_result["alignment_train_init"],sub_result["best_alignment_training"]])
                aux_train.append([sub_result["accuracy_train_init"],sub_result["accuracy_train_final"]])
                aux_test.append([sub_result["accuracy_test_init"],sub_result["accuracy_test_final"]])
            results_gathered_kta.append(np.array(aux_kta))
            results_gathered_train.append(np.array(aux_train))
            results_gathered_test.append(np.array(aux_test))
        
        std_gathered_kta = []
        std_gathered_train = []
        std_gathered_test = []
        for i in range(len(results_gathered_kta)):
            aux_kta = np.std(results_gathered_kta[i], axis = 0)
            aux_train = np.std(results_gathered_train[i], axis = 0)
            aux_test = np.std(results_gathered_test[i], axis = 0)
            results_gathered_kta[i] = np.mean(results_gathered_kta[i],axis = 0)
            results_gathered_train[i] = np.mean(results_gathered_train[i],axis = 0)
            results_gathered_test[i] = np.mean(results_gathered_test[i],axis = 0)
            std_gathered_kta.append(aux_kta)
            std_gathered_train.append(aux_train)
            std_gathered_test.append(aux_test)
            
        results_gathered_kta = np.array(results_gathered_kta)
        results_gathered_train = np.array(results_gathered_train)
        results_gathered_test = np.array(results_gathered_test)
        std_gathered_kta = np.array(std_gathered_kta)
        std_gathered_train = np.array(std_gathered_train)
        std_gathered_test = np.array(std_gathered_test)

        # Number of metrics
        num_metrics = len(results_gathered_train[0])
        num_bars = len(results_gathered_train)

        # Bar width
        bar_width = 0.8 / num_bars

        # Positions of the bars
        positions = [np.arange(num_metrics) + i * bar_width for i in range(num_bars)]

        for i in range(len(results_gathered_kta)):
            lower_kta = np.array(results_gathered_kta[i]) - np.array(std_gathered_kta[i])
            upper_kta = np.array(results_gathered_kta[i]) + np.array(std_gathered_kta[i])
            labels = ["Initial KTA", "Final KTA"]
            plt.bar(positions[i], results_gathered_kta[i], width=bar_width, label=params[i])

        # Add labels
        plt.xlabel('Metrics', fontweight='bold')
        plt.xticks([r + bar_width * (num_bars - 1) / 2 for r in range(num_metrics)], ["Before Alignment", "After Alignment"])
        plt.ylabel("KTA")
        plt.legend()
        
        plt.savefig(os.path.join(save_path, f"{figure_name}_kta.png"), dpi = 300)
        plt.clf()

        for i in range(len(results_gathered_train)):
            lower_train = np.array(results_gathered_train[i]) - np.array(std_gathered_train[i])
            upper_train = np.array(results_gathered_train[i]) + np.array(std_gathered_train[i])
            labels = ["Initial Training Accuracy", "Final Training Accuracy"]
            plt.bar(positions[i], results_gathered_train[i], width=bar_width, label=params[i])

        # Add labels
        plt.xlabel('Metrics', fontweight='bold')
        plt.xticks([r + bar_width * (num_bars - 1) / 2 for r in range(num_metrics)], ["Before Alignment", "After Alignment"])
        plt.ylabel("Train Accuracies")
        plt.legend()
        
        plt.savefig(os.path.join(save_path, f"{figure_name}_train.png"), dpi = 300)
        plt.clf()

        for i in range(len(results_gathered_test)):
            lower_test = np.array(results_gathered_test[i]) - np.array(std_gathered_test[i])
            upper_test = np.array(results_gathered_test[i]) + np.array(std_gathered_test[i])
            labels = ["Initial Testing Accuracy", "Final Testing Accuracy"]
            plt.bar(positions[i], results_gathered_test[i], width=bar_width, label=params[i])

        # Add labels
        plt.xlabel('Metrics', fontweight='bold')
        plt.xticks([r + bar_width * (num_bars - 1) / 2 for r in range(num_metrics)], ["Before Alignment", "After Alignment"])
        plt.ylabel("Test Accuracies")
        plt.legend()
        
        plt.savefig(os.path.join(save_path, f"{figure_name}_test.png"), dpi = 300)
        plt.clf()

    def performance_plots_csv(self,params,save_path, figure_name):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        results_plot = []
        for i, param in enumerate(params):
            aux = []
            for j,run in enumerate(self.runs):
                if param in run:
                    aux.append(self.results[j])
            results_plot.append(aux)
        
        # Get initial and final KTA, train_acc, test_acc
        results_gathered_kta_init = []
        results_gathered_train_init = []
        results_gathered_test_init = []
        results_gathered_kta_final = []
        results_gathered_train_final = []
        results_gathered_test_final = []
        for result in results_plot:
            aux_kta_init = []
            aux_train_init = []
            aux_test_init = []
            aux_kta_final = []
            aux_train_final = []
            aux_test_final = []
            for sub_result in result:
                aux_kta_init.append(sub_result["alignment_train_init"])
                aux_train_init.append(sub_result["accuracy_train_init"])
                aux_test_init.append(sub_result["accuracy_test_init"])
                aux_kta_final.append(sub_result["best_alignment_training"])
                aux_train_final.append(sub_result["accuracy_train_final"])
                aux_test_final.append(sub_result["accuracy_test_final"])
            results_gathered_kta_init.append(np.array(aux_kta_init))
            results_gathered_train_init.append(np.array(aux_train_init))
            results_gathered_test_init.append(np.array(aux_test_init))
            results_gathered_kta_final.append(np.array(aux_kta_final))
            results_gathered_train_final.append(np.array(aux_train_final))
            results_gathered_test_final.append(np.array(aux_test_final))
            
        results_gathered_kta_init = np.array(results_gathered_kta_init)
        results_gathered_train_init = np.array(results_gathered_train_init)
        results_gathered_test_init = np.array(results_gathered_test_init)
        results_gathered_kta_final = np.array(results_gathered_kta_final)
        results_gathered_train_final = np.array(results_gathered_train_final)
        results_gathered_test_final = np.array(results_gathered_test_final)
        
        min_train_init = np.min(results_gathered_train_init, axis = 1)
        max_train_init = np.max(results_gathered_train_init, axis = 1)
        mean_train_init = np.mean(results_gathered_train_init, axis = 1)
        std_train_init = np.std(results_gathered_train_init, axis = 1)
        min_test_init = np.min(results_gathered_test_init, axis = 1)
        max_test_init = np.max(results_gathered_test_init, axis = 1)
        mean_test_init = np.mean(results_gathered_test_init, axis = 1)
        std_test_init = np.std(results_gathered_test_init, axis = 1)
        min_kta_init = np.min(results_gathered_kta_init, axis = 1)
        max_kta_init = np.max(results_gathered_kta_init, axis = 1)
        mean_kta_init = np.mean(results_gathered_kta_init, axis = 1)
        std_kta_init = np.std(results_gathered_kta_init, axis = 1)

        min_train_final = np.min(results_gathered_train_final, axis = 1)
        max_train_final = np.max(results_gathered_train_final, axis = 1)
        mean_train_final = np.mean(results_gathered_train_final, axis = 1)
        std_train_final = np.std(results_gathered_train_final, axis = 1)
        min_test_final = np.min(results_gathered_test_final, axis = 1)
        max_test_final = np.max(results_gathered_test_final, axis = 1)
        mean_test_final = np.mean(results_gathered_test_final, axis = 1)
        std_test_final = np.std(results_gathered_test_final, axis = 1)
        min_kta_final = np.min(results_gathered_kta_final, axis = 1)
        max_kta_final = np.max(results_gathered_kta_final, axis = 1)
        mean_kta_final = np.mean(results_gathered_kta_final, axis = 1)
        std_kta_final = np.std(results_gathered_kta_final, axis = 1)

        # Write to CSV
        with open(os.path.join(save_path, f"{figure_name}.csv"), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['min_train_init', 'max_train_init', 'mean_train_init', 'std_train_init',
                             'min_test_init', 'max_test_init', 'mean_test_init', 'std_test_init',
                             'min_kta_init', 'max_kta_init', 'mean_kta_init', 'std_kta_init',
                             'min_train_final', 'max_train_final', 'mean_train_final', 'std_train_final',
                             'min_test_final', 'max_test_final', 'mean_test_final', 'std_test_final',
                             'min_kta_final', 'max_kta_final', 'mean_kta_final', 'std_kta_final'])

            for i in range(len(min_train_init)):
                writer.writerow([min_train_init[i], max_train_init[i], mean_train_init[i], std_train_init[i],
                                 min_test_init[i], max_test_init[i], mean_test_init[i], std_test_init[i],
                                 min_kta_init[i], max_kta_init[i], mean_kta_init[i], std_kta_init[i],
                                 min_train_final[i], max_train_final[i], mean_train_final[i], std_train_final[i],
                                 min_test_final[i], max_test_final[i], mean_test_final[i], std_test_final[i],
                                 min_kta_final[i], max_kta_final[i], mean_kta_final[i], std_kta_final[i]])

    
    def plot_boundary(self, save_path, figure_name, dataset, dataset_size, num_features, feature_scaling, initial = False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        results_plot = []
        for i, param in enumerate(params):
            aux = []
            for j,run in enumerate(self.runs):
                if param in run:
                    aux.append(self.results[j])
            results_plot.append(aux)
        
        # Plot KTA over epochs
        boundaries = []
        for result in results_plot:
            aux = []
            for sub_result in result:
                if initial:
                    aux.append(sub_result["boundary_predictions_initial"])
                else:
                    aux.append(sub_result["boundary_predictions_best"])
            boundaries.append(np.array(aux))

        xxx = np.linspace(-np.pi/2, np.pi/2, int(np.sqrt(len(aux[0]))))
        yyy = np.linspace(-np.pi/2, np.pi/2, int(np.sqrt(len(aux[0]))))
        X, Y = np.meshgrid(xxx, yyy)

        boundaries = boundaries[0][0]
        if initial:
            accuracy_train = results_plot[0][0]["accuracy_train_init"]
            accuracy_test = results_plot[0][0]["accuracy_test_init"]
        else:
            accuracy_train = results_plot[0][0]["accuracy_train_final"]
            accuracy_test = results_plot[0][0]["accuracy_test_final"]

        Z = boundaries.reshape(X.shape)

        if dataset == "make_classification":
            x,y = make_classification(n_samples = dataset_size,n_features = num_features,
                                  n_informative = num_features, n_redundant = 0)
            num_points_test = int(0.2 * len(x))
            dataset_size = dataset_size - num_points_test
        else:
            dataset_npy = np.load("/home/users/coelho/quantum_kernels/efficient_kta/datasets/final_datasets.npy", allow_pickle=True).item()
            x_train = dataset_npy[dataset][num_features][dataset_size]["x_train"]
            y_train = dataset_npy[dataset][num_features][dataset_size]["y_train"]
            x_test = dataset_npy[dataset][num_features][dataset_size]["x_test"]
            y_test = dataset_npy[dataset][num_features][dataset_size]["y_test"]

            num_points_test = x_test.shape[0]
            x = np.concatenate((x_train,x_test))
            y = np.concatenate((y_train,y_test))
        
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

        x_train_x = x_train[:,0]
        x_train_y = x_train[:,1]
        x_test_x = x_test[:,0]
        x_test_y = x_test[:,1]

        class_colors = {-1: "red", 1: "blue"}
        colors_train = [class_colors[class_] for class_ in y_train]
        colors_test = [class_colors[class_] for class_ in y_test]

        plt.scatter(x_train_x, x_train_y, c = colors_train, marker = "o", label = "Train")
        plt.scatter(x_test_x, x_test_y, marker = "o", facecolors="none", edgecolors=colors_test, label = "Test")
        plt.contourf(X,Y,Z, levels = [-1,0,1], colors = ["red", "blue"], alpha=0.3)
        plt.contour(X,Y,Z, levels=[0], colors = "black")
        plt.xlabel("X")
        plt.ylabel("Y")
        # Create custom legend handles
        train_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=6, label='Train')
        test_handle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=6, markerfacecolor='none', label='Test')
        plt.legend(handles=[train_handle, test_handle])
        plt.annotate(f"Train Accuracy: {accuracy_train}", xy = (1,0.9), bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        plt.annotate(f"Test Accuracy: {accuracy_test}", xy = (1,0.6),bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        plt.savefig(os.path.join(save_path, f"{figure_name}.png"), dpi = 300)


        
if __name__ == "__main__":
    plotter = Plotter("path_to_ray_results_folder")
    plotter.load_run(gradients = False)

    params_0 = [
        "=random",
        "=lowest",
        "=highest",
        "=prob_lowest",
        "=prob_highest",
        "=curriculum"
    ]

    #params_1 = [
    #    "num_landmarks=1,std=0.0",
    #    "num_landmarks=1,std=0.1",
    #    "num_landmarks=1,std=0.2",
    #    "num_landmarks=1,std=0.5"
    #]
#
    #params_2 = [
    #    "num_landmarks=2,std=0.0",
    #    "num_landmarks=2,std=0.1",
    #    "num_landmarks=2,std=0.2",
    #    "num_landmarks=2,std=0.5"
    #]
#
    #params_3 = [
    #    "num_landmarks=4,std=0.0",
    #    "num_landmarks=4,std=0.1",
    #    "num_landmarks=4,std=0.2",
    #    "num_landmarks=4,std=0.5"
    #]
#
    #params_4 = [
    #    "num_landmarks=8,std=0.0",
    #    "num_landmarks=8,std=0.1",
    #    "num_landmarks=8,std=0.2",
    #    "num_landmarks=8,std=0.5"
    #]


    labels_0 = [
        "Random",
        "Lowest",
        "Highest",
        "Lowest Soft",
        "Highest Soft",
        "Curriculum"
        ]


    labels = [
        labels_0
    ]

    params = [
        params_0
    ]

    figure_names = ["plot"]

    #plotter.plot_noise(params = params, save_path = "figures/10_12_2024/corners/curriculum/",figure_name = figure_names[0], labels = labels[0])

    for i in range(len(params)):
        plotter.performance_plots(params = params[i], save_path = "figures/12_12_2024/checkers/curriculum/",figure_name = figure_names[i], labels = labels[i])