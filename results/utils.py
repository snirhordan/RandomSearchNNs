import os 
import numpy as np
import pickle as pkl

def get_results(path, data_name, model_name, total_ids, last_id=1, verbose=False): 
    val_results, test_results, hyperparameters = [], [], []

    for i in range(total_ids): 
        id_path = os.path.join(path, f'{data_name}_{model_name}_{i}.pkl')
        if os.path.exists(id_path): # we are looking at accuracy which we aim to maximize, so this is okay
            with open(id_path, 'rb') as file: 
                results = pkl.load(file)
                val_means, val_stds = np.mean(results['valid_aucs']), np.std(results['valid_aucs'])
                test_means, test_stds = np.mean(results['test_aucs']), np.std(results['test_aucs'])
                
                val_results.append((val_means, val_stds))
                test_results.append((test_means, test_stds))
                hyperparameters.append(results)

    if len(hyperparameters) > 0 : 
        best_id = np.argsort([np.mean(val_results[i][0]) for i in range(len(val_results))])[-last_id]
        best_val_mean, best_val_std = val_results[best_id][0],  val_results[best_id][1]
        best_test_mean, best_test_std = test_results[best_id][0], test_results[best_id][1]
        best_hyperparameters = hyperparameters[best_id]
        
        # if verbose: 
        #     print(f"{best_hyperparameters['data_name']} {best_hyperparameters['model_name']} LR {best_hyperparameters['lr']} NL {best_hyperparameters['num_layers']} " \
        #           f"Hid Dim {best_hyperparameters['h_dim']} m {best_hyperparameters['m']} Reduce {best_hyperparameters['reduce']} " \
        #           f"Test Performance: {best_test_mean:.3f} +/- {best_test_std:.3f}")

        if verbose: 
            print(f"{best_hyperparameters['data_name']} {best_hyperparameters['model_name']} LR {best_hyperparameters['lr']} NL {best_hyperparameters['num_layers']} " \
                  f"Hid Dim {best_hyperparameters['h_dim']} Reduce {best_hyperparameters['reduce']} Test Performance: {best_test_mean:.3f} +/ {best_test_std:.3f}")
    else: 
        best_test_mean = None
        best_test_std = None

    return best_test_mean, best_test_std

def get_results_m(path, data_name, model_name, total_ids, m, last_id=1, verbose=False): 
    val_results, test_results, hyperparameters = [], [], []

    for i in range(total_ids): 
        id_path = os.path.join(path, f'{data_name}_{model_name}_{i}.pkl')
        if os.path.exists(id_path): # we are looking at accuracy which we aim to maximize, so this is okay
            with open(id_path, 'rb') as file: 
                results = pkl.load(file)
                if results['m'] == m: 
                    val_means, val_stds = np.mean(results['valid_aucs']), np.std(results['valid_aucs'])
                    test_means, test_stds = np.mean(results['test_aucs']), np.std(results['test_aucs'])
                    
                    val_results.append((val_means, val_stds))
                    test_results.append((test_means, test_stds))
                    hyperparameters.append(results)

    if len(hyperparameters) > 0 : 
        best_id = np.argsort([np.mean(val_results[i][0]) for i in range(len(val_results))])[-last_id]
        best_val_mean, best_val_std = val_results[best_id][0],  val_results[best_id][1]
        best_test_mean, best_test_std = test_results[best_id][0], test_results[best_id][1]
        best_hyperparameters = hyperparameters[best_id]
        
        # if verbose: 
        #     print(f"{best_hyperparameters['data_name']} {best_hyperparameters['model_name']} LR {best_hyperparameters['lr']} NL {best_hyperparameters['num_layers']} " \ 
        #           f"Hid Dim {best_hyperparameters['h_dim']} m {best_hyperparameters['m']} Reduce {best_hyperparameters['reduce']} " \
        #           f"Test Performance: {best_test_mean:.3f} +/- {best_test_std:.3f}")

        if verbose: 
            print(f"{best_hyperparameters['data_name']} {best_hyperparameters['model_name']} LR {best_hyperparameters['lr']} NL {best_hyperparameters['num_layers']} " \
                  f"Hid Dim {best_hyperparameters['h_dim']} Reduce {best_hyperparameters['reduce']} Test Performance: {best_test_mean:.3f} +/ {best_test_std:.3f}")
            
    else: 
        best_test_mean = None
        best_test_std = None

    return best_test_mean, best_test_std


