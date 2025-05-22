import numpy as np
import subprocess
import argparse
import os
import time
import shutil

def get_lrs(base, mult):
    lrs = []
    for b in base: 
        for m in mult: 
            lrs.append(b * m)
    return lrs

def format_command(command, hyperparameters, device_idx, n_splits, model_idx, model_name, data_name, save_path): 
    args = [command]
    for key in hyperparameters.keys(): 
        args.append(f' --{key} {hyperparameters[key]}')
        
    args.append(f' --device_idx {device_idx}')
    args.append(f' --n_splits {n_splits}')
    args.append(f' --model_idx {model_idx}')
    args.append(f' --model_name {model_name}')
    args.append(f' --data_name {data_name}')
    args.append(f' --save_path {save_path}')
    
    string = ''
    for arg in args: 
        string += arg
        
    return string

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    # hyperparameters
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--lrs_b', type=float, nargs='+')
    parser.add_argument('--lrs_m', type=int, nargs='+')
    parser.add_argument('--dropouts', type=float, nargs='+')
    parser.add_argument('--h_dims', type=int, nargs='+')
    parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[5])
    parser.add_argument('--num_layers', type=int, nargs='+', default=[2])
    parser.add_argument('--ms', type=int, nargs='+', default=[4])
    parser.add_argument('--ls', type=int, nargs='+', default=[25])
    parser.add_argument('--ws', type=int, nargs='+', default=[8])
    parser.add_argument('--nb', type=bool, default=True)
    parser.add_argument('--reduce', type=str, nargs='+', default=['max'])
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[64])
    parser.add_argument('--n_splits', type=int, default=10)

    # dataset and search parameters
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--command', type=str, default='python3 main_train_benchmarks_fps.py')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('--start_idx', type=int, default=0) # for additional runs after the first search.

    # load arguments
    args = parser.parse_args()
    lrs_b = list(args.lrs_b)
    lrs_m = list(args.lrs_m)
    dropouts = list(args.dropouts)
    h_dims = list(args.h_dims)
    kernel_sizes = list(args.kernel_sizes)
    num_layers = list(args.num_layers)
    ms = list(args.ms)
    ls = list(args.ls)
    ws = list(args.ws)
    nb = args.nb
    batch_sizes = list(args.batch_sizes)
    reduce = args.reduce
    n_splits = args.n_splits
    model_name = args.model_name
    command = args.command
    gpus = args.gpus
    data_name = args.data_name
    exp_name = args.exp_name
    start_idx = args.start_idx

    print(f'Starting hyperparameter search for model {model_name} on dataset {data_name}...')
    
    # create save directory
    save_path = os.path.join('/home/mbito/project_rwnns_molecules/results/', exp_name)
    if not os.path.exists(save_path): 
        os.mkdir(save_path)

    # create model directory
    model_path = '/data1/mbito/models/'
    if not os.path.exists(model_path): 
        os.mkdir(model_path)

    # create search grid
    grid = []
    hyperparameters = {}
    hyperparameters['lrs'] = get_lrs(lrs_b, lrs_m)
    hyperparameters['dropouts'] = dropouts
    hyperparameters['h_dims'] = h_dims
    hyperparameters['kernel_sizes'] = kernel_sizes
    hyperparameters['num_layers'] = num_layers
    hyperparameters['ms'] = ms
    hyperparameters['ls'] = ls
    hyperparameters['ws'] = ws
    hyperparameters['reduce'] = reduce
    hyperparameters['batch_sizes'] = batch_sizes

    for ks in hyperparameters['kernel_sizes']: 
        for lr in hyperparameters['lrs']: 
            for h_dim in hyperparameters['h_dims']:
                for nl in hyperparameters['num_layers']:
                    for dr in hyperparameters['dropouts']: 
                        for rd in hyperparameters['reduce']: 
                            for bs in hyperparameters['batch_sizes']: 
                                for m in hyperparameters['ms']: 
                                    for l in hyperparameters['ls']: 
                                        for w in hyperparameters['ws']: 
                                            grid.append({'lr': lr, 'h_dim': h_dim, 'num_layers': nl, 'kernel_size': ks,
                                                     'dropout': dr, 'reduce': rd, 'batch_size': bs, 'm': m, 'l': l, 'w': w, 'nb': nb})
    
    # run search in parallel
    available_gpus = {gpu: subprocess.Popen('', shell=True) for gpu in gpus} # open a process for each available gpu
    model_idx = start_idx
    while len(grid) > 0: 
        for gpu in available_gpus.keys(): 
            if available_gpus[gpu].poll() == 0 and len(grid) > 0: 
                hypers = grid.pop()
                command = format_command(command, hypers, gpu, n_splits, model_idx, model_name, data_name, save_path)
                available_gpus[gpu] = subprocess.Popen(command, shell=True)
                model_idx += 1
        
        time.sleep(1) # wait 1 seconds to check for available gpus

    # wait for all processes to finish, so that next search doesn't overlap with current one
    for gpu in available_gpus.keys():
        available_gpus[gpu].wait()

    # delete model path
    # shutil.rmtree(model_path)




