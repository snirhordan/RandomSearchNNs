#!/bin/sh

# activate correct environment
conda activate torch

python3 main_generate_benchmarks_rws.py --idx_smiles -1 --idx_y 0 --l_max 0 --scaffold 0 --data_name pcba --save_directory /home/mbito/data/benchmarks_proc/molecule_learning
python3 main_generate_benchmarks_rws.py --idx_smiles -1 --idx_y 5 --l_max 0 --scaffold 0 --data_name pcba --save_directory /home/mbito/data/benchmarks_proc/molecule_learning
python3 main_generate_benchmarks_rws.py --idx_smiles -1 --idx_y 52 --l_max 0 --scaffold 0 --data_name pcba --save_directory /home/mbito/data/benchmarks_proc/molecule_learning

# python3 main_generate_benchmarks_rws.py --idx_smiles 0 --idx_y 2 --l_max 0 --scaffold 0 --data_name clintox --save_directory /home/mbito/data/benchmarks_proc/molecule_learning
# python3 main_generate_benchmarks_rws.py --idx_smiles 0 --idx_y 1 --l_max 0 --scaffold 0 --data_name sider --save_directory /home/mbito/data/benchmarks_proc/molecule_learning
# python3 main_generate_benchmarks_rws.py --idx_smiles 3 --idx_y 2 --l_max 0 --scaffold 1 --data_name BBBP --save_directory /home/mbito/data/benchmarks_proc/molecule_learning
# python3 main_generate_benchmarks_rws.py --idx_smiles 0 --idx_y 2 --l_max 0 --scaffold 1 --data_name bace --save_directory /home/mbito/data/benchmarks_proc/molecule_learning

# python3 main_generate_benchmarks_rws.py --idx_smiles 0 --idx_y 1 --l_max 0 --scaffold 0 --data_name toxcast --save_directory /home/mbito/data/benchmarks_proc/molecule_learning
# python3 main_generate_benchmarks_rws.py --idx_smiles 13 --idx_y 0 --l_max 0 --scaffold 0 --data_name tox21 --save_directory /home/mbito/data/benchmarks_proc/molecule_learning
# python3 main_generate_benchmarks_rws.py --idx_smiles 13 --idx_y 7 --l_max 0 --scaffold 0 --data_name tox21_7 --save_directory /home/mbito/data/benchmarks_proc/molecule_learning
# python3 main_generate_benchmarks_rws.py --idx_smiles 0 --idx_y 2 --l_max 0 --scaffold 1 --data_name HIV --save_directory /home/mbito/data/benchmarks_proc/molecule_learning
