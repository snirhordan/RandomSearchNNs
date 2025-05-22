#!/bin/sh

# activate the correct conda environment ~ pytorch (torch)
conda activate torch

python3 main_search_benchmarks.py --lrs_b .001 --lrs_m 1 --batch_size 64 --reduce mean max --h_dims 128 --ms 1 4 8 16 --nb False --num_layers 2 --dropouts 0 --model_name RWNN_base_ada --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .001 --lrs_m 1 --batch_size 64 --reduce mean max --h_dims 128 --ms 1 4 8 16 --nb True --num_layers 2 --dropouts 0 --model_name RWNN_mdlr_ada --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .001 --lrs_m 1 --batch_size 64 --reduce mean max --h_dims 128 --ms 1 4 8 16 --nb False --num_layers 2 --dropouts 0 --model_name RWNN_rum_ada --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --batch_size 64 --reduce mean max --h_dims 64 128 --ms 1 8 16 24 --nb True --num_layers 2 --dropouts 0 --model_name RWNN_crwl_ada --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --batch_size 64 --reduce max --h_dims 128 --ms 1 8 16 24 --ls 25 --nb False --num_layers 2 --dropouts 0 --model_name RWNN_rum --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --batch_size 64 --reduce max --h_dims 128 --ms 1 8 16 24 --ls 25 --nb True --num_layers 2 --dropouts 0 --model_name RWNN_mdlr --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --batch_size 64 --reduce max --h_dims 128 --ms 1 8 16 24 --ls 25 --nb False --num_layers 2 --dropouts 0 --model_name RWNN_base --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --batch_size 64 --reduce mean max --h_dims 64 128 --ms 1 4 8 16 --ls 25 --num_layers 2 3 --dropouts 0 --model_name RWNN_crwl --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --batch_size 64 --reduce mean max --h_dims 64 128 --ms 1 4 8 16 --num_layers 2 3 --dropouts 0 --model_name RSNN --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --batch_size 16 --reduce mean --h_dims 16 --ms 1 4 8 16 --nb True --num_layers 2 --dropouts 0 --model_name RWNN_TRSF_crwl_ada --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
python3 main_search_benchmarks.py --lrs_b .01 .001 --lrs_m 1 --batch_size 16 --reduce mean --h_dims 16 --ms 1 4 8 16 --num_layers 2 --dropouts 0 --model_name RSNN_TRSF --data_name "clintox_rws" --exp_name clintox --command "python3 main_train_benchmarks_rws.py"
