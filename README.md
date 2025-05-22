# RandomSearchNNs
[Under Review] Efficient and Expressive Graph Learning via Random Search Neural Networks

## Instructions for Reproducibility
1. In the `generation` directory, use the following command to download and preprocess datasets from PyTorch Geometric:
   ```bash
   source generate_benchmarks.sh
   ```
2. Once the datasets are downloaded and preprocessed, in the `root` directory, use the following command to train models and save results to the `results` directory.
   ```bash
   source search_benchmarks_rws.sh
   ```
3. After training the models and saving the results, load the results in the `load_results.ipynb` notebook located in the `results` directory.
