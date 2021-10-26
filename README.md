# Improving Anytime Prediction with Parallel Cascaded Networks and a Temporal-Difference Loss

---

This repo provides code for "Improving Anytime Prediction with Parallel Cascaded Networks and a Temporal-Difference Loss" [paper](https://arxiv.org/abs/2102.09808). 

---

# Environment Setup
Create the environment with Conda: `conda env create -f conda_env.yml`

Then activate environment: `conda activate cascaded_nets`

---

# Training
Use `train.sh` as a template to set your own hyperparameters, which launches `train.py`.
In `train.sh`, be sure to specify `DATASET_ROOT`, `EXPERIMENT_ROOT`, and `SPLIT_IDXS_ROOT` to locations on you system where your datasets are located, where you want the output from `train.py` to be written (contains ckpt files, etc.), and the location for the dataset `split_idxs` (train, val, test splits) for reproducibility and consistency, respectively.

# Evaluate
Use `eval.sh` to load and evaluate the model stored in `EXPERIMENT_NAME`. This script will evaluate the performance of the model and, if `--keep_logits` is specified, generate and store the logits for all examples in the specified dataset. The logits are useful for downstream tasks, such as training metacognition models.

Similar to `train.sh`, specify `DATASET_ROOT`, `EXPERIMENT_ROOT`, and `SPLIT_IDXS_ROOT`.