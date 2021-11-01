# Improving Anytime Prediction with Parallel Cascaded Networks and a Temporal-Difference Loss

This repository is the official implementation of [Improving Anytime Prediction with Parallel Cascaded Networks and a Temporal-Difference Loss](https://arxiv.org/abs/2102.09808). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
Use `train.sh` as a template to set your own hyperparameters, which launches `train.py`.
In `train.sh`, be sure to specify `DATASET_ROOT`, `EXPERIMENT_ROOT`, and `SPLIT_IDXS_ROOT` to locations on you system where your datasets are located, where you want the output from `train.py` to be written (contains ckpt files, etc.), and the location for the dataset `split_idxs` (train, val, test splits) for reproducibility and consistency, respectively.

To train the model(s), run the `train.sh` script.


## Evaluation
Use `eval.sh` to load and evaluate the model stored in `EXPERIMENT_NAME`. This script will evaluate the performance of the model and, if `--keep_logits` is specified, generate and store the logits for all examples in the specified dataset. The logits are useful for downstream tasks, such as training metacognition models.

Similar to `train.sh`, specify `DATASET_ROOT`, `EXPERIMENT_ROOT`, and `SPLIT_IDXS_ROOT`.

To evaluate the model, run `eval.sh`

Analyze results with `Analysis.ipynb`

## Results

<object data="figures/speed_acc.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="figures/speed_acc.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="figures/speed_acc.pdf">Download PDF</a>.</p>
    </embed>
</object>
