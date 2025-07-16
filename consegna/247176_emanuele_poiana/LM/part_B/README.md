## Wandb benchmarking

Is possible to log automatically the run on Weight&Biases, to do so in `main.py`:
1. set the `wandb` flag to `True`
2. set your API_KEY, entity and project name in the corresponding variables substituting the placeholders

> [!Warning]
> Be sure to remove the `wandb import` if not installed in the environment

## Models
under the `\bin` directory are present:
- WEIGHT: which is the SGD baseline (part 1A.1 LSTM) with the same parameters but weight tying, which improves the baseline
- VARIATIONAL-SGD: which is the best Weight Tying + Variational Dropout SGD model
- VARIATIONAL-SGD-NT: which is the best Weight Tying + Variational Dropout SGD => NT-ASGD model 
- VARIATIONAL-Adam: which is the best model I have trained for this task. (ppl = 88)

See the report for the model hyperparameters setting

## Notes
Using LAB 4 as baseline, the functions taken directly from the lab will not be commented or described in particular (if I do not have made any modification to those)