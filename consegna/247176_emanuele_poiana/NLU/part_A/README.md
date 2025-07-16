## Wandb benchmarking

Is possible to log automatically the run on Weight&Biases, to do so in `main.py`:
1. set the `wandb` flag to `True`
2. set your API_KEY, entity and project name in the corresponding variables substituting the placeholders

## Models
under the `\bin` directory are present:
- BIDIRECTIONAL: which is the best bidirectional model
- DROPOUT: which is the best bidirectional + dropout model
- IAS: which is the baseline model with the best configuration

See the report for the model hyperparameters setting
## Notes
Using LAB 5 as baseline, the functions taken directly from the lab will not be commented or described in particular (if I do not have made any modification to those)

Inside `functions` have been copied the functions necessary to run `evaluate` from `conll.py`.

Original code on [link](https://raw.githubusercontent.com/BrownFortress/NLU-2024-Labs/main/labs/conll.py)