## Wandb benchmarking

Is possible to log automatically the run on Weight&Biases, to do so in `main.py`:
1. set the `wandb` flag to `True`
2. set your API_KEY, entity and project name in the corresponding variables substituting the placeholders

> [!Warning]
> Be sure to remove the `wandb import` if not installed in the environment

## Models
under the `\bin` directory are present:
- RNN: Best RNN model 
- LSTM: Best LSTM model with the same hidden and output sizes
- DROPOUT: Best LSTM + dropout model, with both a dropout layer before the output and one after the embedding
- DROPOUT-ADAM: The same as DROPOUT but with AdamW, the best model of 1A(ppl 98)

See the report for the model hyperparameters setting

## Notes
Using LAB 4 as baseline, the functions taken directly from the lab will not be commented or described in particular (if I do not have made any modification to those)