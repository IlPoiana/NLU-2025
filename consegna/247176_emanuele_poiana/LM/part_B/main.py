# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import wandb
import os
# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Write the code to load the datasets and to run your functions
    # Print the results

    # Variables definition
    # wandb
    log = False

    #Architecture
    architecture = 'VARIATIONAL' #VARIATIONAL, WEIGHT
    hid_size = 512
    emb_size = 512

    #Dropout Variables
    emb_dropout = 0.3# 0.4 
    out_dropout = 0.2# 0.5

    #Optimizer
    optim_name = 'SGD'
    lr = 0.5 #0.005
    weight_decay = 0 #0.1
    
    #NT variables
    NT_flag = False 
    logging_int = 1 #logging interval 
    n_int = 5 # minimum number of logs

    #Training loop
    clip = 4
    epochs = 64
    patience = 5

    fname = f'{architecture}-{lr}-{weight_decay}-{emb_dropout}-{out_dropout}-NT{NT_flag}'
    if log:
        os.environ['WANDB_API_KEY'] = "your_API_KEY"
        os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'

        wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"))
        # Wandb script to track a run
        run = wandb.init(
            name = fname, #the parameters not specified are considered default
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="placeholder",
            # Set the wandb project where this run will be logged.
            project="placeholder",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                "architecture": architecture,
                "epochs": str(epochs),
            },
        )

        wandb_logger = run
    else: 
        wandb_logger = None


    #initialize the model
    model_params = {
        'hidden_size': hid_size,
        'emb_size': emb_size,
        'emb_dropout': emb_dropout,
        'out_dropout': out_dropout,
    }
    model = init_model(architecture, model_params)
    
    model.apply(init_weights)

    #initialize the optimizer
    optim_params = {
        'lr': lr,
        'weight_decay': weight_decay
    }

    optimizer = init_optimizer(optim_name, model,optim_params)

    #set the training parameters
    training_params = {
        'epochs': epochs, 
        'clip': clip, 
        'patience': patience, 
        'wandb': wandb_logger
    }
    # Non monotonically triggered switch
    if NT_flag:
        training_params = {
        'epochs': epochs, 
        'clip': clip, 
        'patience': patience, 
        'logging_int': logging_int,
        'n_int': n_int, 
        'wandb': wandb_logger
        }
        print("NT optimizer training_params: ",training_params)
        trained_model, best_ppl, test_ppl= train_and_evaluate_NT(model, optimizer, **training_params)
    else:
        training_params = {
            'epochs': epochs, 
            'clip': clip, 
            'patience': patience, 
            'wandb': wandb_logger
        }
        print("std. optimizer training_params: ",training_params)
        trained_model, best_ppl, test_ppl= train_and_evaluate(model, optimizer, **training_params)

    print("finished training: ", best_ppl)
    print("test ppl: ", test_ppl)

    path = f"bin/{fname}.pt"
    torch.save(trained_model.state_dict(), path)