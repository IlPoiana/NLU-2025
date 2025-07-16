# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py
import wandb
import os

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results

    log = False #wandb flag, set True if you want to log on it

    lr = 0.0001 # learning rate
    weight_decay = 0.1

    model = init_model() #load the BERT fine tune model

    optim_params = {
        'lr':lr,
        'weight_decay': weight_decay
    }

    optimizer = init_optimizer(model, **optim_params)

    epochs = 200
    patience = 5
    logging_int = 1

    fname = f'BERT-{lr}-{weight_decay}'
    if log:
        os.environ['WANDB_API_KEY'] =  "your_wb_api_key"
        os.environ['WANDB_BASE_URL'] = 'https://api.wandb.ai'

        wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"))
        # Wandb script to track a run
        run = wandb.init(
            name = fname,
            # Set the wandb entity where your project will be logged (generally your team name).
            entity= "wandb_entity",
            # Set the wandb project where this run will be logged.
            project= "project_name",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": lr,
                "architecture": "BERT",
                "epochs": str(epochs),
            },
        )

        run.define_metric(step_metric = f"x_axis_{logging_int}", name = "train_loss")
        run.define_metric(step_metric = f"x_axis_{logging_int}", name = "dev_loss")
        run.define_metric(step_metric = f"x_axis_{logging_int}", name = "dev_f1")
        run.define_metric(step_metric = f"x_axis_{logging_int}", name = "dev_accuracy")
        

        wandb_logger = run
    else: 
        wandb_logger = None
    
    
    training_params = {
        'epochs': epochs, 
        'patience': patience, 
        'logging_int': logging_int,
        'wandb': wandb_logger
    }
    print("training_params: ",training_params)
    trained_model, slot_f1, intent_accuracy= train_and_evaluate(model, optimizer, **training_params)

    print('Slot F1: ', slot_f1)
    print('Intent Accuracy:', intent_accuracy)
    
    #save the model
    path = f"bin/{fname}.pt"
    torch.save(trained_model.state_dict(), path)

    if log:
        wandb_logger.finish()


    