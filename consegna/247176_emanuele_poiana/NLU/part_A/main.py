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

    
    architecture = 'IAS'  # 'DROPOUT' 'BIDIRECTIONAL'

    hid_size = 200
    emb_size = 300
    emb_dropout = 0
    out_dropout = 0.2

    lr = 0.0001 # learning rate
    weight_decay = 0.1
    clip = 5 # Clip the gradient

    model = init_model(architecture, hid_size,emb_size, emb_dropout, out_dropout) #load the correct model
    model.apply(init_weights)

    optim_params = {
        'lr':lr,
        'weight_decay': weight_decay
    }

    optimizer, criterion_slots, criterion_intents = init_optimizer(model, **optim_params)

    runs_number=10
    epochs = 200
    patience = 5
    logging_int = 5

    fname = f'multi-{architecture}-{emb_size}-{hid_size}-{lr}-{weight_decay}'
    
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
                "architecture": architecture,
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
        'clip': clip, 
        'patience': patience, 
        'criterion_slots': criterion_slots,
        'criterion_intents': criterion_intents,
        'logging_int': logging_int,
        'wandb': wandb_logger
    }
    print("training_params: ",training_params)
    best_model, avg_slot, avg_intent = train_and_evaluate_multi(runs_number,model, optimizer, **training_params)

    print('Slot F1: ', avg_slot)
    print('Intent Accuracy:', avg_intent)
    
    #save the model
    path = f"bin/{fname}.pt"
    torch.save(best_model.state_dict(), path)
    
    if log:
        wandb_logger.log({'avg_slot_f1': avg_slot,'avg_intent_accuracy': avg_intent})
        wandb_logger.finish()

    