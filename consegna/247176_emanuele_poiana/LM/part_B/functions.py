import torch
import torch.optim as optim
import torch.nn as nn
import math
from tqdm import tqdm
import copy
from utils import *
from model import *



def train_loop(data, optimizer, criterion, model, clip=5):
    """
    Train loop

    - `data`: the data loader
    - `optimizer`: what optimizer to use
    - `criterion`: the loss to use
    - `model`: the model to use and to train, forward pass
    """
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph

        # clip the gradient to avoid exploding gradients, truncated BPTT
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights

    return sum(loss_array)/sum(number_of_tokens)


# Evaluation loop
# 1. Deactivation of Dropout: dropout layers are turned off
# 2. Batch Normalization Behavior: batch normalization layers employ population statistics rather than batch statistics
def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

# random weight initialization
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)



def init_model(architecture = 'RNN', params = {
        'hid_size': 512,
        'emb_size': 512,
        'emb_dropout': 0.3,
        'out_dropout': 0.2,
    }):
    """
    Function for initilize the model.
    ## parameters
    - `architecture`: specify the model architecture among `RNN`, `LSTM` and `DROPOUT`
    - `params`: model parameters, it's a dictionary containing all the relevant parameters for the model architecture

    params:
    ```
    {
        emb_size: 
        hid_size:
        vocab_len:
        emb_dropout:
        out_dropout:
    }
    ```
    """
    if architecture == 'LSTM':
        model = LM_LSTM(**params, output_size=VOCAB_LEN,pad_index=LANG.word2id["<pad>"]).to(DEVICE)
    elif architecture == 'WEIGHT': #Weight tying
        model = LSTM_WeightTying(**params, output_size=VOCAB_LEN,pad_index=LANG.word2id["<pad>"]).to(DEVICE)
    elif architecture == 'VARIATIONAL':
        model = LSTM_VariationalDropout(**params, output_size=VOCAB_LEN,pad_index=LANG.word2id["<pad>"]).to(DEVICE)    
    return model

def init_optimizer(optimizer, model,params = {'lr': 0.1,'weight_decay': 0.1}):
    """
    Return the initialized optimizer
    - `optimizer`: what optimizer to use between `SGD` and `AdamW`
    - `model`: an initialized torch model
    - `params`: optimizer parameters
    """
    if optimizer == 'SGD': #SGD
        optimizer = optim.SGD(model.parameters(), **params)
    elif optimizer == 'AdamW': #AdamW
        optimizer = optim.AdamW(model.parameters(), **params) 
    else:
        print('Optmizer', optimizer, ' not valid!')
        return None
    return optimizer

def train_and_evaluate(model, optimizer, epochs, clip, patience, wandb = None):
    """
    Train a `model` for `epochs`
    Returns the best trained model, the best dev perplexity and the test preplexity
    - `wandb`: a `wandb` object initialized and logged in, where print the run
    """

    criterion_train = nn.CrossEntropyLoss(ignore_index=LANG.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=LANG.word2id["<pad>"], reduction='sum')

    trained_model = model
    best_ppl = math.inf
    patience_count = 0
    for epoch in tqdm(range(epochs)):

        loss = train_loop(TRAIN_LOADER, optimizer, criterion_train, model, clip)
        ppl_dev, loss_dev = eval_loop(DEV_LOADER, criterion_eval, model)
        
        #wandb log
        if wandb is not None:
            wandb.log({"train_loss": loss,"dev_loss": loss_dev, "dev_ppl": ppl_dev})
        # evaluating the model
        if  ppl_dev < best_ppl: 
                best_ppl = ppl_dev
                trained_model = copy.deepcopy(model).to('cpu')
                if patience > 0:
                    patience_count -= 1 #decrease the patience when finding a better model(not resetting)
        else: #patience increased when I've not found a better ppl
            patience_count += 1
        if patience_count == patience:
            print('\nEarly stopping')
            break
    # reload the final best model to the GPU
    trained_model.to(DEVICE)
    final_ppl,  _ = eval_loop(TEST_LOADER, criterion_eval, trained_model)
    print('Test ppl: ', final_ppl)
    if wandb is not None:
        wandb.log({"final_ppl" : final_ppl})
        wandb.finish()
    return trained_model, best_ppl, final_ppl
        

def train_and_evaluate_NT(model, optimizer, epochs, clip, patience, logging_int,n_int,wandb = None):
    """
    Same as train_and_evaluate but with non monotonically triggered ASDG
    - `wandb`: a `wandb` object initialized and logged in, where print the run
    - `n_int`: non-monotone interval
    - `logging_int:` logging interval >= 1 epoch
    """
    

    criterion_train = nn.CrossEntropyLoss(ignore_index=LANG.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=LANG.word2id["<pad>"], reduction='sum')

    trained_model = model
    best_ppl = math.inf
    patience_count = 0

    log_epoch = 0
    actual_optimizer = optimizer
    switched = True

    for epoch in tqdm(range(epochs)):
        #increase the count of the logged epochs
        if epoch % logging_int == 0: 
            log_epoch += 1  
        
        loss = train_loop(TRAIN_LOADER, actual_optimizer, criterion_train, model, clip)
        ppl_dev, loss_dev = eval_loop(DEV_LOADER, criterion_eval, model)
        #n_int set a minimum bound to switch to ASGD, switched is to prevent more than one switch
        if ppl_dev > best_ppl and log_epoch > n_int and switched: 
            #switch to ASGD
            opt_prms = optimizer.param_groups[0]
            actual_optimizer = optim.ASGD(opt_prms['params'],opt_prms['lr'],t0=0,lambd=0.,weight_decay=opt_prms['weight_decay'])
            switched = False
            print("switched to ASGD")

        #wandb log
        if wandb is not None:
            wandb.log({"train_loss": loss,"dev_loss": loss_dev, "dev_ppl": ppl_dev})
        if ppl_dev < best_ppl: # the lower, the better
            best_ppl = ppl_dev
            trained_model = copy.deepcopy(model).to('cpu')
            if patience > 0:
                patience_count -= 1
        else:
            patience_count += 1
        if patience_count == patience:
            print('\nEarly stopping')
            break
    # reload the final best model to the GPU
    trained_model.to(DEVICE)
    final_ppl,  _ = eval_loop(TEST_LOADER, criterion_eval, trained_model)
    print('Test ppl: ', final_ppl)
    if wandb is not None:
        wandb.log({"final_ppl" : final_ppl})
        wandb.finish()
    return trained_model, best_ppl, final_ppl
        

