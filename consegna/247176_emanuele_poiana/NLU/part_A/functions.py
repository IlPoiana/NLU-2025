# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from utils import *
from model import *
import copy
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import re

def init_weights(mat):
    """
    initialize weights for all the model parameters
    """
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)


#### copying the functions from conll to avoid conll import 

def evaluate(ref, hyp, otag='O'):
    # evaluation for NLTK
    aligned = align_hyp(ref, hyp)
    return conlleval(aligned, otag=otag)

def align_hyp(ref, hyp):
    # align references and hypotheses for evaluation
    # add last element of token tuple in hyp to ref
    if len(ref) != len(hyp):
        raise ValueError("Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))

    out = []
    for i in range(len(ref)):
        if len(ref[i]) != len(hyp[i]):
            raise ValueError("Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))
        out.append([(*ref[i][j], hyp[i][j][-1]) for j in range(len(ref[i]))])
    return out


def conlleval(data, otag='O'):
    # token, segment & class level counts for TP, TP+FP, TP+FN
    tok = stats()
    seg = stats()
    cls = {}

    for sent in data:

        prev_ref = otag      # previous reference label
        prev_hyp = otag      # previous hypothesis label
        prev_ref_iob = None  # previous reference label IOB
        prev_hyp_iob = None  # previous hypothesis label IOB

        in_correct = False  # currently processed chunks is correct until now

        for token in sent:

            hyp_iob, hyp = parse_iob(token[-1])
            ref_iob, ref = parse_iob(token[-2])

            ref_e = is_eoc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_e = is_eoc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)

            ref_b = is_boc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_b = is_boc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)

            if not cls.get(ref) and ref:
                cls[ref] = stats()

            if not cls.get(hyp) and hyp:
                cls[hyp] = stats()

            # segment-level counts
            if in_correct:
                if ref_e and hyp_e and prev_hyp == prev_ref:
                    in_correct = False
                    seg['cor'] += 1
                    cls[prev_ref]['cor'] += 1

                elif ref_e != hyp_e or hyp != ref:
                    in_correct = False

            if ref_b and hyp_b and hyp == ref:
                in_correct = True

            if ref_b:
                seg['ref'] += 1
                cls[ref]['ref'] += 1

            if hyp_b:
                seg['hyp'] += 1
                cls[hyp]['hyp'] += 1

            # token-level counts
            if ref == hyp and ref_iob == hyp_iob:
                tok['cor'] += 1

            tok['ref'] += 1

            prev_ref = ref
            prev_hyp = hyp
            prev_ref_iob = ref_iob
            prev_hyp_iob = hyp_iob

        if in_correct:
            seg['cor'] += 1
            cls[prev_ref]['cor'] += 1

    return summarize(seg, cls)


def parse_iob(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, None)


def is_boc(lbl, iob, prev_lbl, prev_iob, otag='O'):
    """
    is beginning of a chunk

    supports: IOB, IOBE, BILOU schemes
        - {E,L} --> last
        - {S,U} --> unit

    :param lbl: current label
    :param iob: current iob
    :param prev_lbl: previous label
    :param prev_iob: previous iob
    :param otag: out-of-chunk label
    :return:
    """
    boc = False

    boc = True if iob in ['B', 'S', 'U'] else boc
    boc = True if iob in ['E', 'L'] and prev_iob in ['E', 'L', 'S', otag] else boc
    boc = True if iob == 'I' and prev_iob in ['S', 'L', 'E', otag] else boc

    boc = True if lbl != prev_lbl and iob != otag and iob != '.' else boc

    # these chunks are assumed to have length 1
    boc = True if iob in ['[', ']'] else boc

    return boc


def is_eoc(lbl, iob, prev_lbl, prev_iob, otag='O'):
    """
    is end of a chunk

    supports: IOB, IOBE, BILOU schemes
        - {E,L} --> last
        - {S,U} --> unit

    :param lbl: current label
    :param iob: current iob
    :param prev_lbl: previous label
    :param prev_iob: previous iob
    :param otag: out-of-chunk label
    :return:
    """
    eoc = False

    eoc = True if iob in ['E', 'L', 'S', 'U'] else eoc
    eoc = True if iob == 'B' and prev_iob in ['B', 'I'] else eoc
    eoc = True if iob in ['S', 'U'] and prev_iob in ['B', 'I'] else eoc

    eoc = True if iob == otag and prev_iob in ['B', 'I'] else eoc

    eoc = True if lbl != prev_lbl and iob != otag and prev_iob != '.' else eoc

    # these chunks are assumed to have length 1
    eoc = True if iob in ['[', ']'] else eoc

    return eoc


def score(cor_cnt, hyp_cnt, ref_cnt):
    # precision
    p = 1 if hyp_cnt == 0 else cor_cnt / hyp_cnt
    # recall
    r = 0 if ref_cnt == 0 else cor_cnt / ref_cnt
    # f-measure (f1)
    f = 0 if p+r == 0 else (2*p*r)/(p+r)
    return {"p": p, "r": r, "f": f, "s": ref_cnt}


def summarize(seg, cls):
    # class-level
    res = {lbl: score(cls[lbl]['cor'], cls[lbl]['hyp'], cls[lbl]['ref']) for lbl in set(cls.keys())}
    # micro
    res.update({"total": score(seg.get('cor', 0), seg.get('hyp', 0), seg.get('ref', 0))})
    return res

def stats():
    return {'cor': 0, 'hyp': 0, 'ref': 0}

###### end conll functions


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for idx, sample in enumerate(data):
        optimizer.zero_grad() # Zeroing the gradient
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        
        loss = loss_intent + loss_slot
      
        loss_array.append(loss.item())
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step() # Update the weights
    return loss_array

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    #softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            
            loss = loss_intent + loss_slot
            
            
            loss_array.append(loss.item())
            # Intent inference
            # Get the highest probable class
            out_intents = [lang.id2intent[x]
                           for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total":{"f":0}}

    #intent classification, accuracy metric inside report_intent(dict with accuracy value)
    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent, loss_array

def init_optimizer(model,lr,weight_decay):
    """
    Returns the initialized optimizer and loss functions
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() 

    return optimizer,criterion_slots, criterion_intents

def init_model(architecture, hid_size, emb_size, _ , out_dropout):
    """
    Returns the model specified by `architecture`, with the structural parameters specified by the other variables
    """
    if architecture == 'IAS': #std architecture
        model = ModelIAS(hid_size, OUT_SLOT, OUT_INT, emb_size, VOCAB_LEN, pad_index=PAD_TOKEN).to(DEVICE)
    elif architecture == 'DROPOUT': # Bidirectional + 1 dropout layer
        model = DropoutIAS(hid_size, OUT_SLOT, OUT_INT, emb_size, VOCAB_LEN, out_dropout,pad_index=PAD_TOKEN).to(DEVICE)
    elif architecture == 'BIDIRECTIONAL': # Bidirectional
        model = BidirectionalIAS(hid_size, OUT_SLOT, OUT_INT, emb_size, VOCAB_LEN, pad_index=PAD_TOKEN).to(DEVICE)
    return model

    
def train_and_evaluate(model, optimizer, epochs,clip,patience, logging_int,wandb,criterion_slots,criterion_intents):    
    """
    Assure that the number of epochs is a multiple of the `logging_int`
    
    returns: `best_model`, `test_f1`, `test_intent_accuracy` 
    best model is the test time model with the highest f1 score and accuracy sum
    """
    running_patience = patience
    best_value = 0
    for x in tqdm(range(1,epochs + 1)):
        loss = train_loop(TRAIN_LOADER, optimizer, criterion_slots,
                        criterion_intents, model, clip=clip)
        
        if x % logging_int == 0: # We check the performance every patience
            results_dev, intent_res, loss_dev = eval_loop(DEV_LOADER, criterion_slots,
                                                        criterion_intents, model, LANG)
            
            
            train_loss = np.asarray(loss).mean()
            dev_loss = np.asarray(loss_dev).mean()

            f1 = results_dev['total']['f']

            #wandb logs
            if wandb is not None:
                log_dict = {
                    f"x_axis_{logging_int}": x,
                    "train_loss": train_loss,
                    "dev_loss": dev_loss,
                    "dev_f1": f1,
                    "dev_accuracy": intent_res['accuracy']
                }
                wandb.log(log_dict)

            if f1 + intent_res['accuracy'] > best_value:
                best_value = f1 + intent_res['accuracy']
                # Here you should save the model
                best_model = copy.deepcopy(model).to('cpu') #scaving the "best" model
                running_patience = patience
            else:
                running_patience -= 1
            if running_patience <= 0: # Early stopping with patience
                print("Early stopping")

    #testing the model, the best one
    results_test, intent_test, _ = eval_loop(TEST_LOADER, criterion_slots,
                                            criterion_intents, best_model.to(DEVICE), LANG)
    
    test_f1 = results_test['total']['f'] #f1 score for slot filling
    test_intent_accuracy = intent_test['accuracy'] #accuracy for intent 
    if wandb is not None:
        log_dict = {
            "test_f1": test_f1,
            "test_accuracy": test_intent_accuracy
        }
        wandb.log(log_dict)
    
    return best_model, test_f1, test_intent_accuracy
    

def train_and_evaluate_multi(runs_number,model, optimizer,training_params):
    """
    Do multiple training runs and averages the resulting metrics, return the best model of all runs and the avg accuracy and f1.
    - `runs_number`: indicates how many runs do
    - `wandb`: if present, is the metric logger to wandb 
    """
    slot_array= []
    intent_array = []
    best_slot = 0
    best_intent = 0
    wandb = training_params['wandb']
    instance_params = copy.deepcopy(training_params)
    instance_params['wandb']  = None
    for run_idx in range(runs_number):
        trained_model, slot_f1, intent_accuracy= train_and_evaluate(model, optimizer, **instance_params)
        slot_array.append(slot_f1)
        intent_array.append(intent_accuracy)
        if slot_f1 + intent_accuracy > best_slot + best_intent:
            best_slot = slot_f1
            best_intent = intent_accuracy
            best_model = trained_model
        if wandb is not None:
            log_dict = {
                "test_f1": slot_f1,
                "test_accuracy": intent_accuracy
            }
            wandb.log(log_dict)
    return best_model, np.asarray(slot_array).mean(), np.asarray(intent_array).mean()

