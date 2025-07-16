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


def train_loop(data, optimizer, my_model, clip=5):
    my_model.train()
    loss_array = []

    intent_loss = nn.CrossEntropyLoss()
    slot_loss = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN,reduction='mean')
    for samples in data:
        
        optimizer.zero_grad() # Zeroing the gradient
        
        tokenized_samples = samples['tokens']
        intent_logits, slot_logits = my_model(tokenized_samples) #returns a batch of samples logits

        new_slots = []
        for slot in samples['slots']:
            full = torch.full((slot_logits.size(1),), PAD_TOKEN) #create a full -1 target slot vector
            full[:slot.size(0)] = slot #put the actual slots in the vector
            new_slots.append(full)
        samples['slots'] = new_slots # list of tensors, each tensor is a max_batch_len = "max batch sample sequence length" padded of shape(max_batch_len,130)

        #loading the targets
        intent_targets = torch.tensor(samples['intent']).to('cuda:0')
        slots_targets = torch.stack(samples['slots'], dim=0).to('cuda:0')
        
        #intent loss computation
        intent_loss_res = intent_loss(intent_logits, intent_targets)

        #slot loss computation
        logits_for_loss = slot_logits.permute(0, 2, 1) #to correctly compute the CrossEntropy
        slot_loss_res = slot_loss(logits_for_loss, slots_targets) #sum all the contributes
        
        #joint loss computation
        joint_loss = intent_loss_res + slot_loss_res
        
        loss_array.append(joint_loss.item())
        # #Parameter Update

        joint_loss.backward()
        
        #gradient clipping
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), clip)

        optimizer.step() # Update the weights

    return loss_array

def eval_loop(data, my_model, lang):
    my_model.eval()

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []

    with torch.no_grad(): # It used to avoid the creation of computational graph
        for samples in data:
            tokenized_samples = samples['tokens']
            intent_logits, slot_logits = my_model(tokenized_samples) #returns a batch of samples logits

            new_slots = []
            for slot in samples['slots']:
                full = torch.full((slot_logits.size(1),), PAD_TOKEN) #create a full -1 target slot vector
                full[:slot.size(0)] = slot #put the actual slots in the vector
                
                new_slots.append(full)
            samples['slots'] = new_slots # list of tensors, each tensor is a max_batch_len = "max batch sample sequence length" padded of shape(max_batch_len,130)
            
            # Intent inference
            # Get the most probable class
            try:
                out_intents = [lang.id2intent[int(x)]
                            for x in torch.argmax(intent_logits, dim=1)]
                gt_intents = [lang.id2intent[int(x)] for x in samples['intent']]
                ref_intents.extend(gt_intents)
                hyp_intents.extend(out_intents)
            except:
                print("something wrong in intent evaluation")
                breakpoint()
            
            # Slot inference
            # get the most probable slot for each element of the sequence (non padded elements) 
            output_slots = torch.argmax(slot_logits, dim=2) #shape(Batch_size * max_seq_length)
    
            # filter out the PAD_TOKEN
            predicted_encoded_slots = [output_slots[idx,sequence_slot != PAD_TOKEN] for idx,sequence_slot in enumerate(samples['slots'])] #from model prediction
            ref_encoded_slots = [sequence_slot[sequence_slot != PAD_TOKEN] for sequence_slot in samples['slots']] #from targets


            # slot_logits is max_seq_lenght - 2(SEP and CLS token), so I know that taking only the token ids that are different from the padded target slots gives me the right tokens
            #the token is not actually used to compute the evaluate, so instead of wasting computation filtering the tokens and convert those to words I just use a placeholder

            #convert the target and the predicted to IOB
            batch_ref_slots = [('placeholder', lang.id2slot[int(elem)]) for single_sample in ref_encoded_slots for elem in single_sample ] #list of concatenated samples IOB
            batch_hyp_slots = [('placeholder',lang.id2slot[int(elem)]) for single_sample in predicted_encoded_slots for elem in single_sample ]

            ref_slots.append(batch_ref_slots)
            hyp_slots.append(batch_hyp_slots)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        results = {"total":{"f":0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                          zero_division=False, output_dict=True)
    return results, report_intent

def init_optimizer(model,lr,weight_decay):
    """
    initialize the Optimizer
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def init_model():
    """
    initialize the FineTune class and return the model 
    """
    model = MODEL.to('cuda:0') 
    my_model = BERTFineTune(model)
    my_model = my_model.to('cuda:0')

    return my_model

    
def train_and_evaluate(model, optimizer, epochs,patience, logging_int,wandb):    
    """
    Assure that the number of epochs is a multiple of the `logging_int`
    
    returns: `best_model`, `test_f1`, `test_intent_accuracy` 
    best model and that model test time f1 score and accuracy
    """
    running_patience = patience
    best_value = 0
    for x in tqdm(range(1,epochs + 1)):
        loss = train_loop(TRAIN_LOADER, optimizer, model)
        
        if x % logging_int == 0: # We check the performance every "logging_int"

            results_dev, intent_res = eval_loop(DEV_LOADER, model, LANG)
            
            
            train_loss = np.asarray(loss).mean()

            f1 = results_dev['total']['f']

            #wandb logs
            if wandb is not None:
                log_dict = {
                    f"x_axis_{logging_int}": x,
                    "train_loss": train_loss,
                    "dev_f1": f1,
                    "dev_accuracy": intent_res['accuracy']
                }
                wandb.log(log_dict)

            # the best model is chosen based on the combination of accuracy and f1
            if f1 + intent_res['accuracy'] > best_value:
                best_value = f1 + intent_res['accuracy']
                # Here you should save the model
                best_model = copy.deepcopy(model).to('cpu')
                running_patience = patience
            else:
                running_patience -= 1
            if running_patience <= 0: # Early stopping with patience
                print("Early stopping")
                break # Not nice but it keeps the code clean

    #testing the model, the best one
    results_test, intent_test = eval_loop(TEST_LOADER, best_model.to(DEVICE), LANG)
    
    test_f1 = results_test['total']['f'] #f1 score for slot filling
    test_intent_accuracy = intent_test['accuracy'] #accuracy for intent 
    if wandb is not None:
        log_dict = {
            "test_f1": test_f1,
            "test_accuracy": test_intent_accuracy
        }
        wandb.log(log_dict)
    
    return best_model, test_f1, test_intent_accuracy
    
