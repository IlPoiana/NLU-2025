# Add functions or classes used for data loading and preprocessing
import os
import json
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from model import TOKENIZER

PAD_TOKEN = 0
DEVICE = 'cuda:0'

def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


## DEV SET CREATION


def initialize_dev_set():
    """
    returns train and dev set
    """
    # First we get the 10% of the training set, then we compute the percentage of these examples
    tmp_train_raw = load_data(os.path.join('dataset/ATIS','train.json'))

    portion = 0.10

    intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=labels)
    X_train.extend(mini_train)
    return X_train, X_dev


class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        self.id2slot = {v:k for k, v in self.slot2id.items()}
        self.id2intent = {v:k for k, v in self.intent2id.items()}

    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
                vocab[elem] = len(vocab)
        return vocab

class BertIntentsAndSlots (data.Dataset):
    """
    This implementation prepare all the data to be directly feed into the Dataloader, optimizing the preprocessing
    """

    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance']) #array of utterances
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = self.utterances[idx]
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample

    # Auxiliary methods

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res

def BERT_collate_fn(data):
    """
    Is assumed to have the BERT tokenizer
    `data`: batch o samples
    Returns:

    """
    #call the tokenizer
    tok = TOKENIZER([d['utterance'] for d in data], return_tensors = "pt", padding=True).to('cuda:0')
    slots = []
    #generate the tokens
    # ' cases
    # ## cases
    for sample in data:
        tokens = TOKENIZER.tokenize(sample['utterance'])
        orig_slots = sample['slots']

        new_slots = []
        slot_idx = 0
        prev_was_quote = False

        for token in tokens:
            # if subword, apostrophe, or right after apostrophe → -1
            # if token.startswith("##") or token == "'" or token == '.' or prev_was_quote:
            if token.startswith("##") or token == "'" or token == '.':
                new_slots.append(PAD_TOKEN)
            elif prev_was_quote and token == 'clock':# handeling 5 o'clock 4 => 2
                new_slots.append(PAD_TOKEN)
            else:
                # assign next original-slot and advance
                try:
                    new_slots.append(orig_slots[slot_idx])
                    slot_idx += 1
                except:
                    breakpoint()

            # flag if this token is an apostrophe, to catch the next one
            prev_was_quote = (token == "'")

        # sanity check: consumed exactly all original slots
        if slot_idx != len(orig_slots):
            print("slots conversion went wrong!", slot_idx, orig_slots)
            breakpoint()

        slots.append(torch.tensor(new_slots))

    tokenized_utt = [TOKENIZER.tokenize(d['utterance']) for d in data] #extract the char tokens for each sample

    intent = [d['intent'] for d in data]
    return {'tokens': tok, 'tokenized_utt': tokenized_utt,'slots': slots, 'intent': intent}


    
train_raw, dev_raw = initialize_dev_set()

#Raw creation
test_raw = load_data(os.path.join('dataset/ATIS','test.json'))

words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute
                                                            # the cutoff
corpus = train_raw + dev_raw + test_raw 

slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

LANG = Lang(words, intents, slots, cutoff=0)

OUT_SLOT = len(LANG.slot2id)
OUT_INT = len(LANG.intent2id)
VOCAB_LEN = len(LANG.word2id)

# Datasets creation
train_dataset = BertIntentsAndSlots(train_raw, LANG)
dev_dataset = BertIntentsAndSlots(dev_raw, LANG)
test_dataset = BertIntentsAndSlots(test_raw, LANG)

# Dataloader instantiations
TRAIN_LOADER = DataLoader(train_dataset, batch_size=256, collate_fn=BERT_collate_fn,  shuffle=True)
DEV_LOADER = DataLoader(dev_dataset, batch_size=128, collate_fn=BERT_collate_fn)
TEST_LOADER = DataLoader(test_dataset, batch_size=128, collate_fn=BERT_collate_fn)