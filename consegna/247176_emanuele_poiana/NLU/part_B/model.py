import torch.nn as nn
from transformers import BertTokenizer, BertModel


TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased") # Download the tokenizer
MODEL = BertModel.from_pretrained("bert-base-uncased") # Download the model

class BERTFineTune(nn.Module):
    def __init__(self, model, hidden_dim = 768,intent_size = 26, slot_vocab_size = 130):
        super(BERTFineTune, self).__init__()
        self.encoder = model

        self.intent_number = intent_size #26
        self.slot_number = slot_vocab_size  # 130
        self.last_hidden_dim_size = hidden_dim #768

        self.intent_classificator = nn.Linear(self.last_hidden_dim_size, self.intent_number)
        self.slot_classificator = nn.Linear(self.last_hidden_dim_size, self.slot_number)

    def forward(self, input):
        """
        `input`: the tokenized batch to pass to the
        """
        output = self.encoder(**input)

        pooler_output = output.pooler_output # for the intention classification, <CLS> token alone
        last_hidden = output.last_hidden_state # for the slot filling, <CLS>,<SEP> and <PAD> token have to be processed/masked

        #compute the intentions
        intentions = self.intent_classificator(pooler_output)
        #compute the slots
        masked_last_hidden_states = self.slot_classificator(last_hidden[:,1:-1,:]) #removing the CLS token and the SEP or PAD token

        return intentions, masked_last_hidden_states