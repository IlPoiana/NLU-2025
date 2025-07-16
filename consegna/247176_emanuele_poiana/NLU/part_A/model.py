import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=False, batch_first=True)
        self.slot_out = nn.Linear(hid_size, out_slot)
        self.intent_out = nn.Linear(hid_size, out_int)
        

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    
class BidirectionalIAS(nn.Module):
    """
    Bidirectional IAS model
    """
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
    
        super(BidirectionalIAS, self).__init__()

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        #Linear layer for slot
        self.slot_out = nn.Linear(2 * hid_size, out_slot) #doubleing the incoming hidden size cause for bidirectionality
        #Linear layer for intent
        self.intent_out = nn.Linear(hid_size, out_int) 

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        # Get the last hidden state, but taking the first element cause it's bidirectional so is the reverse layer
        last_hidden = last_hidden[0,:,:]
        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1) # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
    
class DropoutIAS(nn.Module):
    """
    Bidirectional + Dropout IAS model
    """
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, dropout,n_layer=1, pad_index=0):
        #dropout: the dropout rate 
        super(DropoutIAS, self).__init__()
        

        self.hid_size = hid_size
        #declare the dropout layer
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True, batch_first=True)
        self.slot_out = nn.Linear(2 * hid_size, out_slot) #doubleing the incoming hidden size cause for bidirectionality
        self.intent_out = nn.Linear(hid_size, out_int) 

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost

        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        # Process the batch
        packed_output, (_, _) = self.utt_encoder(packed_input)

        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        #Apply the Dropout mask for the last hidden and utterance
        utt_dropped = self.dropout(utt_encoded)
        
        last_hidden = utt_dropped.permute(1,0,2)[0] #this access the batch_size * hidden_size last element first element of the sequence [straight_layer, reverse_layer]
        last_hidden = last_hidden[:,self.hid_size:] #taking the logits of the reverse direction
    
        # Compute slot logits
        slots = self.slot_out(utt_dropped)
        # Compute intent logits
        intent = self.intent_out(last_hidden)

        slots = slots.permute(0,2,1)
        return slots, intent