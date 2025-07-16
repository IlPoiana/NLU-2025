import torch.nn as nn

#Original papaer adapted version
class VariationalDropout(nn.Module):
    """
    VariationalDropout: apply the same dropout mask to each temporal istance
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        # check if I am in inference or the dropout has not been set
        if not self.training or not dropout:
            return x
        # print("data: ",x.data) #remeber is a batch! 64 * seuquence_l * 512 
        m = x.data.new(x.size(2)).bernoulli_(1 - dropout) # the same mask for all the sample in the batch
        # print(m.shape)
        # wrapper for the autograd computation of the mask
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1,n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's LSTM layer: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.lstm(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output

    
class LSTM_WeightTying(nn.Module):
    """
    *In other words, by reusing the embedding matrix in the output projection layer (with a transpose)
    and letting the neural network do the necessary linear mapping h → Ah, we get the same result as
    we would have in the first place.*
    """
    
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1,n_layers=1):
        super(LSTM_WeightTying, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's LSTM layer: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size, bias= False) # Bias False because  we don't want two different bias parameters matrices learned.
        #weight tying
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.lstm(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output
    
class LSTM_VariationalDropout(nn.Module):
    """
    Incremental version, Weight Tying + Variational Dropout
    Three dropouts:
    1. Embeddings to the LSTM
    2. From Hidden to Hidden state in the LSTM
    3. From the hidden to the output
    """
    
    def __init__(self, emb_size, hidden_size, output_size, emb_dropout, out_dropout,pad_index=0, n_layers=1):
        super(LSTM_VariationalDropout, self).__init__()
        #Dropout 1
        self.emb_dropout = emb_dropout
        self.dropout1 = VariationalDropout()
        #Dropout 3
        self.out_dropout = out_dropout
        self.dropout3 = VariationalDropout()

        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's LSTM layer: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.output = nn.Linear(hidden_size, output_size, bias= False) # Bias False because  we don't want two different bias parameters matrices learned.
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        #missing the hidden to hidden
        emb = self.embedding(input_sequence)
        #first dropout after embedding layer
        drop1_emb = self.dropout1(emb, self.emb_dropout)
        rnn_out, _  = self.lstm(drop1_emb)
        #second dropout before the output layer
        drop3_hidden = self.dropout3(rnn_out, self.out_dropout)
        output = self.output(drop3_hidden).permute(0,2,1)
        return output
    
