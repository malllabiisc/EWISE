import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WSD_BiLSTM(nn.Module):
    """BiLSTM WSD model with attention """
    @staticmethod
    def getDefaultArgs():
        kwargs = {
            'n_hidden':1024, #in one direction
            'n_layers':2, #rnn layers
            'rnn_type':'LSTM', #LSTM, GRU
            'dropout':0,
        }
        return kwargs

    def __init__(self, kwargs):
        super(WSD_BiLSTM, self).__init__()
        self.n_hidden = kwargs['n_hidden']
        self.n_layers = kwargs['n_layers']
        self.rnn_type = kwargs['rnn_type']
        self.dropout = kwargs['dropout']
    
        if 'input_emb_matrix' not in kwargs:
            raise ValueError('Provide pretrained input embeddings if not training')
        else:
            self.input_emb_matrix = kwargs['input_emb_matrix']
            self.n_input_token = self.input_emb_matrix.size(0)
            self.n_input_emb = self.input_emb_matrix.size(1)

        #Encoder
        self.encoder = nn.Embedding(self.n_input_token, self.n_input_emb)
        self.encoder.weight = nn.Parameter(self.input_emb_matrix)
        self.encoder.weight.requires_grad = False

        #RNN
        self.rnn = getattr(nn, self.rnn_type)(self.n_input_emb, self.n_hidden,
                                self.n_layers, dropout=self.dropout, bidirectional=True)
        #Attention
        attention_size = self.n_hidden*2
        self.attention_query = nn.Linear(self.n_hidden*2, attention_size)
        self.attention_key = nn.Linear(self.n_hidden*2, attention_size)
        self.attention_value = nn.Linear(self.n_hidden*2, attention_size)
        self.scale_factor = np.sqrt(attention_size)

        #Decoder
        dinput = self.n_hidden*2 + attention_size
        if 'output_emb_matrix' in kwargs:
            self.output_emb_matrix = kwargs['output_emb_matrix']
            self.n_output_token = self.output_emb_matrix.size(0)
            self.n_output_emb = self.output_emb_matrix.size(1)
        else:
            self.n_output_emb = kwargs['n_output_emb']
            self.n_output_token = kwargs['n_output_token']
        self.context2sense = nn.Linear(dinput, self.n_output_emb, bias=False)
        self.decoder = nn.Linear(self.n_output_emb, self.n_output_token, bias=False)
        self.decoder_bias = nn.Linear(self.n_output_emb, 1, bias=False)

        self.b_fix_output_embedding = False
        if 'output_emb_matrix' in kwargs:
            self.decoder.weight = nn.Parameter(self.output_emb_matrix)
            self.decoder.weight.requires_grad = False
            self.b_fix_output_embedding = True

        self.drop = nn.Dropout(self.dropout)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.context2sense.weight.data.uniform_(-initrange, initrange)

        self.decoder_bias.weight.data.uniform_(-initrange, initrange)
        if not self.b_fix_output_embedding:
            self.decoder.weight.data.uniform_(-initrange, initrange)

        self.attention_query.bias.data.fill_(0)
        self.attention_query.weight.data.uniform_(-initrange, initrange)    
        self.attention_key.bias.data.fill_(0)
        self.attention_key.weight.data.uniform_(-initrange, initrange)    
        self.attention_value.bias.data.fill_(0)
        self.attention_value.weight.data.uniform_(-initrange, initrange)   

    def forward(self, x): #x N*bsz
        #Encoder
        emb = self.drop(self.encoder(x)) #N*bsz*n_input_emb
        output, hidden = self.rnn(emb) #output N*bsz*n_hiddenXnD
        output = self.drop(output)

        #Attention module
        attention_input = output
        q = self.attention_query(attention_input) #N*bsz*n_hiddenXnD
        k = self.attention_key(attention_input) #N*bsz*n_hiddenXnD
        v = self.attention_value(attention_input) #N*bsz*n_hiddenXnD
        u = torch.bmm(q.permute(1, 0, 2), k.permute(1, 2, 0)) #bsz*N1*N2
        u = u / self.scale_factor
        a = F.softmax(u, 2)
        #bsz*N*N * bsz*N*n_hiddenXnD -> bsz*N*n_hiddenXnD -> N*bsz*n_hiddenXnD
        c = torch.bmm(a, v.permute(1,0,2)).permute(1,0,2)
        output = torch.cat([output, c], 2) #N*bsz*n_hiddenXnDX2
   
        #Decoder
        output = self.drop(self.context2sense(output))
        #N*bsz*n_hiddenXnDX1/2
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

        bias = self.decoder_bias(self.decoder.weight).squeeze(-1).unsqueeze(0).unsqueeze(0)
        decoded = decoded + bias
        return decoded
