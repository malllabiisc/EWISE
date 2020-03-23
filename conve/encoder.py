import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, zeros_

class DefinitionEncoder(nn.Module):
    def __init__(self): 
        super(DefinitionEncoder, self).__init__()
        ndim = 2048
        self.enc_lstm = nn.LSTM(300, ndim, 1, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.project = nn.Linear(ndim*2, 200)

    def init(self):
        xavier_normal_(self.project.weight)
        zeros_(self.project.bias)

    def forward(self, sent_tuple):
        sent, sent_len = sent_tuple

        #padding logic from https://github.com/facebookresearch/InferSent
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)
        idx_sort = torch.from_numpy(idx_sort).to(sent.device)
        sent = sent.index_select(1, idx_sort)

        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0] 
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        idx_unsort = torch.from_numpy(idx_unsort).to(sent.device)
        sent_output = sent_output.index_select(1, idx_unsort)

        sent_output[sent_output == 0] = -1e9
        emb = torch.max(sent_output, 0)[0]

        emb_proj = self.dropout(emb)
        emb_proj = self.project(emb_proj)

        return emb_proj, emb
