import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF

class BERT_CRF(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_CRF, self).__init__()
        self.opt = opt
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)

        self.ff = nn.Sequential(
            nn.Linear(opt.bert_dim, opt.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(opt.hidden_dim, opt.hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.logit = nn.Linear(opt.hidden_dim, opt.tagset_size)
        self.crf = CRF(opt.tagset_size, batch_first=True)

    def forward(self, inputs):
        conversation_utters = inputs[0].squeeze(0) # batch_size x num_utterances x seq_len
        targets = inputs[1].squeeze(0) # batch_size x num_utterances
        masks = inputs[2].squeeze(0) # batch_size x num_utterances x seq_len

        # unstack context utterances
        all_utters = torch.unbind(conversation_utters, dim=1) # num_utterances x (batch_size x seq_len)
        all_masks = torch.unbind(masks, dim=1) # num_utterances x (batch_size x seq_len)

        # create conversation vectors
        conversation_vectors = []
        for utter, mask in zip(all_utters, all_masks):
            _, utter_pooled = self.bert(utter, mask) # batch_size x bert_dim
            utter_pooled = self.dropout(utter_pooled) # batch_size x num_utterances x bert_dim
            conversation_vectors.append(utter_pooled) # batch_size x num_utterances x bert_dim

        # conversation layer
        conversation_vectors = torch.stack(conversation_vectors, dim=0) # num_utterances x batch_size x bert_dim
        output = torch.transpose(conversation_vectors, 0, 1) # batch_size x num_utterances x bert_dim
        output = self.dropout(output)

        output = self.ff(output)  # batch_size x num_utterances x 100
        logit = self.logit(output)
        
        loss = self.crf(logit, targets)

        return loss, logit