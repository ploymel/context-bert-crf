from flask import Flask, json, request, jsonify
import re
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, DialogueActTaggingDataset
from models import BERT_CRF
from pytorch_transformers import BertModel

import argparse
import torch

def preprocess_sentence(sent):
    sent = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sent))
    sent = re.sub(r"[ ]+", " ", sent)
    sent = re.sub(r"\!+", "!", sent)
    sent = re.sub(r"\,+", ",", sent)
    sent = re.sub(r"\?+", "?", sent)
    return sent.lower()

def to_tag(idx):
        ALL_CF = ['Feedback', 'Commissive', 'Directive', 'Statement', 'PropQ', 'SetQ', 'ChoiceQ', 'Salutation', 'Apology', 'Thanking']
        return ALL_CF[idx]

def prepare_data(file, tokenizer):
    conversation = file.read().decode("utf-8").split('\n')
    messages = [utter for utter in conversation if utter != '']
    # tags = [utter.split(',')[1] for utter in conversation if utter != '']

    conversation = []
    for sent in messages:
        conversation.append({'sentence': sent})

    dial_sent = []
    dial_tag = []
    bert_dial_sent = []
    bert_dial_mask = []
    for utt in conversation:
        sent = utt['sentence']
        # tag = utt['tagging']

        # preprocess data
        sent = preprocess_sentence(sent)
        if sent == '':
            continue
        
        sent_idx = tokenizer.text_to_sequence(sent)
        bert_sent_idx = tokenizer.text_to_sequence("[CLS] " + sent + " [SEP]")
        tag_idx = 0
        attention_mask = []
        for bert_i in bert_sent_idx:
            if bert_i == 0:
                attention_mask.append(0)
            else:
                attention_mask.append(1)

        dial_sent.append(sent_idx)
        dial_tag.append(tag_idx)
        bert_dial_sent.append(bert_sent_idx)
        bert_dial_mask.append(attention_mask)

    data = {
        'dialogue_utters': dial_sent,
        'dialogue_tags': dial_tag,
        'dialogue_size': len(dial_sent),
        'attention_mask': bert_dial_mask,
        'bert_dialogue_utters': bert_dial_sent
    }

    return data

app = Flask(__name__)

# tokenizer = build_tokenizer(
#     fnames=['../data/5folds/train-tokenize-fold-1.txt', '../data/5folds/test-tokenize-fold-1.txt'],
#     max_seq_len=20,
#     dat_fname='out/tokenize/swda+maptask+oasis_fold_1_tokenizer.dat')
# embedding_matrix = build_embedding_matrix(
#     word2idx=tokenizer.word2idx,
#     embed_dim=300,
#     dat_fname='out/embedding/300_swda+maptask+oasis_fold_1_embedding_matrix.dat')

# parser = argparse.ArgumentParser()
# parser.add_argument('--dropout', default=0.2, type=float)
# parser.add_argument('--l2reg', default=0.01, type=float)
# parser.add_argument('--embed_dim', default=300, type=int)
# parser.add_argument('--hidden_dim', default=300, type=int)
# parser.add_argument('--max_seq_len', default=20, type=int)
# parser.add_argument('--max_utterances', default=None, type=int)
# parser.add_argument('--tagset_size', default=10, type=int)
# parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')


class Opt:
    def __init__(self):
        self.dropout = 0.2
        self.embed_dim = 300
        self.hidden_dim = 300
        self.max_seq_len = 20
        self.max_utterances = None
        self.tagset_size = 10
        self.device = 'cuda:0'
        self.pretrained_bert_name = 'bert-base-uncased'

opt = Opt()

tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
bert = BertModel.from_pretrained(opt.pretrained_bert_name)
model = BERT_CRF(bert, opt).to(opt.device)
# model = BiLSTM_CRF(embedding_matrix, opt).to(opt.device)
model.load_state_dict(torch.load('state_dict/bert_crf'))
model.eval()
torch.autograd.set_grad_enabled(False)

@app.route('/')
def hello():
    return 'Test!'

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get csv file from the request
        file = request.files['file']
        # convert to bytes
        data = prepare_data(file, tokenizer)
        utters = torch.tensor([[data['bert_dialogue_utters']]], dtype=torch.int64).to(opt.device)
        tags = torch.tensor([[data['dialogue_tags']]], dtype=torch.int64).to(opt.device)
        masks = torch.tensor([[data['attention_mask']]], dtype=torch.int64).to(opt.device)
        inputs = [utters, tags, masks]

        _, emission = model(inputs)

        outputs = model.crf.decode(emission)
        l_outputs = [tag for batch in outputs for tag in batch][-1]

        return jsonify({'tags': to_tag(l_outputs)})
