# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from pytorch_transformers import BertTokenizer
from gensim.models.keyedvectors import KeyedVectors
from tqdm import trange
import random
import pandas as pd
import sys
import logging
from sklearn.model_selection import KFold
import math

logging.getLogger().setLevel(logging.INFO)

def prepare_train_test_split(root_dir, data_sources=['Switchboard', 'MapTask', 'Oasis']):
    kf = KFold(n_splits=5)
    train_files = {1: [], 2: [], 3: [], 4:[], 5: []}
    test_files = {1: [], 2: [], 3: [], 4:[], 5: []}

    for source in data_sources:
        data_dir = root_dir + source + '/'
        files = os.listdir(data_dir)
        logging.info('preparing training and testing split: from -> {} to output folder -> data/5folds/'.format(data_dir))

        fold = 1
        for train_indices, test_indices in kf.split(files):
            train = [data_dir + files[idx] for idx in train_indices]
            test = [data_dir + files[idx] for idx in test_indices]

            train_files[fold].extend(train)
            test_files[fold].extend(test)

            fold += 1
    
    if not os.path.isdir('../data/5folds/'):
        os.mkdir('../data/5folds/')
        
    for fold in range(1, 6):       
        with open('../data/5folds/train-fold-{}.txt'.format(fold), 'w') as file:
            for fname in train_files[fold]:
                file.write(fname + '\n')
        with open('../data/5folds/test-fold-{}.txt'.format(fold), 'w') as file:
            for fname in test_files[fold]:
                file.write(fname + '\n')
                
def build_tokenizer(fnames, max_seq_len, dat_fname):
    """
    args: fnames - 5 folds file names
          max_seq_len - max len of each utterance
          dat_fname - tokenizer file (.dat)
    """
    if os.path.exists(dat_fname):
        logging.info('loading tokenizer: {}'.format(dat_fname))
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        logging.info('building tokenizer: output file -> {}'.format(dat_fname))
        text = ''
        for fname in fnames:
            data = []
            with open(fname, 'r') as txt_file:
                dial_files_list = [f.replace('\n', '') for f in txt_file.readlines()]
            for dial_file in dial_files_list:
                try:
                    df = pd.read_csv(dial_file, header=None)
                    for sent in df[0]:
                        data.append(sent)
                except:
                    continue
            for sent in data:
                text += preprocess_sentence(sent) + ' '
        
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer

def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        logging.info('loading embedding_matrix: output file -> {}'.format(dat_fname))
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else './glove.42B.300d.txt'
        logging.info('loading word vectors: from -> {}'.format(fname))
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        logging.info('building embedding_matrix: output file -> {}'.format(dat_fname))
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x

def preprocess_sentence(sent):
    sent = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sent))
    sent = re.sub(r"[ ]+", " ", sent)
    sent = re.sub(r"\!+", "!", sent)
    sent = re.sub(r"\,+", ",", sent)
    sent = re.sub(r"\?+", "?", sent)
    return sent.lower()

class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)
    
class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

class DialogueActTaggingDataset(Dataset):
    """Dialogue Act Tangging Dataset"""
    def __init__(self, fname, tokenizer, opt):
        logging.info('preprocessing data: from -> {}'.format(fname))
        self.opt = opt

        data_raw = []
        with open(fname, 'r') as txt_file:
            dial_files_list = [f.replace('\n', '') for f in txt_file.readlines()]
            if opt.data_rate is not None and 'train' in fname:
                max_data = int(len(dial_files_list) * opt.data_rate)
                dial_files_list = dial_files_list[:max_data]
            for dial_file in dial_files_list:
                try:
                    df = pd.read_csv(dial_file, header=None)
                    df = df[[0, 1]]
                    dial_data = []
                    for sent, tag in zip(df[0], df[1]):
                        dial_data.append({'sentence': sent, 'tagging': tag})
                    data_raw.append(dial_data)
                except:
                    continue

        all_data = []
        for idx in range(0, len(data_raw)):
            dial_sent = []
            bert_dial_sent = []
            bert_dial_mask = []
            dial_tag = []
            count_utt = 1
            eval_tag = []
            for utt in data_raw[idx]:
                sent = utt['sentence']
                tag = utt['tagging']
                
                # preprocess data
                sent = preprocess_sentence(sent)
                if sent == '':
                    continue
                sent_idx = tokenizer.text_to_sequence(sent)
                bert_sent_idx = tokenizer.text_to_sequence("[CLS] " + sent + " [SEP]")
                tag_idx = self._to_label(tag)
                attention_mask = []
                for bert_i in bert_sent_idx:
                    if bert_i == 0:
                        attention_mask.append(0)
                    else:
                        attention_mask.append(1)

                dial_sent.append(sent_idx)
                bert_dial_sent.append(bert_sent_idx)
                dial_tag.append(tag_idx)
                eval_tag.append(0)
                bert_dial_mask.append(attention_mask)

                if opt.max_utterances is not None:
                    if count_utt == opt.max_utterances: 
                        data = {
                            'dialogue_utters': dial_sent,
                            'bert_dialogue_utters': bert_dial_sent,
                            'dialogue_tags': dial_tag,
                            'eval_tags': eval_tag,
                            'dialogue_size': len(dial_sent),
                            'attention_mask': bert_dial_mask
                        }
                        all_data.append(data)
                        
                        dial_sent = []
                        bert_dial_sent = []
                        bert_dial_mask = []
                        dial_tag = []
                        dial_dimension = []
                        count_utt = 0
                        eval_tag = []
                
                count_utt += 1
            if len(dial_sent) > 0:
                data = {
                    'dialogue_utters': dial_sent,
                    'bert_dialogue_utters': bert_dial_sent,
                    'dialogue_tags': dial_tag,
                    'eval_tags': eval_tag,
                    'dialogue_size': len(dial_sent),
                    'attention_mask': bert_dial_mask
                }
                all_data.append(data)

        grouped_data = {}
        self.max_batch_size = opt.batch_size
        
        # group data according to dialogue's utterances length
        for idx in range(len(all_data)):
            dial_size = all_data[idx]['dialogue_size']
            if dial_size not in grouped_data.keys():
                grouped_data[dial_size] = [all_data[idx]]
            else:
                grouped_data[dial_size].append(all_data[idx])

        self.data = []
        for key in grouped_data.keys():
            if len(grouped_data[key]) > opt.batch_size:
                extend = []
                num_split_group = math.ceil(len(grouped_data[key])/opt.batch_size)
                for i in range(num_split_group):
                    extend.append(grouped_data[key][i*opt.batch_size:opt.batch_size*(i+1)])
                self.data.extend(extend)
            else:
                self.data.append(grouped_data[key])

    def _to_label(self, tag):
        DA_TAGS = ['Feedback', 'Commissive', 'Directive', 'Statement', 'PropQ', 'SetQ', 'ChoiceQ', 'Salutation', 'Apology', 'Thanking']
        return DA_TAGS.index(tag)
            
    def __getitem__(self, index):
        data = self.data[index]
        stacked_data = {}
        for item in data:
            for key in item.keys():
                if key not in stacked_data.keys():
                    stacked_data[key] = [item[key]]
                else:
                    stacked_data[key].append(item[key])

        for key in stacked_data.keys():
            stacked_data[key] = torch.LongTensor(stacked_data[key])

        return stacked_data

    def __len__(self):
        return len(self.data)    