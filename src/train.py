import logging
import argparse
import math

import os
import sys
from time import strftime, localtime
import random
import numpy
import time

from pytorch_transformers import BertModel

from sklearn import metrics
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, DialogueActTaggingDataset, prepare_train_test_split

from models import BERT_CRF

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

target_names = None

class Instructor:
    def __init__(self, opt, fold):
        self.opt = opt
        self.fold = fold

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_files['train'], opt.dataset_files['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_fold_{1}_tokenizer.dat'.format(opt.dataset, fold))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_fold_{2}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset, fold))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        self.trainset = DialogueActTaggingDataset(opt.dataset_file['train'], tokenizer, opt)
        self.valset = DialogueActTaggingDataset(opt.dataset_file['dev'], tokenizer, opt)
        self.testset = DialogueActTaggingDataset(opt.dataset_file['test'], tokenizer, opt)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()
    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        n_batch = len(train_data_loader) - 1

        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                if i_batch == n_batch:
                    break
                global_step += 1
                
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                log_ll, emission = self.model(inputs)
                targets = sample_batched['dialogue_tags'].to(self.opt.device).squeeze(0)

                loss = -log_ll
                loss.backward()
                optimizer.step()

                outputs = self.model.crf.decode(emission)
                outputs = torch.LongTensor(outputs).to(self.opt.device)
                
                n_correct += (outputs == targets).sum().item()
                n_total += outputs.size()[0] * outputs.size()[1]
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1 = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_fold_{2}'.format(self.opt.model_name, self.opt.dataset, self.fold)
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
                
        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['dialogue_tags'].to(self.opt.device).squeeze(0)
                _, t_emission = self.model(t_inputs)

                t_outputs = self.model.crf.decode(t_emission)
                t_outputs = torch.LongTensor(t_outputs).to(self.opt.device)

                n_correct += (t_outputs == t_targets).sum().item()
                n_total += t_outputs.size()[0] * t_outputs.size()[1]

                if t_targets_all is None:
                    t_targets_all = [tag for batch in t_targets.cpu().tolist() for tag in batch]
                    t_outputs_all = [tag for batch in t_outputs.cpu().tolist() for tag in batch]
                else:
                    t_targets_all.extend([tag for batch in t_targets.cpu().tolist() for tag in batch])
                    t_outputs_all.extend([tag for batch in t_outputs.cpu().tolist() for tag in batch])

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all, t_outputs_all, labels=[i for i in range(len(target_names))], average='macro', zero_division=0)
        logger.info(metrics.classification_report(t_targets_all, t_outputs_all, labels=[i for i in range(len(target_names))], target_names=target_names, digits=3, zero_division=0))
        return acc, f1

    def run(self):
        # Loss and Optimizer
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=1, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=1, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=1, shuffle=False)

        self._reset_params()
        best_model_path = self._train(optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_acc, test_f1 = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))

def main():
    def str2int(v):
        global target_names
        if v == 'da':
            target_names = ['Feedback', 'Commissive', 'Directive', 'Statement', 'PropQ', 'SetQ', 'ChoiceQ', 'Salutation', 'Apology', 'Thanking']
            return len(target_names)
        else:
            raise argparse.ArgumentTypeError('Invalid value! Please choose between [semantic, general, som].')
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_crf', type=str)
    parser.add_argument('--dataset', default='da', type=str, help='dialogue act dataset')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=8, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=20, type=int)
    parser.add_argument('--max_utterances', default=None, type=int)
    parser.add_argument('--task', default='da', type=lambda f: str2int(f), dest="tagset_size")
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=2021, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--data_rate', default=None, type=float)
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed()
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_classes = {
        'bert_crf': BERT_CRF
    }
    input_colses = {
        'bert_crf': ['bert_dialogue_utters', 'dialogue_tags', 'attention_mask'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    initializer = opt.initializer
    optimizer = opt.optimizer
    device = opt.device

    dial_root_dir = '../data/{}/'.format(opt.dataset)
    prepare_train_test_split(dial_root_dir, data_sources=['Switchboard'])
    
    log_file = 'logs/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))
    
    for fold in range(1, 6):
        dataset_files = {
            'train': '../data/5folds/train-fold-{}.txt'.format(fold),
            'dev': '../data/5folds/test-fold-{}.txt'.format(fold),
            'test': '../data/5folds/test-fold-{}.txt'.format(fold)
        }
        opt.model_class = model_classes[opt.model_name]
        opt.dataset_file = dataset_files
        opt.inputs_cols = input_colses[opt.model_name]

        opt.initializer = initializers[initializer]
        opt.optimizer = optimizers[optimizer]
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device is None else torch.device(device)
        
        if not os.path.exists('logs'):
            os.mkdir('logs')

        ins = Instructor(opt, fold)
        ins.run()

        gc.collect()
        del ins

if __name__ == '__main__':
    main()
