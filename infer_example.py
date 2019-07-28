# -*- coding: utf-8 -*-
# file: infer.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn.functional as F
import argparse
import pandas as pd
from data_utils import build_tokenizer, build_embedding_matrix
from models.bert_spc import BERT_SPC

from pytorch_pretrained_bert import BertModel
from sklearn import metrics
from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSAInfer

from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from torch.utils.data import DataLoader, random_split

class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            self.tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=self.tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    '''
    def evaluate(self, text, drug):
        context_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().strip()) for raw_text in text]
        # aspect_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().strip()) for raw_text in drug]
        aspect_seqs = [self.tokenizer.text_to_sequence('null')] * len(text)
        print(aspect_seqs)
        context_indices = torch.tensor(context_seqs, dtype=torch.int64).to(self.opt.device)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64).to(self.opt.device)

        print(context_indices.shape)
        print(aspect_indices.shape)
        t_inputs = [context_indices, aspect_indices]
        # print(t_inputs)
        # print(t_inputs.shape)
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs
    '''
    def evaluate(self, raw_texts):
        context_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().strip()) for raw_text in raw_texts]
        aspect_seqs = [self.tokenizer.text_to_sequence('null')] * len(raw_texts)
        context_indices = torch.tensor(context_seqs, dtype=torch.int64).to(self.opt.device)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64).to(self.opt.device)

        t_inputs = [context_indices, aspect_indices]
        t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_outputs_all = None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                # t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                t_outputs = torch.argmax(t_outputs, -1)
                if t_outputs_all is None:
                    t_outputs_all = t_outputs
                else:
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        return t_outputs_all


if __name__ == '__main__':
    model_classes = {
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'aoa': AOA,
        'bert_spc': BERT_SPC
    }
    # set your trained models here
    model_state_dict_paths = {
        'bert_spc':'state_dict/bert_spc_twitter_val_acc0.7121',
        'atae_lstm': 'state_dict/atae_lstm_restaurant_acc0.7786',
        'ian': 'state_dict/ian_restaurant_acc0.7911',
        'memnet': 'state_dict/memnet_restaurant_acc0.7911',
        'aoa': 'state_dict/aoa_restaurant_acc0.8063',
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen_bert': ['text_raw_bert_indices', 'aspect_bert_indices'],
    }


    class Option(object): pass
    opt = Option()
    opt.model_name = 'bert_spc'
    opt.model_class = model_classes[opt.model_name]
    opt.dataset = 'restaurant'
    opt.dataset_file = {
        'train': './datasets/semeval14/Restaurants_Train.xml.seg',
        'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
    }
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 256
    opt.polarities_dim = 3
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.dropout = 0.1
    opt.bert_dim = 768
    opt.inputs_cols = input_colses[opt.model_name]
    opt.batch_size = 32


    inf = Inferer(opt)
    # t_probs = inf.evaluate(['happy memory', 'the service is terrible', 'just normal food'])
    test_data = pd.read_csv('../test_tOlRoBf.csv')
    testset = ABSAInfer(test_data, inf.tokenizer)
    test_data_loader = DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)
    outputs = inf._evaluate_acc_f1(test_data_loader)
    print(outputs)
    test_data['sentiment'] = outputs
    print(test_data['sentiment'].value_counts())
    test_data = test_data.drop(['text', 'drug'], axis = 1)
    test_data.to_csv('result1.csv', index = False)
