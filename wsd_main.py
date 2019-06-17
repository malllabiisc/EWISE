# coding: utf-8
import sys
import os
from os import path
import time 
from itertools import ifilter
import argparse
import json
import numpy as np
import math
import pickle
import random
import subprocess
import signal
from sklearn.preprocessing import Imputer

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from wsd_model import *
import mfs
from batcher import batcher

from nltk.corpus import wordnet as wn

pos_dict = {'NOUN':'n', 'PROPN':'n', 'VERB':'v', 'AUX':'v', 'ADJ':'a', 'ADV':'r'}

parser = argparse.ArgumentParser(description='Train/Test WSD')
parser.add_argument('--seed', type=int, default=1,
                    help='seed for numpy')
parser.add_argument('--train_ratio', type=float, default=1.0,
                    help='ratio of training data to use')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpuid', type=int, default=0,
                    help='use gpu_id')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='droput probability')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--lr', type=float, default=.1,
                    help='initial learning rate')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='Number of epochs')
parser.add_argument('--input_directory', type=str, default='',
                    help='Path to training and test files ... <add more detail here>')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--predict_on_unseen', action='store_true',
                    help='consider unseen senses during prediction')
parser.add_argument('--abstain_on_unseen', action='store_true',
                    help='dont output anything for unseen words')
parser.add_argument('--scorer', type=str,  default='./',
                    help='path to scorer executable')
parser.add_argument('--evaluate', action='store_true',
                    help='only evaluate')
parser.add_argument('--pretrained', type=str,  default='',
                    help='pretrained model')
parser.add_argument('--output_embedding', type=str,  default='',
                    help='custom-<filename>')
parser.add_argument('--enc_lstm_dim', type=int, default=1024, metavar='N',
                    help='LSTM dim in 1 direction')
parser.add_argument('--output_embedding_size', type=int, default=512, metavar='N',
                    help='output embedding size')
parser.add_argument('--train', type=str,  default='semcor',
                    help='train file')
parser.add_argument('--val', type=str,  default='semeval2007',
                    help='val file name')
parser.add_argument('--test_file', type=str,  default='',
                    help='custom test file name')

def init_random(seed):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

def pred_idx_to_key(pred_idx):
    if pred_idx in o_id_to_o_token:
        pred_key = o_id_to_o_token[pred_idx]
    else: #pred_idx in o_id_remainingWordNet_to_o_token:
        pred_key = o_id_remainingWordNet_to_o_token[pred_idx]
    return pred_key

def predict_from_list(cand_list, output):
    #return top candidate, or (cand, prob) pairs for all elements
    prob = []
    for item in cand_list:
        prob.append(output[item])
    sorted_cand_pred = [(c,p) for p,c in sorted(zip(prob,cand_list), key=lambda pair: pair[0])][::-1]
    pred_idx = sorted_cand_pred[0][0]
    return pred_idx

def predict_on_batch(batch, output, n_input_tokens, i_id_to_candidate_wn_o_id, i_id_to_candidate_train_o_id, file_out):
    x,y,a,tag,stem,POStag  = batch
    bsz, seqlen = x.shape[0], x.shape[1]
    x, y = x.T, y.T
    output = output.data.cpu().numpy()
    for i in range(bsz):
        for pos_idx, pos in enumerate(a[i]):
            pred_idx = -1
            xidx, yidx, stemidx, POStagidx, outputidx = x[pos,i], y[pos,i], stem[i][pos_idx], POStag[i][pos_idx], output[pos,i,:]
            candidx = list(i_id_to_candidate_wn_o_id[(stemidx, pos_dict[POStagidx])])
            candidx_seen = list(i_id_to_candidate_train_o_id[(stemidx, pos_dict[POStagidx])])
            candidx = [c-n_input_tokens for c in candidx]
            candidx_seen = [c-n_input_tokens for c in candidx_seen]

            if not args.predict_on_unseen:
                if len(candidx_seen) > 0:
                    pred_idx = predict_from_list(candidx_seen, outputidx)
                elif not args.abstain_on_unseen:
                    if len(candidx) == 0:
                        continue
                    pred_idx = candidx[0] #Back-off, WNs1
            else: #Predict on unseen
                candidx_all = list(set(candidx+candidx_seen))
                if len(candidx_all) == 0:
                    continue
                pred_idx = predict_from_list(candidx_all, outputidx)
            
            pred_tag = tag[i] + '.t' + '{:03}'.format(pos_idx)
            if pred_idx >= 0:
                pred_idx += n_input_tokens
                pred_key = pred_idx_to_key(pred_idx) 
                #write
                file_out.write('{} {}\n'.format(pred_tag, pred_key))

def evaluate_output(gold_file_prefix, out_file):
    eval_cmd = ['java','-cp', os.getcwd(), 'Scorer', gold_file_prefix +'.gold.key.txt', out_file]
    output = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE ).communicate()[0]
    output = output.splitlines()
    p,r,f1 =  [float(output[i].split('=')[-1].strip()[:-1]) for i in range(3)]
    return p, r, f1

def train_test(model, batches, n_output_tokens, n_output_train_tokens, n_input_tokens,
                i_id_to_candidate_wn_o_id, i_id_to_candidate_train_o_id,
                optimizer=None, lr=None, criterion=None, epoch=0,
                test_pred_filename="", gold_file_prefix="", test=False):
    if test:
        model.eval()
        batch_order = range(len(batches))
        file_pred = open(test_pred_filename, 'w')
    else:
        model.train()
        batch_order = np.random.randint(0, len(batches), size=(len(batches)))
        total_loss = 0
        total_size = 0

    start_time = time.time()

    for i, batch_idx in enumerate(batch_order):
        x,y,a,tag,stem,POStag = batches[batch_idx]
        bsz, seqlen = x.shape[0], x.shape[1]

        y = y - n_input_tokens
        y[y<0] = n_output_tokens - n_input_tokens
        y[y>=n_output_train_tokens-n_input_tokens]  = n_output_tokens - n_input_tokens

        x, y = x.T, y.T
        x = torch.LongTensor(x).cuda(args.gpuid)
        y = torch.LongTensor(y).cuda(args.gpuid)
        
        if not test:
            optimizer.zero_grad()

        output = model(x)

        if not test:
            y = y.view(-1)
            mask = y<n_output_train_tokens-n_input_tokens
            y_masked = y[mask]
            if y_masked.size(0) == 0:
                continue
            output = output.view(-1, n_output_tokens-n_input_tokens+1)[:, :n_output_train_tokens-n_input_tokens]
            output_masked = torch.nn.functional.embedding(mask.nonzero().view(-1), output) 
            loss = criterion(output_masked, y_masked)

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            total_loss += loss.data * bsz
            total_size += bsz
        else:
            predict_on_batch(batches[batch_idx], output, n_input_tokens, i_id_to_candidate_wn_o_id, i_id_to_candidate_train_o_id, file_pred)

        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / total_size
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i, len(batches), lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            total_size = 0
            start_time = time.time()

    if test:
        file_pred.close()
        p, r, f1 = evaluate_output(gold_file_prefix, test_pred_filename)
        return p, r, f1

#code starts here
args = parser.parse_args()
print (args)
if args.seed != -1:
    init_random(args.seed)

test_files = ['senseval2', 'senseval3', 'semeval2013', 'semeval2015', 'ALL', 'semeval2007']
d = args.input_directory
bsz = args.batch_size

#Read input files
print ('Reading training and evaluation files from {}'.format(d))
i_id_to_i_token = pickle.load(open(path.join(d, 'i_id_to_i_token.pickle')))
o_id_to_o_token = pickle.load(open(path.join(d, 'o_id_to_o_token.pickle')))
o_id_remainingWordNet_to_o_token = pickle.load(open(path.join(d, 'o_id_remainingWordNet_to_o_token.pickle')))
i_id_to_candidate_wn_o_id = pickle.load(open(path.join(d, 'i_id_to_candidate_wn_o_id.pickle')))
i_id_to_candidate_train_o_id = pickle.load(open(path.join(d, 'i_id_to_candidate_train_o_id.pickle')))
i_id_embedding = pickle.load(open(path.join(d, 'i_id_embedding_glove.pickle')))

if args.output_embedding != "":
    if args.output_embedding.split('-')[0] == 'custom':
        fname = args.output_embedding.split('-')[1]
        o_id_embedding = pickle.load(open(path.join(d, fname)))
    elif args.output_embedding.split('-')[0] == 'customnpz':
        fname = args.output_embedding.split('-')[1]
        o_id_embedding = np.load(path.join(d, fname))['embeddings']
    elif args.output_embedding.split('-')[0] == 'customimpnpz':
        fname = args.output_embedding.split('-')[1]
        o_id_embedding = np.load(path.join(d, fname))['embeddings']
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        o_id_embedding = imp.fit_transform(o_id_embedding)        
    elif args.output_embedding.split('-')[0] == 'customimp':
        fname = args.output_embedding.split('-')[1]
        o_id_embedding = pickle.load(open(path.join(d, fname)))
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        o_id_embedding = imp.fit_transform(o_id_embedding)        

train_batches = batcher(path.join(d, '{}_indexed.json'.format(args.train)), bsz)
if args.train_ratio < 1.0:
    np.random.shuffle(train_batches)
    len_train = len(train_batches)
    updated_len = int(args.train_ratio * len_train)
    train_batches = train_batches[:updated_len]
    print ('Reducing training batches count from {} to {}'.format(len_train, len(train_batches)))

val_batches = batcher(path.join(d, '{}_indexed.json'.format(args.val)), bsz)
test_batches = {}
for t in test_files:
    test_batches[t] = batcher(path.join(d, '{}_indexed.json'.format(t)), bsz)

n_input_tokens = len(i_id_to_i_token)
n_output_tokens = len(o_id_to_o_token) + n_input_tokens
n_additional_tokens = len(o_id_remainingWordNet_to_o_token)

kwargs = WSD_BiLSTM.getDefaultArgs()
kwargs['dropout'] = args.dropout
kwargs['input_emb_matrix'] = torch.FloatTensor(i_id_embedding)
kwargs['n_hidden'] = args.enc_lstm_dim

n_output_train_tokens = n_output_tokens
if args.output_embedding != "":
    o_id_embedding_train = o_id_embedding[:n_output_tokens,:]
    if args.predict_on_unseen:
        o_emb = o_id_embedding
        n_output_tokens = n_output_tokens + n_additional_tokens
    else:
        o_emb = o_id_embedding_train
    o_emb = np.append(o_emb[n_input_tokens:n_output_tokens, :], np.zeros((1,o_emb.shape[1])), axis=0)
    kwargs['output_emb_matrix'] = torch.FloatTensor(o_emb)
else:
    kwargs['n_output_emb'] =  args.output_embedding_size
    kwargs['n_output_tokens'] = n_output_tokens - n_input_tokens + 1
print ('kwargs', kwargs)

weights = np.zeros(n_output_tokens-n_input_tokens+1)
weights[n_output_train_tokens-n_input_tokens:] = 0
_, _, lemma_freq = mfs.build_mfs(train_batches, use_stem=True, return_lemma_freq=True)
for y in lemma_freq:
     weights[y-n_input_tokens] = 1.0/lemma_freq[y]

min_weight = np.min(weights[:n_output_train_tokens-n_input_tokens])
weights = np.clip(weights, None, min_weight*100.0)
sum_weight = np.sum(weights[:n_output_train_tokens-n_input_tokens])
print (min_weight, sum_weight)
weights = weights/sum_weight

weights = weights[:n_output_train_tokens-n_input_tokens]
weights = torch.FloatTensor(weights)
if args.cuda:
    weights = weights.cuda(args.gpuid)
criterion = nn.CrossEntropyLoss(weight=weights)

model = WSD_BiLSTM(kwargs)
#load if model is provided
if args.pretrained != '':
    print ("Loading model dict from {}".format(args.pretrained))
    model.load_state_dict(torch.load(args.pretrained))

if args.cuda:
        model = model.cuda(args.gpuid)
print (model)

parameters = ifilter(lambda p: p.requires_grad, model.parameters())
lr = args.lr
optimizer = optim.Adam(parameters, lr=lr)

def display_results():
    val_p, val_r, val_f1 = train_test(model, val_batches, n_output_tokens, n_output_train_tokens, n_input_tokens,
                    i_id_to_candidate_wn_o_id, i_id_to_candidate_train_o_id,
                    test_pred_filename=args.save+'_' + args.val + '_pred.key', gold_file_prefix=path.join(d, args.val), test=True)
    print('val P {:5.4f} | R {:5.4f} | F1 {:5.4f}'.format(val_p, val_r, val_f1))
    for t in test_files:
        test_p, test_r, test_f1 = train_test(model, test_batches[t], n_output_tokens, n_output_train_tokens, n_input_tokens,
                        i_id_to_candidate_wn_o_id, i_id_to_candidate_train_o_id,
                        test_pred_filename=args.save+ '_' + t + '_pred.key', gold_file_prefix=path.join(d, t), test=True)
        print('{:15} | test P {:5.4f} | R {:5.4f} | F1 {:5.4f}'.format(t, test_p, test_r, test_f1))
    return val_p, val_r, val_f1

def display_exit(signal, frame):
    for epoch, lr, val_f1 in stats:
        print('E {:3d} |  val F1 {:5.4f}'.format(epoch, val_f1))
    model.load_state_dict(torch.load(args.save))
    display_results()
    sys.exit(0)

if args.evaluate:
    display_results()
    exit(0)

epochs = args.epochs
signal.signal(signal.SIGINT, display_exit)

best_val_metric = None
best_sel_metric = None
stats = []
for epoch in range(epochs):
    train_test(model, train_batches, n_output_tokens, n_output_train_tokens, n_input_tokens,
                    i_id_to_candidate_wn_o_id, i_id_to_candidate_train_o_id,
                    optimizer=optimizer, lr=lr, criterion=criterion, epoch=epoch)
    print('-'*80)
    print('End of epoch {}'.format(epoch))
    val_p, val_r, val_f1 = display_results()
    print('-'*80)

    if not best_sel_metric or val_f1 > best_sel_metric:
        best_sel_metric = val_f1
        best_model = epoch
        torch.save(model.state_dict(), args.save)
    print ('Best epoch {} Best metric {}'.format(best_model, best_sel_metric))
    stats.append((epoch, lr, val_f1))

display_exit(None, None)
