#    This code builds on the AWD-LSTM codebase
#    (https://github.com/salesforce/awd-lstm-lm).
#
#    groc is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License version 3 as
#    published by the Free Software Foundation.
#
#    groc is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with groc. If not, see http://www.gnu.org/licenses/

import argparse
import os, shutil
import hashlib
import time
import math
import numpy as np
import torch
import torch.nn as nn
import data
import model
from utils import batchify, get_batch, repackage_hidden, get_external_knowledge
import sys
import random
import pickle

from collections import deque

import IPython as ipy

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--test_data', type=str, default='data/penn/',
                    help='location of the test data corpus')
parser.add_argument('--save', type=str,
                    help='path from which to load the model')
parser.add_argument('--cuda', action="store_true", default=False,
                    help='use CUDA device')
parser.add_argument('--cuda_device', type=int, default=-1,
                    help='set CUDA device')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--test_batch_size', type=int, default=64,
                    help='test batch size')
parser.add_argument('--adapt_method', default="change_vocab",
                    help='method to adapt to new vocabulary')
parser.add_argument('--lam', type=float, default=0.9,
                    help='interpolation weight for the cache distribution')
parser.add_argument('--lamu', type=float, default=0.99,
                    help='interpolation weight for the uniform distribution')
parser.add_argument('--theta', type=float, default=0.24,
                    help='flattening parameter for neural cache, between 0 and 1')
parser.add_argument('--cache_size', type=int, default=10000,
                    help='length of history to cache')
parser.add_argument('--global_norm', action="store_true",
                    help='use global normalization')
parser.add_argument('--alpha', type=float, default=2.0,
                    help='weight for global normalization')
parser.add_argument('--hyp_search', type=str, default=None,
                    help='search over ranges for various hyperparams (edit file to specify values)')
parser.add_argument('--downweight_oov', type=float, default=-1.0,
                    help='weight for new words in test vocab')
args = parser.parse_args()

interpolate_methods = ["interpolate_uniform", "interpolate_unigram", "interpolate_neural"]

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        logging("WARNING: You have a CUDA device, so you should probably run with --cuda and --cuda_device [device_id]")
    else:
        torch.cuda.set_device(int(args.cuda_device))
        torch.cuda.manual_seed(args.seed)

def logging(s, print_=True, log_=True):
    print(s, file=sys.stderr)
    if log_:
        with open(os.path.join(args.save, 'eval_log.txt'), 'a+') as f_log:
            f_log.write(str(s) + '\n')

def log_interpolate(d1, d2, weight=0.5):
    """
    Interpolates the two distributions in log space. Uses torch.logsumexp
    just in case, even though we're only adding pairs of probabilities,
    because some of the probabilities in question can be quite small to
    begin with!

    Inputs:
      d1: Tensor, requires_grad=True, size=(batch_size, vocab_size)
      d2: Tensor, size=(batch_size, vocab_size)
      weight: int. the interpolation weight (relative weight of d1 and d2)
        makes the resulting distribution sum to 1 and not 2 :)

    Returns:
      A Tensor of size (batch_size, vocab_size) representing the interpolation of
      d2 with each of the distributions for the batch instances in d1
    """
    batch_size, vocab_size = d1.size()
    lse_tensor = torch.cat((d1.view(batch_size,vocab_size,1) + np.log(weight),
                            d2.view(batch_size,vocab_size,1) + np.log(1-weight))
                           ,2)
    return torch.logsumexp(lse_tensor,2)


def model_load(fn, device=0):
    with open(fn+'/model.pt', 'rb') as f:
        model = torch.load(f, map_location=f'cuda:{device}')
    with open(fn+'/criterion.pt', 'rb') as f:
        criterion = torch.load(f, map_location=f'cuda:{device}')
    with open(fn+'/optimizer.pt', 'rb') as f:
        optimizer = torch.load(f, map_location=f'cuda:{device}')
    return model, criterion, optimizer

def corpus_load(corpus_path, test, use_unk=False):
    if test:
        fn = 'corpus.{}.data'.format(hashlib.md5((corpus_path.strip('/')+"-test").encode()).hexdigest())
    else:
        fn = 'corpus.{}.data'.format(hashlib.md5(corpus_path.strip('/').encode()).hexdigest())
    print (fn)
    if os.path.exists(fn):
        logging('Loading cached dataset from {}...'.format(corpus_path))
        corpus = torch.load(fn)
    else:
        logging('Producing dataset from {} ...'.format(corpus_path))
        corpus = data.Corpus(args.test_data, use_unk=use_unk)
        torch.save(corpus, fn)
    return corpus

def evaluate(model, criterion, data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    n = data_source.size(0)
    cache_N = args.cache_size
    output_dim = model.H.emsize
    V = len(model.dict.word2idx)
    batch_row_idx = torch.arange(batch_size).long()
    uniform_prob = 1/V
    uniform_dist = (torch.zeros((1,V)).cuda() + uniform_prob).log()
    unigram_counts = torch.zeros((batch_size,V)).cuda()
    cache = deque(maxlen=cache_N)

    hidden = model.init_hidden(batch_size)
    if not os.path.isfile(os.path.join(args.save, 'recover-state.pkl')):
        start_iter = 0
        total_loss = 0
    else:
        logging("Restoring from recover-state.pkl...")
        with open(os.path.join(args.save, 'recover-state.pkl'),'rb') as f:
            start_iter, total_loss = pickle.load(f)
        logging("Restoring from recover-cache-targets.pt...")
        if args.adapt_method in ["interpolate_neural", "interpolate_unigram"]:
            hidden = torch.load(os.path.join(args.save, 'recover-hidden.pt'))
            cache_targets = torch.load(os.path.join(args.save, 'recover-cache-targets.pt'))
            if args.adapt_method == "interpolate_neural":
                logging("Restoring from recover-cache.pt...")
                cache_vectors = torch.load(os.path.join(args.save, 'recover-cache.pt'))
            else:
                cache_vectors = torch.zeros((batch_size,2,cache_targets.size(0)))
            for ci in range(cache_targets.size(0)):
                cache.append((cache_targets[ci],cache_vectors[:,:,ci]))
                unigram_counts[batch_row_idx,cache_targets[ci]] += 1
        if len(cache) > 0:
            print("{} {}".format(cache[0][0].size(), cache[0][1].size()))
        logging("Restore complete.")

    if args.bptt > 1 and args.adapt_method in interpolate_methods:
        logging("Warning: cache will not work with bptt > 1")
    if isinstance(criterion, nn.CrossEntropyLoss):
        softmax = nn.LogSoftmax(dim=1)
        loss_function = nn.NLLLoss()

    if args.downweight_oov > 0.0 and args.downweight_oov < 1.0:
        dw_inds = model.get_new()
    uniform_dist_shaped = None
    for i in range(start_iter, n-1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, weight, bias, hidden = model(data, hidden, eval_mode=True)

        if isinstance(criterion, nn.CrossEntropyLoss):
            logits = torch.mm(output,weight.t())
            if args.downweight_oov > 0.0 and args.downweight_oov < 1.0:
                lexp = logits.exp()
                lexp[:,dw_inds] *= args.downweight_oov
                logits = lexp.log()
            logits += bias
            if args.adapt_method == "change_vocab":
                loss = criterion(logits, targets).data
            elif args.adapt_method in interpolate_methods:
                # first do uniform interpolation by adding a constant to logits
                model_dist = softmax(logits)
                if uniform_dist_shaped is None:
                    uniform_dist_shaped = uniform_dist.repeat(model_dist.size(0),1)
                model_dist = log_interpolate(model_dist,
                                             uniform_dist_shaped,
                                             weight=args.lamu)
                if args.adapt_method == "interpolate_unigram":
                    unigram_dist = softmax(unigram_counts)
                    model_dist = log_interpolate(model_dist, unigram_dist,
                                                 weight=args.lam)
                elif args.adapt_method == "interpolate_neural" and i != 0:
                    cache_inds = torch.cat([c[0].view(1,-1) for c in cache],dim=0)

                    # b * n * m (n = 1, m = dim)
                    batched_output = output.view(batch_size,1,-1)
                    # b * m * p (m = dim, p = number of elements in cache)
                    batched_cache_weight = torch.cat([c[1].view(batch_size,-1,1) for c in cache],
                                                     dim=2)
                    # b * n * p (n = 1, p = number of elements in cache. squeeze out n to get
                    # b distributions over the cache (batch_size * cache_size)
                    cache_scores = torch.bmm(batched_output,batched_cache_weight).view(batch_size,-1)
                    if args.global_norm:
                        cache_scores = (cache_scores * args.theta + args.alpha).exp()
                    else:
                        cache_scores = (cache_scores * args.theta).exp()
                    cache_vocab_scores = torch.zeros_like(logits)
                    cache_vocab_scores.scatter_add_(dim=1,
                                                    index=cache_inds.t(),
                                                    src=cache_scores)
                    if args.global_norm:
                        model_dist = softmax((logits.exp() + cache_vocab_scores).log())
                    else:
                        neural_cache_dist = softmax(cache_vocab_scores.log())
                        model_dist = log_interpolate(model_dist, neural_cache_dist,
                                                     weight=args.lam)

                loss = loss_function(model_dist, targets).data
            else:
                raise ValueError("Unknown adaptation method: {}".format(args.adapt_method))
            total_loss += len(data) * loss
        hidden = repackage_hidden(hidden)

        # update cache
        if len(cache) == cache.maxlen:
            old_targets, _ = cache.popleft()
            unigram_counts[batch_row_idx,old_targets] -= 1
        if args.adapt_method == "interpolate_unigram":
            cache.append((targets,None))
        elif args.adapt_method == "interpolate_neural":
            cache.append((targets,output))
        unigram_counts[batch_row_idx,targets] += 1

        print("\r{}/{} - ppl: {:3.2f}".format(i,n, math.exp(total_loss/n) ))

        if i % 100 == 0 and i > 0:
            logging("{}/{} - ppl: {:3.2f}".format(i,n, math.exp(total_loss/n)))
        if i % 5000 == 0 and i > 0:
            logging("Saving to recover-state.pkl...")
            with open(os.path.join(args.save, 'recover-state.pkl'),'wb') as f:
                pickle.dump((i, total_loss),f)
            if args.adapt_method in ["interpolate_neural", "interpolate_unigram"]:
                torch.save(hidden, os.path.join(args.save,'recover-hidden.pt'))
                if args.adapt_method == "interpolate_neural":
                    logging("Saving to recover-cache.pt... ({})".format(batched_cache_weight.size()))
                    torch.save(batched_cache_weight, os.path.join(args.save,'recover-cache.pt'))
                else:
                    cache_inds = torch.cat([c[0].view(1,-1) for c in cache],dim=0)
                logging("Saving to recover-cache-targets.pt... ({})".format(cache_inds.size()))
                torch.save(cache_inds,os.path.join(args.save, 'recover-cache-targets.pt'))
            logging("Save complete.")

    return total_loss.item() / n

# Log command
logging("Command: python " + " ".join(sys.argv))

# Load the best saved model.
model, criterion, optimizer = model_load(args.save, device=args.cuda_device)

if args.adapt_method in interpolate_methods:
    """
    use the original vocab for indexing, no change to model parameters
    K = size of new words (OOVs in the original vocab)
    N = size of train vocab plus new words
    uniform interpolation:
    - likelihood = (lambda * likelihood) + ((1-lambda) * 1/N)
    unigram interpolation:
    - likelihood = (lambda * likelihood) +
                   ((1-lambda) * (p_unigram(target)/sum(p_unigram(w) for w in observed_vocab)))
    neural cache interpolation:
    - likelihood = (lambda * likelihood) +
                   ((1-lambda) * (p_cache(target)/sum(p_cache(w) for w in cache_vocab)))
    """
    test_corpus = corpus_load(args.test_data, test=True)
    model.H.ntoken = len(test_corpus.dictionary.idx2word)
    char_arr, rel_arr, def_arr = get_external_knowledge(model.H, test_corpus)
    model.change_embedding_vocab(char_arr, rel_arr, def_arr,
                                 test_corpus.dictionary, set_zero=True)
    logging("Vocab size pre-change: {}".format(len(model.old_dict.word2idx)))
    logging("Vocab size post-change: {}".format(len(model.dict.word2idx)))
else:
    raise AssertionError("new vocabulary provided but model vocab not changed or interpolated")

test_data = batchify(test_corpus.test, args.test_batch_size, args)

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
else:
    model = model.cpu()
    criterion = criterion.cpu()

# Run on test data.
logging("Evaluating...")
with torch.no_grad():
    if args.hyp_search is not None:
        best_score = (np.inf,0.0,0.0)
        scores = np.zeros((5,6))
        import pickle
        # grid search is ok here bc for few hyperparams and small k,
        # it helps minimize gaps. also, based on Grave et al. (2016)
        # we expect lam and theta are ~equally important/sensitive here
        for i,lam in enumerate([0.833, 0.866, 0.9, 0.933, 0.966]):
            for j,theta in enumerate([0, 0.1, 0.3, 0.5, 0.7, 0.9]):
                args.lam = lam
                args.theta = theta
                test_loss = evaluate(model, criterion, test_data, args.test_batch_size)
                scores[i,j] = math.exp(test_loss)
                if math.exp(test_loss) < best_score[0]:
                    best_score = (math.exp(test_loss), lam, theta)
                print("    ({},{})".format(i,j),flush=True)
        with open('hyperparam_search_{}.pkl'.format(args.hyp_search), 'wb') as f:
            pickle.dump(scores,f)
        ppl, lam, theta = best_score
        logging(f"Best ppl {ppl} with lambda {lam} and theta {theta}")
    else:
        test_loss = evaluate(model, criterion, test_data, args.test_batch_size)
        print("")
        logging('=' * 89)
        logging('| End of evaluation | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(test_loss, math.exp(test_loss), test_loss / math.log(2)))
        logging
        logging('=' * 89)
