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
import sys
import hashlib
import time
import math
import numpy as np
import torch
import torch.nn as nn
import data
import model
from utils import *

import IPython as ipy

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--rnn_type', type=str, default='LSTM',
                    help='type of recurrent net (LSTM)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-03,
                    help='initial learning rate')
parser.add_argument('--beta0', type=float, default=0.9,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--joint_emb', type=int, default=None,
                    help='whether to use joint embedding or not')
parser.add_argument('--fullsoftmax', action='store_true',
                    help='whether to use full softmax or not')
parser.add_argument('--joint_emb_depth', type=int, default=1,
                    help='depth of the joint embedding')
parser.add_argument('--joint_dropout', type=float, default=0.2,
                    help='dropout for joint embedding layers')
parser.add_argument('--joint_emb_dense', action='store_true', default=False,
                    help='add residuals to all previous joint embedding projections')
parser.add_argument('--joint_emb_activation', type=str, default='Sigmoid',
                    help='activation function for the joint embedding layers')
parser.add_argument('--joint_emb_dual', action='store_true', default=False,
                    help='whether to use projection on both input and output side or not')
parser.add_argument('--joint_locked_dropout', action='store_true', default=False,
                    help='whether to use locked dropout or not for the joint space')
parser.add_argument('--joint_residual_prev', action='store_true', default=False,
                    help='whether to use residual connection to previous layer')
parser.add_argument('--joint_noresid', action='store_true', default=False,
                    help='disable residual connections')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action="store_true", default=False,
                    help='use CUDA device')
parser.add_argument('--cuda_device', type=int, default=-1,
                    help='set CUDA device')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--init', type=float, default=0.1,
                    help='range for initialization')
parser.add_argument('--resume', action="store_true", default=False,
                    help='resume or not')
parser.add_argument('--char_emb', action="store_true", default=False,
                    help='use character embedding or not')
parser.add_argument('--rel_emb', action="store_true", default=False,
                    help='use relation embedding or not')
parser.add_argument('--def_emb', action="store_true", default=False,
                    help='use definition embedding or not')
parser.add_argument('--combine', type=str, default="add",
                    help='how to combine different forms (add | concat | multiply)')
parser.add_argument('--predict_bias', action="store_true", default=False,
                    help='use the bias estimator or not')
parser.add_argument('--bias_drop', type=float, default=0.2,
                    help='dropout for bias estimator (0 = no dropout)')
parser.add_argument('--bias_out', type=int, default=1,
                    help='output dim for bias estimator')
parser.add_argument('--bias_activation', type=str, default="tanh",
                    help='activation for bias estimator')
parser.add_argument('--div', type=int, default=500,
                    help='number to divide the vocab size for batch computation')
parser.add_argument('--offset', type=int, default=None,
                    help='offset for reducing compute for the char emb')
parser.add_argument('--hdepth', type=int, default=1,
                    help='character emb highway depth')
parser.add_argument('--max_deflen', type=int, default=10,
                    help='maximum definition length')
parser.add_argument('--max_rellen', type=int, default=3,
                    help='maximum relation length')
parser.add_argument('--max_charlen', type=int, default=20,
                    help='maximum char length')
parser.add_argument('--defenc', type=str,  default=None,
                    help='encoder for description and relations (lstm | highway)')
parser.add_argument('--char_emsize', type=int, default=16,
                    help='size of character embeddings')
parser.add_argument('--char_activation', type=str, default="selu",
                    help='character embedding activation (applies to convolutions and highway network)')
parser.add_argument('--optimizer', type=str,  default='adam',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--cnnsoftmax', action="store_true",
                    default=False, help='whether to use cnn softmax or not')
parser.add_argument('--cnncorr', type=int, default=None,
                    help='whether to use cnn correction or not and its dimensionality')
parser.add_argument('--bilinear', action="store_true",
                    default=False, help='whether to use bilinear output embedding not')
parser.add_argument('--adaptiveoutputs', action="store_true",
                    default=False, help='whether to use adaptive output embedding or not')
parser.add_argument('--adaptiveoutputs_tied', action="store_true",
                    default=False, help='whether to use tied adaptive output embedding or not')
parser.add_argument('--adaptivecutoffs', type=str,
                    default="[5000]", help='cutoffs for adaptive outputs')
parser.add_argument('--output_dropout', type=float,
                    default=0.0, help='locked dropout for outputs')
parser.add_argument('--char_update_ratio', type=float,
                    default=1.0, help='portion of regular updates versus detached ones for the grounded embedding. ')
parser.add_argument('--char_nohighways', action="store_true",
                    help='do not use highways after the convolutional network. ')
parser.add_argument('--coverage', type=float, default=1.0,
                    help='control vocabulary coverage from the external knowledge base. ')
parser.add_argument('--finetune', action="store_true",
                    help="fine-tune an existing model on new data (with a new vocab)")
parser.add_argument('--finetune_save', default=None,
                    help="save fine-tuned model and logs to this directory (not the original")


args = parser.parse_args()
args.tied = True

if args.joint_emb is not None:
    args.tied = False
if args.fullsoftmax:
    args.tied = False
if args.adaptiveoutputs:
    args.tied = False

if args.finetune:
    args.resume = True
    save_dir = args.finetune_save
    load_dir = args.save
else:
    save_dir = args.save
    load_dir = args.save

if args.finetune or not args.resume:
    try:
        create_exp_dir(save_dir, scripts_to_save=['main.py', 'model.py'])
    except:
        res = input("Directory exists! Try again with --resume. ")
        exit(0)

with open(os.path.join(save_dir, 'replicate.sh'), 'w') as f:
    f.write('python ' + ' '.join(sys.argv))

def logging(s, print_=True, log_=True):
    """
       Function to print logs to be used by different files.
    """
    print(s)
    if log_:
        with open(os.path.join(save_dir, 'log.txt'), 'a+') as f_log:
            f_log.write(str(s) + '\n')

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    if not args.cuda:
        logging("WARNING: You have a CUDA device, so you should probably run with --cuda and --cuda_device [device_id]")
    else:
        torch.cuda.set_device(int(args.cuda_device))
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(model, criterion, optimizer, save):
    save_checkpoint(model, criterion, optimizer, save)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn+'/model.pt', 'rb') as f:
        model = torch.load(f, map_location="cuda:%d" % args.cuda_device)
        model.H.coverage = args.coverage
        model.H.char_update_ratio = args.char_update_ratio
        if hasattr(model, "rel_arr") and model.rel_arr is not None and len(model.rel_arr) > 0:
            model.rel_arr = model.coverage_filter(model.rel_arr.to(args.cuda_device))
        if hasattr(model, "def_arr") and model.def_arr is not None and len(model.def_arr) > 0:
            model.def_arr = model.coverage_filter(model.def_arr.to(args.cuda_device))
    with open(fn+'/criterion.pt', 'rb') as f:
        criterion = torch.load(f)
    with open(fn+'/optimizer.pt', 'rb') as f:
        optimizer = torch.load(f)
    return model, criterion, optimizer

def corpus_load(corpus_path, use_unk=False):
    fn = 'corpus.{}.data'.format(hashlib.md5(corpus_path.strip('/').encode()).hexdigest())
    print (fn)
    if os.path.exists(fn):
        logging('Loading cached dataset from {}...'.format(corpus_path))
        corpus = torch.load(fn)
    else:
        logging('Producing dataset from {} ...'.format(corpus_path))
        corpus = data.Corpus(args.data, use_unk=use_unk)
        torch.save(corpus, fn)
    return corpus


corpus = corpus_load(args.data)

eval_batch_size = 30
test_batch_size = 2
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)
ntokens = len(corpus.dictionary)
args.ntoken = ntokens

if not args.finetune:
    char_arr, rel_arr, def_arr = get_external_knowledge(args, corpus)

###############################################################################
# Build the model
###############################################################################

criterion = None
if args.resume:
    logging('Resuming model ...')
    model, criterion, optimizer = model_load(load_dir)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
else:
    model = model.RNNModel( args, char_arr=char_arr, rel_arr=rel_arr, def_arr=def_arr, dict=corpus.dictionary)
    criterion = load_criterion(args, ntokens, logging)


if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size() and x.requires_grad)
logging('Args: %s' % args)
logging('Model total parameters: %d ' % total_params)
logging('Model vocab size: %d ' % len(model.dict.word2idx))
logging(model)

if args.finetune:
    char_arr, rel_arr, def_arr = get_external_knowledge(model.H, corpus)
    model.change_embedding_vocab(char_arr, rel_arr, def_arr, corpus.dictionary)
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size() and x.requires_grad)
    logging('Model total parameters post-vocab-change: %d ' % total_params)
    logging('Model vocab size post-vocab-change: %d ' % len(model.dict.word2idx))
    logging(model)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = torch.Tensor([0])
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, weight, bias, hidden = model(data, hidden)
        logits = torch.mm(output,weight.t()) + bias
        total_loss += len(data) * criterion(logits, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, weight, bias, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)


        logits = torch.mm(output,weight.t()) + bias
        raw_loss = criterion(logits, targets)

        loss = raw_loss
        # Activation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5e} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
steps_wo_inc = 0
total_steps_wo_inc = 0

# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(args.beta0, 0.999), eps=1e-09, weight_decay=args.wdecay)
    if args.optimizer == 'asgd':
        optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for j, prm in enumerate(model.parameters()):
                tmp[prm] = prm.data.clone()
                curstate = optimizer.state[prm]
                prm.data = curstate['ax'].clone()
            val_loss2 = evaluate(val_data, eval_batch_size)
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            logging('-' * 89)

            if val_loss2 < stored_loss:
                model_save(model, criterion, optimizer, save_dir)
                logging('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()
        else:
            val_loss = evaluate(val_data, eval_batch_size)
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            logging('-' * 89)
            if np.isnan(val_loss):
                logging('Exiting...(loss has nan value)')
                exit(-1)

            if val_loss < stored_loss:
                model_save(model, criterion, optimizer, save_dir)
                logging('Saving model (new best validation)')
                stored_loss = val_loss
                steps_wo_inc = 0
                total_steps_wo_inc = 0

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                logging('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if val_loss > stored_loss:
                steps_wo_inc += 1
                if steps_wo_inc == 4:
                    logging('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.
                    steps_wo_inc = 0
                total_steps_wo_inc += 1
                if  total_steps_wo_inc == 8:
                    logging('Exiting...')
                    break

            if epoch in args.when:
                logging('Saving model before learning rate decreased')
                model_save(model, criterion, optimizer, '{}.e{}'.format(save_dir, epoch))
                logging('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
# even if we resumed, we want to load the model saved on _this_ run, so use save_dir
model, criterion, optimizer = model_load(save_dir)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
logging('=' * 89)
