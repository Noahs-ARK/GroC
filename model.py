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

import torch
import torch.nn as nn
from locked_dropout import LockedDropout
import torch.nn.functional as F
import numpy as np
from allennlp.modules.highway import Highway
from adaptive_io import AdaptiveEmbedding
from utils import *

import IPython as ipy

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, H, char_arr=None, rel_arr=None, def_arr=None, dict=None):
        super(RNNModel, self).__init__()
        self.H  = H
        self.dict = dict
        self.use_dropout = True
        self._lockdrop = LockedDropout()
        self.define_embedding(H, char_arr, rel_arr, def_arr)
        self.define_rnn(H)
        self.define_joint(H)
        self.define_bias(H)
        self.init_weights()

    def define_joint(self,H):
        """
            Define the joint embedding for the deep residual method.
        """
        if H.joint_emb is not None:
            self._output_network = nn.ModuleList()
            for i in range(H.joint_emb_depth):
                self._output_network.append(nn.Linear(H.joint_emb, H.joint_emb, bias=True))
            self._output_act = eval("torch.nn.functional.%s" % H.joint_emb_activation.lower())

    def define_embedding(self, H, char_arr, rel_arr, def_arr):
        """
            Define the embedding for different methods.
        """
        if H.joint_emb is not None:
            self._jdrop = nn.Dropout(H.joint_dropout if self.use_dropout else 0)

        if H.char_emb or H.cnnsoftmax:
            self.char_arr =  torch.LongTensor(char_arr).cuda()
            self.rel_arr, self.def_arr = None, None
            self._char_emb = nn.Embedding(262, H.char_emsize).cuda()
            self._char_network = nn.ModuleList()
            self._char_network.append(nn.Conv1d(H.char_emsize, 32, 1, stride=(1,)).cuda())
            self._char_network.append(nn.Conv1d(H.char_emsize, 32, 2, stride=(1,)).cuda())
            self._char_network.append(nn.Conv1d(H.char_emsize, 64, 3, stride=(2,)).cuda())
            self._char_network.append(nn.Conv1d(H.char_emsize, 128, 4, stride=(3,)).cuda())
            self._char_network.append(nn.Conv1d(H.char_emsize, 256, 5, stride=(4,)).cuda())
            self._char_network.append(nn.Conv1d(H.char_emsize, 512, 6, stride=(5,)).cuda())
            self._conv_activation = eval("torch.nn.functional.%s" % H.char_activation)
            if not H.char_nohighways:
                self._char_highways = Highway(1024, H.hdepth, activation=self._conv_activation)
            self._char_linear =  nn.Linear(1024, H.emsize, bias=False)
            nforms = 1

            if rel_arr:
                self.rel_arr = self.coverage_filter(torch.LongTensor(rel_arr).cuda())
                nforms += 1
            if def_arr:
                self.def_arr = self.coverage_filter(torch.LongTensor(def_arr).cuda())
                nforms += 1

            self.rel_exist = self.rel_arr is not None
            self.def_exist = self.def_arr is not None
            self.nforms = H.nforms = nforms
            if H.defenc == "lstm":
                if def_arr:
                    defsize = self.def_arr.shape[1]
                    def_h = torch.zeros(H.hdepth,defsize,H.emsize).cuda()
                    self.def_hid = (def_h,def_h)
                if rel_arr:
                    relsize = self.rel_arr.shape[1]
                    rel_h = torch.zeros(H.hdepth,relsize,H.emsize).cuda()
                    self.rel_hid = (rel_h, rel_h)
                self._definition_network = torch.nn.LSTM(H.emsize, H.emsize, num_layers=H.hdepth)
            elif H.defenc == "highway":
                self._definition_network = Highway(H.emsize, H.hdepth, activation=self._conv_activation)

            if H.combine == "concat":
                self._comb_lin = nn.Linear(H.emsize*H.nforms, H.emsize, bias=True)

            if H.cnnsoftmax or H.char_emb:
                if H.cnnsoftmax:
                    self._lookup = nn.Embedding(H.ntoken, H.emsize)
                if H.cnncorr:
                    self._cnnsoftmax_correction = nn.Linear(H.cnncorr, H.ntoken, bias=False)
                    self._cnnsoftmax_M = nn.Linear(H.cnncorr, H.emsize, bias=False)
        else:
            self._lookup = nn.Embedding(H.ntoken, H.emsize)


    def coverage_filter(self, arr):
        """
            Function to control the vocabulary coverage of the external knowledge
            base for relational and definitional forms.
        """
        if self.H.coverage < 1:
            ex_ids = (arr - self.H.ntoken).sum(dim=1).nonzero().squeeze()
            mask = (torch.rand(len(ex_ids)) >= self.H.coverage).cuda().long()
            ids = (mask*torch.arange(len(ex_ids)).cuda()).nonzero()
            arr[ex_ids[ids]] = self.H.ntoken
        return arr


    def change_embedding_vocab(self, char_arr, rel_arr, def_arr, new_dict,
                               set_zero=False):
        """
        if set_zero=True, the new embedding params should be all 0s (so the only
        probability new words get is from uniform interpolation). otherwise, they
        should be randomly initialized for fine-tuning.
        """
        self.H.ntoken = len(new_dict.word2idx)
        self.old_dict = self.dict
        self.dict = new_dict

        new_words = []
        for w in self.dict.word2idx:
            if w not in self.old_dict.word2idx:
                new_words.append(self.dict.word2idx[w])
        self.new_words = torch.LongTensor(new_words)

        if hasattr(self, "_lookup"):
            new_lookup = nn.Embedding(self.H.ntoken, self.H.emsize)
            if set_zero:
                torch.nn.init.zeros_(new_lookup.weight)
            else:
                init = self.H.init
                torch.nn.init.uniform_(new_lookup.weight,-init,init)
            # copy old embeddings
            with torch.no_grad():
                for word in self.old_dict.word2idx:
                    old_idx = self.old_dict.word2idx[word]
                    if word in self.dict.word2idx:
                        new_idx = self.dict.word2idx[word]
                        new_lookup.weight[new_idx] = self._lookup.weight[old_idx]

        if (hasattr(self.H, "char_emb") and self.H.char_emb) or \
           (hasattr(self.H, "cnnsoftmax") and self.H.cnnsoftmax):
            self.char_arr =  torch.LongTensor(char_arr).cuda()
            if self.rel_exist:
                self.rel_arr = torch.LongTensor(rel_arr).cuda()
            if self.def_exist:
                self.def_arr = torch.LongTensor(def_arr).cuda()
            if self.H.cnnsoftmax:
                self._lookup = new_lookup
        else:
            self._lookup = new_lookup

        if not self.H.char_emb:
            if self.H.tied:
                with torch.no_grad():
                    new_decoder = nn.Linear(self.H.emsize, self.H.ntoken)
                    new_decoder.weight = torch.nn.Parameter(self._lookup.weight.data)
                    if set_zero:
                        torch.nn.init.zeros_(new_decoder.bias)
                        new_decoder.bias -= np.inf # give OOVs 0 probability
                    for word in self.old_dict.word2idx:
                        old_idx = self.old_dict.word2idx[word]
                        if word in self.dict.word2idx:
                            new_idx = self.dict.word2idx[word]
                            new_decoder.bias[new_idx] = self._decoder.bias[old_idx]
                self._decoder = new_decoder
            else:
                new_decoder = nn.Linear(self.H.emsize, self.H.ntoken)
                if set_zero:
                    torch.nn.init.zeros_(new_decoder.weight)
                    torch.nn.init.zeros_(new_decoder.bias)
                    with torch.no_grad():
                        new_decoder.bias -= np.inf # give OOVs 0 probability
                else:
                    init = self.H.init
                    torch.nn.init.uniform_(new_decoder.weight,-init,init)
                    torch.nn.init.uniform_(new_decoder.bias,-init,init)
                # copy old embeddings
                with torch.no_grad():
                    for word in self.old_dict.word2idx:
                        old_idx = self.old_dict.word2idx[word]
                        if word in self.dict.word2idx:
                            new_idx = self.dict.word2idx[word]
                            new_decoder.weight[new_idx] = self._decoder.weight[old_idx]
                            new_decoder.bias[new_idx] = self._decoder.bias[old_idx]
                self._decoder = new_decoder

        if hasattr(self, "_bias") and not self.H.predict_bias:
            self._bias = torch.nn.Linear(self.H.ntoken, 1, bias=False)
            self._bias.weight.data.fill_(0)

        if hasattr(self, "_lookup"):
            self._lookup.cuda()

        if hasattr(self, "_decoder"):
            self._decoder.cuda()

    def get_new(self):
        """
        Return indices for words which were not in the model's vocabulary prior
        to the most recent call to change_embedding_vocab
        """
        return self.new_words

    def get_uncovered(self):
        """
        Return indices for words which do not have rel/def coverage
        """
        if self.H.char_emb:
            return (self.rel_arr - self.H.ntoken).sum(dim=1).nonzero().view(-1)
        else:
            return torch.LongTensor([])

    def get_new_uncovered(self):
        uncovered = self.get_uncovered()
        new_uncovered = [w for w in new_words if w in uncovered]
        return torch.LongTensor(new_uncovered)

    def define_rnn(self, H):
        """
            Define the prefix encoder rnn.
        """
        assert H.rnn_type in ['LSTM'], 'RNN type is not supported'
        if H.rnn_type == 'LSTM':
            self._prefix_network = [torch.nn.LSTM(H.emsize if l == 0 else H.nhid, H.nhid if l != H.nlayers - 1 else (H.emsize), 1, dropout=0) for l in range(H.nlayers)]

        self._prefix_network = torch.nn.ModuleList(self._prefix_network)


    def define_bias(self, H):
        """
            Define bias for different methods.
        """
        if H.predict_bias:
            self._bias = torch.nn.Linear(H.emsize, H.bias_out, bias=True)
            self._bias_drop = nn.Dropout(H.bias_drop if self.use_dropout else 0)
            self._bias_activation = eval("torch.nn.functional.%s" % H.bias_activation)
        else:
            if H.adaptiveoutputs or H.adaptiveoutputs_tied:
                self._decoder = AdaptiveEmbedding(H.ntoken, H.emsize, H.emsize, cutoffs=eval(H.adaptivecutoffs))
                if H.adaptiveoutputs_tied:
                    self._lookup = self._decoder
                self._bias = torch.nn.Linear(H.ntoken, 1, bias=False)
            elif H.char_emb or H.joint_emb is not None or H.cnnsoftmax:
                self._bias = torch.nn.Linear(H.ntoken, 1, bias=False)
            else:
                self._decoder = nn.Linear(H.emsize, H.ntoken)
                if H.tied:
                    self._decoder.weight = self._lookup.weight

    def change_vocab(self, newdict):
        """
            Create new embeddings or use existing ones for the words
            in the new vocabulary.
        """
        H = self.H
        init = self.H.init
        new_ntoken = len(newdict.idx2word)
        print("Changing the vocab...")
        if H.tied:
            new_encoder = nn.Embedding(new_ntoken, H.emsize).cuda()
            new_decoder = nn.Linear(H.emsize, new_ntoken).cuda()
            new_encoder.weight.data.uniform_(-init, init)
            new_decoder.weight = new_encoder.weight
            self._lookup = new_encoder
            self._decoder = new_decoder
        self.dict = dict

    def init_weights(self):
        """
            Initialize weights by randomly from the same range.
        """
        init = self.H.init
        if not self.H.char_emb and not self.H.adaptiveoutputs and not self.H.adaptiveoutputs_tied:
            self._lookup.weight.data.uniform_(-init, init)

        if self.H.char_emb:
            for i, conv in enumerate(self._char_network):
                self._char_network[i].weight.data.uniform_(-init, init)
            if hasattr(self, "_char_highways"):
                for j, layer in enumerate(self._char_highways._layers):
                    self._char_highways._layers[j].weight.data.uniform_(-init, init)
            self._char_linear.weight.data.uniform_(-init, init)

        if self.H.defenc == "highway":
            for j, layer in enumerate(self._definition_network._layers):
                self._definition_network._layers[j].weight.data.uniform_(-init, init)

        if hasattr(self, 'decoder'):
            if hasattr(self._decoder, 'bias') and self._decoder.bias is not None:
                self._decoder.bias.data.fill_(0)
            if self.H.joint_emb:
                for i in range(self.H.joint_emb_depth):
                    self._output_network[i][0].weight.data.uniform_(-init, init)
        if hasattr(self, 'bias'):
            if not self.H.predict_bias:
                if hasattr(self, 'bias'):
                    self._bias.weight.data.fill_(0)
            else:
                self._bias.weight.data.uniform_(-init, init)

    def init_hidden(self, bsz):
        """
            Initializes the hidden state and cell state of the prefix network.
        """
        weight = next(self.parameters()).data
        if self.H.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.H.nhid if l != self.H.nlayers - 1 else (self.H.emsize)).zero_(),
                    weight.new(1, bsz, self.H.nhid if l != self.H.nlayers - 1 else (self.H.emsize)).zero_())
                    for l in range(self.H.nlayers)]

    def estimate_bias(self, weight):
       """
           Estimates the vocabulary bias based on the current weight.
       """
       weight = self._bias_drop(weight)
       return self._bias_activation((self._bias(weight))).min(dim=1)[0]

    def char_enc(self, char_arr, cache=None):
        """
            Character-level encoder that extracts surface features for each word represented by its characters.
        """
        char_emb = self._char_emb(char_arr)
        sh = char_emb.shape
        char_emb = char_emb.view(sh[1], sh[3], sh[2])
        token_embedding = cache
        for conv in self._char_network:
            out = conv(char_emb).max(dim=-1)
            convolved = self._conv_activation(out[0])
            if token_embedding is not None:
                token_embedding = torch.cat([token_embedding, convolved], dim=-1)
            else:
                token_embedding = convolved
            del(convolved)
            del(out)
            torch.cuda.empty_cache()
        if hasattr(self, "_char_highways"):
            token_embedding = self._char_highways(token_embedding)
        del(char_emb)
        torch.cuda.empty_cache()
        return self._char_linear(token_embedding)

    def get_weight(self, char_arr):
        """
            Function which encodes and returns the given vocabulary items in a memory efficient way.
        """
        div = self.H.div
        maxbs = round(char_arr.shape[1] / div) + 1
        char_emb, result = None, None
        for i in range(maxbs):
            cur_arr = char_arr[:,(i*div):(i+1)*div,:]
            if cur_arr.shape[1] == 0:
                break
            cur_emb = self.char_enc(cur_arr, cache=None)
            diff = cur_emb.shape[0]
            if result is not None:
                result = torch.cat([result, cur_emb])
            else:
                result = cur_emb
            del (cur_emb)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return result

    def batch_combined_enc(self, l=None, r=None, new=None):
        """
            Compute the compositional representations for the selected vocabulary items in a batch-like mode. For efficiency the updates are made in a sparse way with probability p that is controlled by
            the char_update_ratio argument.
        """
        # Get the indexes correspond to the left and right offsets
        num = r - l
        full = r == self.H.ntoken and l == 0
        if full:
            num += 1 # add empty token

        vocab_idxs = torch.tensor(np.arange(num)) + l
        # Get the combined representations for those indexes
        fixed = np.random.rand() > self.H.char_update_ratio and not new

        if not fixed or not hasattr(self, "combined_cached"):
            self.combined_cached = self.combined_enc(vocab_idxs, full=full, fixed=fixed, l=l, r=r)
        char_emb = self.combined_cached

        if fixed:
            char_emb = char_emb.detach()
        return char_emb

    def combined_enc(self, input, full=False, fixed=None, l=0, r=0):
        """
            Get the combined representation for a given input. The input here
            can be the training batch tensor or a 1d array with indexes pointing
            to the vocabulary elements.
        """
        batch_input = len(input.shape) > 1
        if batch_input:
            # Encode the training batch tensor.
            char_emb = self.char_enc(self.char_arr[:,input.view(-1)])
        else:
            # Encode the elements in the vocabulary.
            char_emb = self.get_weight(self.char_arr[:,input.view(-1)])
            char_emb[-1] = 0 #empty pad
        result = self.batch_combined_forms(char_emb, input, batch_input=batch_input, full=full, fixed=fixed, r=r)
        if full and not batch_input:
            return result[:-1]
        else:
            return result

    def batch_combined_forms(self, char_emb, input, batch_input=None, full=None, fixed=None, r=None):
        """
            Compute the compositional input embedding of words by taking
            into account surface, relational, and definitional features.
        """
        if not self.rel_exist and not self.def_exist:
            result = char_emb
        else:
            div = self.H.div
            maxbs = round(char_emb.shape[0] / div) + 1
            result = None
            for i in range(maxbs):
                if batch_input:
                    cur_emb = char_emb
                else:
                    cur_emb = char_emb[i*div:(i+1)*div,:]
                rel_emb, def_emb = None, None
                cur_rel_emb, cur_def_emb = None, None
                if cur_emb.shape[0] == 0:
                    break
                # Compute relational embedding
                if self.rel_exist:
                    if batch_input:
                        word_ids = self.rel_arr[input.view(-1)]
                    else:
                        word_ids = self.rel_arr[input[i*div:(i+1)*div]]
                    rel_emb = self.embed_features(word_ids, char_emb, full=full, fixed=fixed, r=r)
                    cur_rel_emb = self.fpass(rel_emb, "rel")
                    if full and not batch_input:
                        cur_rel_emb[-1] = 0

                # Compute definitional embedding
                if self.def_exist:
                    if batch_input:
                        word_ids = self.def_arr[input.view(-1)]
                    else:
                        word_ids = self.def_arr[input[i*div:(i+1)*div]]
                    def_emb = self.embed_features(word_ids, char_emb, full=full, fixed=fixed, r=r)
                    cur_def_emb = self.fpass(def_emb, "def")
                    if full and not batch_input:
                        cur_def_emb[-1] = 0

                # Compute combined embedding
                cur_res = self.combine(cur_emb, rel_emb=cur_rel_emb, def_emb=cur_def_emb)

                if result is not None:
                    result = torch.cat([result, cur_res])
                else:
                    result = cur_res

                del (word_ids, cur_res, cur_emb, def_emb)
                del (rel_emb, cur_def_emb, cur_rel_emb)
                torch.cuda.empty_cache()
                if batch_input:
                    break
            torch.cuda.empty_cache()
        return result

    def embed_features(self, word_ids, char_emb, full=False, fixed=False, r=0):
        """
            Get an embedding for relational or definitional features.
        """
        if full:
            # Use the recently encoded vocabulary items to represent relations.
            emb = char_emb[word_ids]
        else:
            # Encode the vocabulary items needed for the relational features.
            sh = word_ids.shape
            emb = self.char_enc(self.char_arr[:,word_ids.view(-1)])
            emb = emb.view(sh[0], sh[1], emb.shape[-1])
        return emb


    def combine(self, cur_emb, rel_emb=None, def_emb=None):
        """
            Function to combine surface, relational and definitional features.
        """

        if self.rel_exist and self.def_exist:
            if self.H.combine == "concat":
                combined = torch.cat((cur_emb, rel_emb, def_emb), 1)
                result = self._comb_lin(combined.view(-1, self.H.emsize*self.nforms))
            elif self.H.combine == "add":
                result = cur_emb + rel_emb + def_emb
            elif self.H.combine == "multiply":
                result = cur_emb * rel_emb * def_emb
        elif self.def_exist:
            if self.H.combine == "concat":
                combined = torch.cat((cur_emb, def_emb), 1)
                result =  self._comb_lin(combined.view(-1, self.H.emsize*self.nforms))
            elif self.H.combine == "add":
                result = cur_emb + def_emb
            elif self.H.combine == "multiply":
                result = cur_emb * def_emb
        elif self.rel_exist:
            if self.H.combine == "concat":
                combined = torch.cat((cur_emb, rel_emb), 1)
                result =  self._comb_lin(combined.view(-1, self.H.emsize*self.nforms))
            elif self.H.combine == "add":
                result = cur_emb + rel_emb
            elif self.H.combine == "multiply":
                result = cur_emb * rel_emb
        else:
            result = cur_emb

        return result

    def fpass(self, y, type):
        """
            Encoding function for the relational and definitional features.
            Currently implemented with a highway net.
        """
        if self.H.defenc == "lstm":
            h = eval("self.%s_hid" % type)
            y, h  = self._definition_network(y, h)
        elif self.H.defenc == "highway":
            y = self._definition_network(y)
        return y.mean(dim=1)

    def batch_apply_output_network(self, weight, l=None, r=None):
        """
            Memory efficient way to use apply_output_network() function onss large vocabularies.
        """
        div = self.H.div
        maxbs = round(weight.shape[0] / div) + 1
        result = None
        for i in range(maxbs):
            cur_res = self.apply_output_network(weight[i*div:(i+1)*div,:])
            if cur_res.shape[0] == 0:
                break
            if result is not None:
               result = torch.cat([result, cur_res])
            else:
               result = cur_res
            del(cur_res)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return result

    def apply_output_network(self, weight):
        """
            Make a forward pass of the given input embeddings stored in <weight> through a deep residual network to get the output embedding.
        """
        sh = weight.shape
        prev_encoder_out = weight
        for i in range(self.H.joint_emb_depth):
            if self.H.joint_locked_dropout:
                cur_weight = self._lockdrop(prev_encoder_out.view(sh[0], 1, sh[1]), self.H.joint_dropout if self.use_dropout else 0).view(sh[0], sh[1])
            else:
                cur_weight = self._jdrop(prev_encoder_out) if self.use_dropout else prev_encoder_out
            if self.H.bilinear:
                cur_weight_proj = self._output_network[i](cur_weight)
            else:
                cur_weight_proj = self._output_act(self._output_network[i](cur_weight))
                cur_weight_proj = cur_weight_proj + weight
            prev_encoder_out = cur_weight_proj
            del (cur_weight_proj)
            del (cur_weight)
            torch.cuda.empty_cache()
        return  prev_encoder_out

    def rnn_pass(self, raw_output, hidden):
        """
            Make a forward pass of the given input stored in <emb> through
            the recurrent network. <hidden> is the state of the reccurrent
            network from the previous timestep.
        """
        new_hidden, raw_outputs, outputs = [], [], []
        for l, rnn in enumerate(self._prefix_network):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.H.nlayers - 1:
                raw_output = self._lockdrop(raw_output, self.H.dropouth if self.use_dropout else 0)
                outputs.append(raw_output)
        return raw_output, raw_outputs, outputs, new_hidden

    def output_dropout(self, weight):
        if self.H.output_dropout > 0 and weight is not None:
            sh = weight.shape
            weight = self._lockdrop(weight.view(sh[0], 1, sh[1]), self.H.output_dropout if self.use_dropout else 0).view(sh[0], sh[1])
        return weight

    def output_embedding(self, l_idx, h_idx, weight=None, adapt_call=False, new=None):
        """
            Return the output embedding for the full vocabulary or for
            a subset of it when there is a call from adaptive softmax.
        """
        if adapt_call:
            if self.H.char_emb or self.H.cnnsoftmax:
                weight = self.batch_combined_enc(l=l_idx, r=h_idx, new=new)
                if self.H.cnncorr:
                    corr = self._cnnsoftmax_M(self._cnnsoftmax_correction.weight)
                    weight = weight + corr[l_idx:h_idx]
            else:
                weight = weight[l_idx: h_idx]
            weight = self.output_dropout(weight)
            if self.H.joint_emb:
                weight = self.batch_apply_output_network(weight)
        else:
            if self.H.adaptiveoutputs or self.H.adaptiveoutputs_tied:
                weight = self._decoder(torch.arange(self.H.ntoken).cuda())
            elif (self.H.char_emb or self.H.cnnsoftmax):
                weight = self.batch_combined_enc(l=0, r=self.H.ntoken)
                if self.H.cnncorr:
                    corr = self._cnnsoftmax_M(self._cnnsoftmax_correction.weight)
                    weight = weight + corr.squeeze()
            else:
                weight = self._lookup.weight if self.H.tied or self.H.joint_emb is not None else self._decoder.weight

            weight = self.output_dropout(weight)

        return weight

    def embed_inputs(self, input, weight=None):
        """
            Embed inputs using the given output embedding or based on the
            corresponding input embedding of each method.
        """
        emb = None
        if emb is None:
            if self.H.adaptiveoutputs or self.H.cnnsoftmax:
                emb = self._lookup(input).view(input.shape[0], input.shape[1], self.H.emsize)
            elif self.H.adaptiveoutputs_tied or self.H.char_emb:
                emb = weight[input.view(-1)].view(input.shape[0], input.shape[1], self.H.emsize)
            else:
                emb = self._lookup(input).view(input.shape[0], input.shape[1], self.H.emsize)
        emb = self._lockdrop(emb, self.H.dropouti if self.use_dropout else 0)
        return emb

    def output_bias(self, weight):
        """
            Return the output bias for the full vocabulary.
        """
        if self.H.predict_bias:
            bias = self.estimate_bias(weight)
        else:
            if hasattr(self, '_bias'):
                bias = self._bias.weight
            elif self.H.tied or self.H.fullsoftmax:
                bias = self._decoder.bias
            elif hasattr(self, 'decoder'):
                bias = self._decoder.weight
        return bias.view(-1)

    def forward(self, input, hidden, return_h=False, eval_mode=False):
        """
            This function makes a forward pass of the input tensor and returns the components for computing the logits as well as optionally the hidden states of the rnn.
        """
        # Load output embedding
        if eval_mode:
            if not hasattr(self, "cached_weight"):
                weight = self.output_embedding(0, self.H.ntoken)
                self.cached_weight = weight
            else:
                weight = self.cached_weight
        else:
            weight = self.output_embedding(0, self.H.ntoken)

        # Embed input tensor
        emb = self.embed_inputs(input, weight=weight)

        # Encode the prefixes of the input tensor
        raw_output, raw_outputs, outputs, hidden = self.rnn_pass(emb, hidden)
        output = self._lockdrop(raw_output, self.H.dropout if self.use_dropout else 0)
        result = output.view(output.size(0)*output.size(1), output.size(2))
        outputs.append(output)

        # Apply output transformation
        if self.H.joint_emb is not None:
            weight = self.batch_apply_output_network(weight)

        # Load bias
        bias = self.output_bias(weight=weight)

        if return_h:
            return result, weight, bias, hidden, raw_outputs, outputs
        else:
            return result, weight, bias, hidden
