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

import os
import torch

from collections import Counter

import IPython as ipy


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def set_unk(self):
        self.unk = "<UNK>"
        self.unk_id = self.add_word(self.unk)


class Corpus(object):
    def __init__(self, path, use_unk=False):
        self.use_unk = use_unk
        self.dictionary = Dictionary()
        print("Indexing words...")
        self.train = self.store_words(os.path.join(path, 'train.txt'))
        self.valid = self.store_words(os.path.join(path, 'valid.txt'))
        self.test = self.store_words(os.path.join(path, 'test.txt'))
        print("Sorting vocab by frequency...")
        self.order_by_freq()
        if self.use_unk:
            print("Adding UNK token...")
            self.dictionary.set_unk()
        print("Tokenizing text...")
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def store_words(self, path):
        """Stores words from a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

    def order_by_freq(self):
        """Ordering vocab by frequency."""
        dd = self.dictionary.counter
        ord_ids = sorted(dd, key=dd.get)[::-1]
        ord_hash, new_counter = {}, {}
        for j, cur_id in enumerate(ord_ids):
            ord_hash[cur_id] = j
        for word in self.dictionary.word2idx.keys():
            cur_id = self.dictionary.word2idx[word]
            self.dictionary.word2idx[word] = ord_hash[cur_id]
            self.dictionary.idx2word[ord_hash[cur_id]] = word
            replaced_count = dd[cur_id]
            new_counter[cur_id] = dd[ord_ids[cur_id]]
        self.dictionary.counter = new_counter


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        print("starting tokenization")
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx[word]
                    elif self.dictionary.unk is not None:
                        ids[token] = self.dictionary.unk_id
                    else:
                        raise ValueError(f"Unknown word: {word}")
                    token += 1
        return ids
