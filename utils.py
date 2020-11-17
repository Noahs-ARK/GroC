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
import os, shutil
import numpy as np
import pickle

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()

    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(model, criterion, optimizer, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(criterion, os.path.join(path, 'finetune_criterion.pt'))
        torch.save(optimizer, os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(criterion, os.path.join(path, 'criterion.pt'))
        torch.save(optimizer, os.path.join(path, 'optimizer.pt'))

def consecutive_groups(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def load_criterion(args, ntokens, logging):
    """
	Function to load different criteria depending on the vocabulary splits and size.
    """
    logging("Using no splits; vocab size {}".format(ntokens))
    criterion = torch.nn.CrossEntropyLoss()
    return criterion



def store_word_ce(args, data_source, model, corpus, criterion, fname='out'):
    """
       Store the cross-entropy loss per word in the vocabulary.
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    batch_size = data_source.shape[1]
    hidden = model.init_hidden(batch_size)

    # Initialize vocabulary structure to store the crossentropy losses.
    vocab, words = {}, corpus.dictionary.idx2word

    # Add the loss per word in the vocabulary structure for each different context.
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, weight, bias, hidden = model(data, hidden)
        pred_targets = torch.mm(output, weight.t()) + bias
        for j, target in enumerate(targets):
            target_loss = criterion(pred_targets[j:j+1], targets[j:j+1]).data
            word = words[target.tolist()]
            if word in vocab:
                vocab[word].append(target_loss.tolist())
            else:
                vocab[word] = [target_loss.tolist()]
        hidden = repackage_hidden(hidden)

    # Store the vocabulary to the disk.
    pickle.dump(vocab, open(fname+'.pkl','wb'))


def get_external_knowledge(args, corpus):
    """
        Function to extract surface, relational, and definitional
        features from an external knowledge base.
    """
    char_arr, char_vocab = None, None
    rel_arr, def_arr = None, None
    if (hasattr(args, "char_emb") and args.char_emb) or \
       (hasattr(args, "cnnsoftmax") and args.cnnsoftmax):
        # extract sufrface forms
        from allennlp.modules.elmo import batch_to_ids
        all_words =  corpus.dictionary.idx2word
        all_words.append('') # empty pad token
        char_arr, char_vocab = batch_to_ids([all_words]), None
        char_arr = char_arr[:,:,:args.max_charlen]
        max_dlen = args.max_deflen
        max_rlen = args.max_rellen
        # extract relational and definitional forms
        if args.rel_emb or args.def_emb:
            from nltk.corpus import wordnet
            definitions, relations = [], []
            rel_lens, def_lens = [], []
            count, rels, defs = 0, 0, 0
            print ("Loading external info...")
            for word in all_words:
                synset = wordnet.synsets(word)
                cur_def = [len(all_words)-1 for i in range(max_dlen)]
                cur_rel = [len(all_words)-1 for i in range(max_rlen)]
                if len(synset) > 0:
                    synonyms = [l.name() for s in synset for l in s.lemmas() if l.name() in corpus.dictionary.word2idx]
                    synonyms = [corpus.dictionary.word2idx[w] for w in np.unique(synonyms)]
                    top_def = [corpus.dictionary.word2idx[w] for w in synset[0].definition().split() if w in corpus.dictionary.word2idx]
                    cur_rel = synonyms
                    cur_def = top_def
                    if len(synonyms) > 0:
                        rels += 1
                        rel_lens.append(len(synonyms))
                    if len(top_def) > 0:
                        defs += 1
                        def_lens.append(len(top_def))
                cur_rel_padded = np.pad(cur_rel, (0,(500 - len(cur_rel))), constant_values=len(all_words)-1)[:max_rlen].tolist()
                cur_def_padded = np.pad(cur_def, (0,(500 - len(cur_def))), constant_values=len(all_words)-1)[:max_dlen].tolist()
                relations.append(cur_rel_padded)
                definitions.append(cur_def_padded)
                print ("(%d/%d)"%(count, len(all_words)), end="\r")
                count += 1
            print ("====== External knowledge ======")
            print ("[*] Relations (%d/%d)" % (rels, len(relations)) )
            print (" nonzero: %.2f%s" % ( (rels*100/len(relations)), '%') )
            print (" avg: %d" % int(np.mean(rel_lens)))
            print (" max: %d" % int(np.max(rel_lens)))
            print (" min: %d" % int(np.min(rel_lens)))
            print ("[*] Definitions (%d/%d)" % (defs, len(definitions)) )
            print (" nonzero: %.2f%s" % ( (defs*100/len(definitions)), '%') )
            print (" avg: %d" % int(np.mean(def_lens)))
            print (" max: %d" % int(np.max(def_lens)))
            print (" min: %d" % int(np.min(def_lens)))
            print ("==================================")
            if args.rel_emb:
                rel_arr = relations
            if args.def_emb:
                def_arr = definitions
    return char_arr, rel_arr, def_arr
