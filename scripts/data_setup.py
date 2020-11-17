import os
import sys
import shutil
import pathlib
import argparse

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--train', nargs='*', default=None,
                    help='source file for your training set (omit for empty file)')
parser.add_argument('--valid', nargs='*', default=None,
                    help='source file for your validation set (omit for empty file)')
parser.add_argument('--test', nargs='*', default=None,
                    help='source file for your test set (omit for empty file)')
parser.add_argument('--save', type=str, default='news.2007_copy',
                    help='where to save the newly constructed dataset (under args.data_base)')

args = parser.parse_args()

if os.path.exists(args.save):
    raise FileExistsError(f"target directory {args.save} already exists! rename or remove")

os.makedirs(args.save)

train_dst = os.path.join(args.save, 'train.txt')
if args.train is not None:
    with open(train_dst,'w') as df:
        for sf in args.train:
            shutil.copyfileobj(open(sf,'r'), df)
else:
    pathlib.Path(train_dst).touch()

valid_dst = os.path.join(args.save, 'valid.txt')
if args.valid is not None:
    with open(valid_dst,'w') as df:
        for sf in args.valid:
            shutil.copyfileobj(open(sf,'r'), df)
else:
    pathlib.Path(valid_dst).touch()

test_dst = os.path.join(args.save, 'test.txt')
if args.test is not None:
    with open(test_dst,'w') as df:
        for sf in args.test:
            shutil.copyfileobj(open(sf,'r'), df)
else:
    pathlib.Path(test_dst).touch()
