# Script to reproduce the data splits from the following paper:
#
# Unbounded cache model for online language modeling with open vocabulary
# Edouard Grave, Moustapha Cisse, Armand Joulin
# NIPS 2017

preprocess() {
    perl tools/tokenizer.perl | tr "[:upper:]" "[:lower:]"
}

subfile() {
    tail -n +$1 $3 | head -n $2
}

mkdir news.2007
subfile 1 87000 raw/news.2007.en.shuffled | preprocess > news.2007/train.txt
subfile 87001 8700 raw/news.2007.en.shuffled | preprocess > news.2007/valid.txt
subfile 95701 440000 raw/news.2007.en.shuffled | preprocess > news.2007/test.txt

mkdir news.2008
subfile 1 87000 raw/news.2008.en.shuffled | preprocess > news.2008/train.txt
subfile 87001 8700 raw/news.2008.en.shuffled | preprocess > news.2008/valid.txt
subfile 95701 440000 raw/news.2008.en.shuffled | preprocess > news.2008/test.txt

mkdir news.2009
subfile 1 87000 raw/news.2009.en.shuffled | preprocess > news.2009/train.txt
subfile 87001 8700 raw/news.2009.en.shuffled | preprocess > news.2009/valid.txt
subfile 95701 440000 raw/news.2009.en.shuffled | preprocess > news.2009/test.txt

mkdir news.2010
subfile 1 87000 raw/news.2010.en.shuffled | preprocess > news.2010/train.txt
subfile 87001 8700 raw/news.2010.en.shuffled | preprocess > news.2010/valid.txt
subfile 95701 440000 raw/news.2010.en.shuffled | preprocess > news.2010/test.txt

mkdir news.2011
subfile 1 87000 raw/news.2011.en.shuffled | preprocess > news.2011/train.txt
subfile 87001 8700 raw/news.2011.en.shuffled | preprocess > news.2011/valid.txt
subfile 95701 440000 raw/news.2011.en.shuffled | preprocess > news.2011/test.txt

mkdir web
# could not find original .shuffled data; manually (re?)-shuffle with shuffle.py after this
# subfile 1 82000 raw/commoncrawl.de-en.en.shuffled | preprocess > web/train.txt
# subfile 82001 8200 raw/commoncrawl.de-en.en.shuffled | preprocess > web/valid.txt
# subfile 90201 410000 raw/commoncrawl.de-en.en.shuffled | preprocess > web/nelm.txt
subfile 1 82000 raw/commoncrawl.de-en.en | preprocess > web/train.txt
subfile 82001 8200 raw/commoncrawl.de-en.en | preprocess > web/valid.txt
subfile 90201 410000 raw/commoncrawl.de-en.en | preprocess > web/test.txt

mkdir news.comm
# could not find original .shuffled data; manually (re?)-shuffle with shuffle.py after this
# subfile 1 400000 raw/training-monolingual/news-commentary-v11.en.shuffled | preprocess > news.comm/nelm.txt
subfile 1 400000 raw/training-monolingual-nc-v11/news-commentary-v11.en | preprocess > news.comm/test.txt

mkdir wiki
subfile 1 36718 raw/wikitext-103-raw/wiki.train.raw | tr "[:upper:]" "[:lower:]" > wiki/train.txt
cat raw/wikitext-103-raw/wiki.valid.raw | tr "[:upper:]" "[:lower:]" > wiki/valid.txt
cat raw/wikitext-103-raw/wiki.test.raw | tr "[:upper:]" "[:lower:]" > wiki/test.txt
subfile 36719 184000 raw/wikitext-103-raw/wiki.train.raw | tr "[:upper:]" "[:lower:]" > wiki/nelm.txt

# added by <anonymized>, 4/12/2020:
python shuffle.py web/*.txt news.comm/*.txt

# create subsets for open-vocab experiments
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --save news2007_train/
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test news.2007/test.txt --save news2007_train.news2007_test
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test news.2008/test.txt --save news2007_train.news2008_test
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test news.2009/test.txt --save news2007_train.news2009_test
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test news.2010/test.txt --save news2007_train.news2010_test
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test news.2011/test.txt --save news2007_train.news2011_test
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test wiki/test.txt --save news2007_train.wiki_test
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test web/test.txt --save news2007_train.web_test

python scripts/data_setup.py --train wiki/train.txt --valid wiki/valid.txt --save wiki_train/
python scripts/data_setup.py --train wiki/train.txt --valid wiki/valid.txt --test web/test.txt --save wiki_train.web_test

# create fine-tuning subsets
# in order to be comparable to the cache/change_vocab models, we don't keep all the extra target domain training vocab during testing. we *do* get to keep words that appear in both target domain training and target domain test.

# 2007-->2008 (does not keep 2008 train/valid vocab during testing unless it appears in test)
python scripts/data_setup.py --train news.2008/train.txt --valid news.2008/valid.txt --test news.2007/train.txt news.2007/valid.txt --save news2007_train.news2008_finetune
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test news.2008/test.txt --save news2007_train.news2008_finetune.news2008_test

# 2007-->2009 (does not keep 2009 train/valid vocab during testing unless it appears in test)
python scripts/data_setup.py --train news.2009/train.txt --valid news.2009/valid.txt --test news.2007/train.txt news.2007/valid.txt --save news2007_train.news2009_finetune
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test news.2009/test.txt --save news2007_train.news2009_finetune.news2009_test

# 2007-->2010 (does not keep 2010 train/valid vocab during testing unless it appears in test)
python scripts/data_setup.py --train news.2010/train.txt --valid news.2010/valid.txt --test news.2007/train.txt news.2007/valid.txt --save news2007_train.news2010_finetune
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test news.2010/test.txt --save news2007_train.news2010_finetune.news2009_test

# 2007-->2011 (does not keep 2011 train/valid vocab during testing unless it appears in test)
python scripts/data_setup.py --train news.2011/train.txt --valid news.2011/valid.txt --test news.2007/train.txt news.2007/valid.txt --save news2007_train.news2011_finetune
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test news.2011/test.txt --save news2007_train.news2011_finetune.news2011_test

# 2007-->wiki (does not keep wiki train/valid vocab during testing unless it appears in test)
python scripts/data_setup.py --train wiki/train.txt --valid wiki/valid.txt --test news.2007/train.txt news.2007/valid.txt --save news2007_train.wiki_finetune
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test wiki/test.txt --save news2007_train.wiki_finetune.wiki_test

# 2007-->web (does not keep web train/valid vocab during testing unless it appears in test)
python scripts/data_setup.py --train web/train.txt --valid web/valid.txt --test news.2007/train.txt news.2007/valid.txt --save news2007_train.web_finetune
python scripts/data_setup.py --train news.2007/train.txt --valid news.2007/valid.txt --test web/test.txt --save news2007_train.web_finetune.web_test

# wiki-->web (does not keep web train/valid vocab during testing unless it appears in test)
python scripts/data_setup.py --train web/train.txt --valid web/valid.txt --test wiki/train.txt wiki/valid.txt --save wiki_train.web_finetune
python scripts/data_setup.py --train wiki/train.txt --valid wiki/valid.txt --test web/test.txt --save wiki_train.web_finetune.web_test


