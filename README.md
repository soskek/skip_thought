# Sentence-level Language Model and Skip-thought Vector

Training script is as follows:

```
python -u train.py -g 3 --train train_data --valid valid_data --vocab vocab.t100 -u 512 --layer 1 --dropout 0.1 --batchsize 128 --out output_dir
```

If you add `--language-model`, a model to be trained is a sentence-level language model.
Otherwise, the model is a skip-thought model by default.

Dataset of training and validation should have one-line-one-sentence format.
Training a skip-thought model uses only neighbor sentences in paragraphs, which are separated by blank lines.

Counting-based vocabulary file `vocab.t100` can be constructed by the script below:

```
python construct_vocab.py --data train_data -t 100 --save vocab.t100
```


### Computation Cost

#### Sentence-level Language Model

For 128 sentence pairs in a minibatch, 512-unit LSTM with vocabulary size of 22231 can process 10 iterations per second on 7.5GB GPU memory.
On dataset with 4,300,000 pairs, training is performed over 5 epoch in 4.5 hours.

### Skip-thought Vector

For 128 sentence pairs in a minibatch, 512-unit GRU with vocabulary size of 22231 can process 2-2.5 iterations per second on 7.5GB GPU memory.
On dataset with 4,000,000 pairs, training is performed over 5 epoch in 18-22 hours.


### Use wikitext103 as Dataset

```
sh prepare_rawwikitext.sh
```

```
PYTHONIOENCODING=utf-8 python preprocess_spacy.py datasets/wikitext-103-raw/wiki.train.raw > datasets/wikitext-103-raw/spacy_wikitext-103-raw.train
PYTHONIOENCODING=utf-8 python preprocess_spacy.py datasets/wikitext-103-raw/wiki.valid.raw > datasets/wikitext-103-raw/spacy_wikitext-103-raw.valid
PYTHONIOENCODING=utf-8 python preprocess_spacy.py datasets/wikitext-103-raw/wiki.test.raw > datasets/wikitext-103-raw/spacy_wikitext-103-raw.test
```

```
PYTHONIOENCODING=utf-8 python preprocess_after_spacy.py datasets/wikitext-103-raw/spacy_wikitext-103-raw.train > datasets/wikitext-103-raw/spacy_wikitext-103-raw.train.after
PYTHONIOENCODING=utf-8 python preprocess_after_spacy.py datasets/wikitext-103-raw/spacy_wikitext-103-raw.valid > datasets/wikitext-103-raw/spacy_wikitext-103-raw.valid.after
PYTHONIOENCODING=utf-8 python preprocess_after_spacy.py datasets/wikitext-103-raw/spacy_wikitext-103-raw.test > datasets/wikitext-103-raw/spacy_wikitext-103-raw.test.after
```

```
python construct_vocab.py --data datasets/wikitext-103-raw/spacy_wikitext-103-raw.train.after -t 100 --save datasets/wikitext-103-raw/spacy_wikitext-103-raw.train.after.vocab.t100
```

```
python -u train.py -g 3 --train datasets/wikitext-103-raw/spacy_wikitext-103-raw.train.after --valid datasets/wikitext-103-raw/spacy_wikitext-103-raw.valid.after --vocab datasets/wikitext-103-raw/spacy_wikitext-103-raw.train.after.vocab.t100 -u 512 --layer 1 --dropout 0.1 --batchsize 128 --out outs/st.u512.l1.d01.b128
```

`--language-model`


---
---

# Efficient Softmax Approximation

Implementations of Blackout and Adaptive Softmax for efficiently calculating word distribution for language modeling of very large vocabularies.

LSTM language models are derived from [rnnlm_chainer](https://github.com/soskek/rnnlm_chainer).

Available output layers are as follows

- Linear + softmax with cross entropy loss. A usual output layer.
- `--share-embedding`: A variant using the word embedding matrix shared with the input layer for the output layer.
- `--adaptive-softmax`: [Adaptive softmax](http://proceedings.mlr.press/v70/grave17a/grave17a.pdf)
- `--blackout`: [BlackOut](https://arxiv.org/pdf/1511.06909.pdf) (BlackOut is not faster on GPU.)

### Adaptive Softmax

- Efficient softmax approximation for GPUs
- Edouard Grave, Armand Joulin, Moustapha Cissé, David Grangier, Hervé Jégou, ICML 2017
- [paper](http://proceedings.mlr.press/v70/grave17a/grave17a.pdf)
- [authors' Lua code](https://github.com/facebookresearch/adaptive-softmax)

### BlackOut

- BlackOut: Speeding up Recurrent Neural Network Language Models With Very Large Vocabularies
- Shihao Ji, S. V. N. Vishwanathan, Nadathur Satish, Michael J. Anderson, Pradeep Dubey, ICLR 2016
- [paper](https://arxiv.org/pdf/1511.06909.pdf)
- [authors' C++ code](https://github.com/IntelLabs/rnnlm)

# How to Run

```
python -u train.py -g 0
```

## Datasets

- PennTreeBank
- Wikitext-2
- Wikitext-103

For wikitext, run `prepare_wikitext.sh` for downloading the datasets.
