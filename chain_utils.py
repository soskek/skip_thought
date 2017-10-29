#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import collections
import io
import json
import os

import numpy as np
import progressbar

import chainer
from chainer import cuda
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import utils
from chainer.dataset import dataset_mixin


def convert_xt_batch_seq(xt_batch_seq, gpu):
    batchsize = len(xt_batch_seq[0])
    seq_len = len(xt_batch_seq)
    xt_batch_seq = np.array(xt_batch_seq, 'i')
    # (bproplen, batch, 2)
    xt_batch_seq = convert.to_device(gpu, xt_batch_seq)
    xp = cuda.get_array_module(xt_batch_seq)
    x_seq_batch = xp.split(
        xt_batch_seq[:, :, 0].T.reshape(batchsize * seq_len),
        batchsize, axis=0)
    t_seq_batch = xp.split(
        xt_batch_seq[:, :, 1].T.reshape(batchsize * seq_len),
        batchsize, axis=0)
    return x_seq_batch, t_seq_batch


def convert_sequence_chain(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return [to_device_batch([x[i] for x in batch])
            for i in range(len(batch[0]))]
    """
    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}
    """


def count_words_from_file(counts, file_path):
    bar = progressbar.ProgressBar()
    for l in bar(io.open(file_path, encoding='utf-8')):
        # TODO: parallel
        if l.strip():
            words = l.strip().split()
            for word in words:
                counts[word] += 1
    return counts


def make_chain_dataset(path, vocab, chain_length=2):
    dataset = []
    chain = []
    unk_id = vocab['<unk>']

    def make_array(chain):
        array_chain = []
        for words in chain:
            tokens = []
            for word in words:
                tokens.append(vocab.get(word, unk_id))
            array_chain.append(np.array(tokens, 'i'))
        return array_chain

    bar = progressbar.ProgressBar()
    n_lines = sum(1 for _ in io.open(path, encoding='utf-8'))
    for line in bar(io.open(path, encoding='utf-8'),
                    max_value=n_lines):
        if not line.strip():
            if len(chain) >= chain_length:
                dataset.append(make_array(chain))
                chain = []
            continue
        words = line.strip().split()
        if len(words) > 50:
            words = words[:50]
        words = ['<eos>'] + words + ['<eos>']
        chain.append(words)
    if len(chain) >= chain_length:
        dataset.append(make_array(chain))
    return dataset


class SequenceChainDataset(dataset_mixin.DatasetMixin):

    def __init__(self, path, vocab, chain_length=2):
        self._dataset = make_chain_dataset(
            path,
            vocab=vocab,
            chain_length=chain_length)
        self.chain_length = chain_length
        self._subchain_numbers = [
            len(chain) + 1 - chain_length
            for chain in self._dataset]
        self._length = sum(self._subchain_numbers)

        self._idx2subchain_keys = []
        chain_idx = 0
        for l in self._subchain_numbers:
            for i in range(l):
                self._idx2subchain_keys.append((chain_idx, i))
            chain_idx += 1

    def __len__(self):
        return self._length

    def get_subchain(self, subchain_keys):
        chain_idx, sub_idx = subchain_keys
        chain = self._dataset[chain_idx]
        subchain = chain[sub_idx: sub_idx + self.chain_length]
        return subchain

    def get_example(self, i):
        return self.get_subchain(self._idx2subchain_keys[i])
