#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import reporter

from black_out import BlackOut
from adaptive_softmax import AdaptiveSoftmaxOutputLayer


def embed_seq_batch(embed, seq_batch, dropout=0., context=None):
    x_len = [len(seq) for seq in seq_batch]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(seq_batch, axis=0))
    ex = F.dropout(ex, dropout)
    if context is not None:
        ids = [embed.xp.full((l, ), i).astype('i')
               for i, l in enumerate(x_len)]
        ids = embed.xp.concatenate(ids, axis=0)
        cx = F.embed_id(ids, context)
        ex = F.concat([ex, cx], axis=1)
    exs = F.split_axis(ex, x_section, 0)
    return exs


class BlackOutOutputLayer(BlackOut):
    def output_and_loss(self, h, t):
        if chainer.config.train:
            return super(BlackOutOutputLayer, self).__call__(h, t)
        else:
            logit = self(h)
            return F.softmax_cross_entropy(
                logit, t, normalize=False, reduce='mean')

    def __call__(self, h):
        return F.linear(h, self.W)

    def output(self, h, t=None):
        return self(h)


class NormalOutputLayer(L.Linear):
    def __init__(self, *args, **kwargs):
        super(NormalOutputLayer, self).__init__(*args, **kwargs)

    def output_and_loss(self, h, t):
        logit = self(h)
        return F.softmax_cross_entropy(
            logit, t, normalize=False, reduce='mean')

    def output(self, h, t=None):
        return self(h)


class SharedOutputLayer(chainer.Chain):
    def __init__(self, W, bias=True, scale=True):
        super(SharedOutputLayer, self).__init__()
        self.W = W
        with self.init_scope():
            if bias:
                self.add_param('b', (W.shape[0], ), dtype='f')
                self.b.data[:] = 0.
            else:
                self.b = None
            if scale:
                self.add_param('scale', (1, ), dtype='f')
                self.scale.data[:] = 1.
            else:
                self.scale = None

    def output_and_loss(self, h, t):
        logit = self(h)
        return F.softmax_cross_entropy(
            logit, t, normalize=False, reduce='mean')

    def __call__(self, x):
        out = F.linear(x, self.W, self.b)
        if self.scale is not None:
            out *= F.broadcast_to(self.scale[None], out.shape)
        return out

    def output(self, h, t=None):
        return self(h)


class SkipThoughtModel(chainer.Chain):
    def __init__(self, n_vocab, n_units, n_layers=2, dropout=0.5,
                 rnn='gru',
                 share_embedding=False, blackout_counts=None,
                 adaptive_softmax=False):
        super(SkipThoughtModel, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)

            if rnn == 'lstm':
                RNN = L.NStepLSTM
            elif rnn == 'gru':
                RNN = L.NStepGRU
            else:
                NotImplementedError()
            self.encoder = RNN(n_layers, n_units, n_units, dropout)
            # TODO: shared decoder with preprojection
            self.decoder_fw = RNN(
                n_layers, n_units * 2, n_units, dropout)
            self.decoder_bw = RNN(
                n_layers, n_units * 2, n_units, dropout)

            assert(not (share_embedding and blackout_counts is not None))
            if share_embedding:
                self.output = SharedOutputLayer(self.embed.W)
            elif blackout_counts is not None:
                sample_size = max(500, (n_vocab // 200))
                self.output = BlackOutOutputLayer(
                    n_units, blackout_counts, sample_size)
                print('Blackout sample size is {}'.format(sample_size))
            elif adaptive_softmax:
                self.output = AdaptiveSoftmaxOutputLayer(
                    n_units, n_vocab,
                    cutoff=[2000, 10000], reduce_k=4)
            else:
                self.output = NormalOutputLayer(n_units, n_vocab)
        self.dropout = dropout
        self.n_units = n_units
        self.n_layers = n_layers

        for name, param in self.namedparams():
            if param.ndim != 1:
                # This initialization is applied only for weight matrices
                param.data[...] = np.random.uniform(
                    -0.1, 0.1, param.data.shape)

        self.loss = 0.

    def __call__(self, x):
        raise NotImplementedError()

    def calculate_loss(self, input_chain):
        # TODO: variable length of input_chain
        loss = 0.

        def proc(seq_batch_1, seq_batch_2, encoder, decoder):
            e_seq_batch_1 = self.embed_seq_batch(seq_batch_1)
            s_hs_1 = self.encode_seq_batch(
                e_seq_batch_1, encoder)[0]  # take h at last step
            s_hs_1 = s_hs_1[-1]  # take final layer h
            seq_batch_2_wo_bos = [seq[1:] for seq in seq_batch_2]
            seq_batch_2_wo_eos = [seq[:-1] for seq in seq_batch_2]
            e_seq_batch_2 = self.embed_seq_batch(
                seq_batch_2_wo_eos, context=s_hs_1)
            t_out_batch_2 = self.encode_seq_batch(
                e_seq_batch_2, decoder)[-1]  # take final h at each step
            n_tok = sum(len(s) for s in seq_batch_2_wo_bos)
            return self.output_and_loss_from_seq_batch(
                t_out_batch_2, seq_batch_2_wo_bos,
                normalize=n_tok)

        loss_fw = proc(input_chain[0], input_chain[1],
                       self.encoder, self.decoder_fw)
        loss_bw = proc(input_chain[1], input_chain[0],
                       self.encoder, self.decoder_bw)
        loss = (loss_fw + loss_bw) / 2.  # Note this is macro average
        reporter.report({'FWperp': self.xp.exp(loss_fw.data)}, self)
        reporter.report({'BWperp': self.xp.exp(loss_bw.data)}, self)
        reporter.report({'perp': self.xp.exp(loss.data)}, self)
        return loss

    def embed_seq_batch(self, x_seq_batch, context=None):
        e_seq_batch = embed_seq_batch(
            self.embed, x_seq_batch,
            dropout=self.dropout,
            context=context)
        return e_seq_batch

    def encode_seq_batch(self, e_seq_batch, encoder):
        if isinstance(encoder, L.NStepLSTM):
            hs, cs, y_seq_batch = encoder(None, None, e_seq_batch)
            return hs, cs, y_seq_batch
        else:
            hs, y_seq_batch = encoder(None, e_seq_batch)
            return hs, y_seq_batch

    def output_and_loss_from_seq_batch(self, y_seq_batch, t_seq_batch, normalize=None):
        y = F.concat(y_seq_batch, axis=0)
        y = F.dropout(y, ratio=self.dropout)
        t = F.concat(t_seq_batch, axis=0)
        loss = self.output.output_and_loss(y, t)
        if normalize is not None:
            loss *= 1. * t.shape[0] / normalize
        else:
            loss *= t.shape[0]
        return loss

    def pop_loss(self):
        # This is for auxiliary loss
        loss = self.loss
        self.loss = 0.
        return loss


class SentenceLanguageModel(SkipThoughtModel):
    def __init__(self, n_vocab, n_units, n_layers=2, dropout=0.5,
                 rnn='lstm',
                 share_embedding=False, blackout_counts=None,
                 adaptive_softmax=False):
        super(SentenceLanguageModel, self).__init__(
            n_vocab, n_units, n_layers, dropout,
            rnn,
            share_embedding,
            blackout_counts,
            adaptive_softmax)
        delattr(self, 'encoder')
        delattr(self, 'decoder_fw')
        delattr(self, 'decoder_bw')

        with self.init_scope():
            if rnn == 'lstm':
                RNN = L.NStepLSTM
            elif rnn == 'gru':
                RNN = L.NStepGRU
            else:
                NotImplementedError()
            self.rnn = RNN(n_layers, n_units, n_units, dropout)

        for name, param in self.namedparams():
            if param.ndim != 1:
                # This initialization is applied only for weight matrices
                param.data[...] = np.random.uniform(
                    -0.1, 0.1, param.data.shape)

    def calculate_loss(self, input_chain):
        # TODO: variable length of input_chain

        seq_batch = input_chain[0]
        seq_batch_wo_bos = [seq[1:] for seq in seq_batch]
        seq_batch_wo_eos = [seq[:-1] for seq in seq_batch]
        e_seq_batch = self.embed_seq_batch(seq_batch_wo_eos)
        t_out_batch = self.encode_seq_batch(
            e_seq_batch, self.rnn)[-1]  # take final h at each step
        n_tok = sum(len(s) for s in seq_batch_wo_bos)
        loss = self.output_and_loss_from_seq_batch(
            t_out_batch, seq_batch_wo_bos,
            normalize=n_tok)
        reporter.report({'perp': self.xp.exp(loss.data)}, self)
        return loss


class RNNForLM(chainer.Chain):
    # TODO: nstep LSTM
    def __init__(self, n_vocab, n_units, n_layers=2, dropout=0.5,
                 rnn='gru',
                 share_embedding=False, blackout_counts=None,
                 adaptive_softmax=False):
        super(RNNForLM, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            if rnn == 'lstm':
                RNN = L.NStepLSTM
            elif rnn == 'gru':
                RNN = L.NStepGRU
            else:
                NotImplementedError()
            self.rnn = RNN(n_layers, n_units, n_units, dropout)
            assert(not (share_embedding and blackout_counts is not None))
            if share_embedding:
                self.output = SharedOutputLayer(self.embed.W)
            elif blackout_counts is not None:
                sample_size = max(500, (n_vocab // 200))
                self.output = BlackOutOutputLayer(
                    n_units, blackout_counts, sample_size)
                print('Blackout sample size is {}'.format(sample_size))
            elif adaptive_softmax:
                self.output = AdaptiveSoftmaxOutputLayer(
                    n_units, n_vocab,
                    cutoff=[2000, 10000], reduce_k=4)
            else:
                self.output = NormalOutputLayer(n_units, n_vocab)
        self.dropout = dropout
        self.n_units = n_units
        self.n_layers = n_layers

        for name, param in self.namedparams():
            if param.ndim != 1:
                # This initialization is applied only for weight matrices
                param.data[...] = np.random.uniform(
                    -0.1, 0.1, param.data.shape)

        self.loss = 0.
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def __call__(self, x):
        raise NotImplementedError()

    def call_rnn(self, e_seq_batch):
        batchsize = len(e_seq_batch)
        if self.h is None:
            self.h = self.xp.zeros(
                (self.n_layers, batchsize, self.n_units), 'f')
        if self.c is None:
            self.c = self.xp.zeros(
                (self.n_layers, batchsize, self.n_units), 'f')
        self.h, self.c, y_seq_batch = self.rnn(self.h, self.c, e_seq_batch)
        return y_seq_batch

    def encode_seq_batch(self, x_seq_batch):
        e_seq_batch = embed_seq_batch(
            self.embed, x_seq_batch, dropout=self.dropout)
        y_seq_batch = self.call_rnn(e_seq_batch)
        return y_seq_batch

    def forward_seq_batch(self, x_seq_batch, t_seq_batch, normalize=None):
        y_seq_batch = self.encode_seq_batch(x_seq_batch)
        loss = self.output_and_loss_from_seq_batch(
            y_seq_batch, t_seq_batch, normalize)
        return loss

    def output_and_loss_from_seq_batch(self, y_seq_batch, t_seq_batch, normalize=None):
        y = F.concat(y_seq_batch, axis=0)
        y = F.dropout(y, ratio=self.dropout)
        t = F.concat(t_seq_batch, axis=0)
        loss = self.output.output_and_loss(y, t)
        if normalize is not None:
            loss *= 1. * t.shape[0] / normalize
        else:
            loss *= t.shape[0]
        return loss

    def output_from_seq_batch(self, y_seq_batch):
        y = F.concat(y_seq_batch, axis=0)
        y = F.dropout(y, ratio=self.dropout)
        return self.output(y)

    def pop_loss(self):
        loss = self.loss
        self.loss = 0.
        return loss
