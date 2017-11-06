from __future__ import print_function
import argparse
import json

import numpy as np

import chainer
from chainer import cuda
from chainer import training
from chainer.training import extensions
from chainer import serializers

import utils
import chain_utils
import nets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gradclip', '-c', type=float, default=10,
                        help='Gradient norm threshold to clip')

    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--share-embedding', action='store_true')
    parser.add_argument('--blackout', action='store_true')
    parser.add_argument('--adaptive-softmax', action='store_true')

    parser.add_argument('--log-interval',
                        type=int, default=500)
    parser.add_argument('--validation-interval', '--val-interval',
                        type=int, default=30000)
    parser.add_argument('--decay-if-fail', action='store_true')

    parser.add_argument('--vocab', required=True)
    parser.add_argument('--train-path', '--train', required=True)
    parser.add_argument('--valid-path', '--valid', required=True)

    parser.add_argument('--resume')
    parser.add_argument('--resume-rnn')
    parser.add_argument('--resume-wordemb')
    parser.add_argument('--resume-wordemb-vocab')
    parser.add_argument('--init-output-by-embed', action='store_true')

    parser.add_argument('--language-model', action='store_true')
    parser.add_argument('--rnn', default='gru', choices=['lstm', 'gru'])

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    vocab = json.load(open(args.vocab))
    n_vocab = len(vocab)
    print('vocab is loaded', args.vocab)
    print('vocab =', n_vocab)

    if args.language_model:
        train = chain_utils.SequenceChainDataset(
            args.train_path, vocab, chain_length=1)
        valid = chain_utils.SequenceChainDataset(
            args.valid_path, vocab, chain_length=1)
    else:
        train = chain_utils.SequenceChainDataset(
            args.train_path, vocab, chain_length=2)
        valid = chain_utils.SequenceChainDataset(
            args.valid_path, vocab, chain_length=2)

    print('#train =', len(train))
    print('#valid =', len(valid))
    print('#vocab =', n_vocab)

    # Create the dataset iterators
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                  repeat=False, shuffle=False)

    # Prepare an RNNLM model
    if args.blackout:
        counts = utils.count_words(train)
        assert(len(counts) == n_vocab)
    else:
        counts = None

    if args.language_model:
        model = nets.SentenceLanguageModel(
            n_vocab, args.unit, args.layer, args.dropout,
            rnn=args.rnn,
            share_embedding=args.share_embedding,
            blackout_counts=counts,
            adaptive_softmax=args.adaptive_softmax)
    else:
        model = nets.SkipThoughtModel(
            n_vocab, args.unit, args.layer, args.dropout,
            rnn=args.rnn,
            share_embedding=args.share_embedding,
            blackout_counts=counts,
            adaptive_softmax=args.adaptive_softmax)
    print('RNN unit is {}'.format(args.rnn))

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Set up an optimizer
    # optimizer = chainer.optimizers.SGD(lr=1.0)
    # optimizer = chainer.optimizers.Adam(alpha=1e-3, beta1=0.)
    optimizer = chainer.optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(1e-6))

    iter_per_epoch = len(train) // args.batchsize
    log_trigger = (iter_per_epoch // 100, 'iteration')
    eval_trigger = (log_trigger[0] * 50, 'iteration')  # every half epoch

    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=chain_utils.convert_sequence_chain, device=args.gpu,
        loss_func=model.calculate_loss)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(
        valid_iter, model,
        converter=chain_utils.convert_sequence_chain, device=args.gpu,
        eval_func=model.calculate_loss),
        trigger=eval_trigger)
    """
    trainer.extend(utils.SentenceEvaluater(
        model, valid, vocab, 'val/',
        batchsize=args.batchsize,
        device=args.gpu,
        k=args.beam,
        print_sentence_mod=args.print_sentence_mod),
        trigger=eval_trigger)
    """
    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/perp',
        trigger=eval_trigger)
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)

    trainer.extend(extensions.LogReport(trigger=log_trigger),
                   trigger=log_trigger)

    if args.language_model:
        keys = [
            'epoch', 'iteration',
            'main/perp',
            'validation/main/perp',
            'elapsed_time']
    else:
        keys = [
            'epoch', 'iteration',
            'main/perp',
            'main/FWperp',
            'main/BWperp',
            'validation/main/perp',
            'elapsed_time']
    trainer.extend(extensions.PrintReport(keys),
                   trigger=log_trigger)
    trainer.extend(extensions.ProgressBar(update_interval=50))

    print('iter/epoch', iter_per_epoch)
    print('Training start')

    trainer.run()


if __name__ == '__main__':
    main()
