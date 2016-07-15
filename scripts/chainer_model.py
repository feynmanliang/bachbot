import click

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import codecs
import re
import six
import sys
import datetime

from constants import *

@click.group()
def chainer_model():
    """Models built with Python chainer."""
    pass

class RNNForLM(Chain):
    def __init__(self, n_vocab, n_embed, n_units, train=True):
        super(RNNForLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_embed),
            l1=L.LSTM(n_embed, n_units),
            bn1=L.BatchNormalization(n_units),
            l2=L.LSTM(n_units, n_units),
            bn2=L.BatchNormalization(n_units),
            out=L.Linear(n_units, n_vocab),
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-1, 0.1, param.data.shape)
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        xe = self.embed(x)
        h1 = self.bn1(self.l1(xe), test=not self.train)
        h2 = self.bn2(self.l2(h1), test=not self.train)
        y = self.out(h2)
        return y

class ParallelSequentialIterator(chainer.dataset.Iterator):
    """
    Dataset iterator to create a batch of sequences at different positions.

    This iterator returns a pair of current words and the next words. Each
    example is a part of sequences starting from the different offsets
    equally spaced within the whole sequence.
    """

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0 # incremented if every word is visited at least once since prior increment
        self.is_new_epoch = False # True if the epoch is incremented at the last iteration.
        self.repeat = repeat
        length = len(dataset)

        # Offsets maintain the position of each sequence in the minibatch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]

        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0

    def __next__(self):
        """
        This iterator returns a list representing a mini batch.

        Each item indicates a different position in the original sequence. Each
        item is represented by a pair of two word IDs. The first word is at the
        "current" position, while the second word at the next position. At each
        iteration, the iteration count is incremented, which pushes forward the
        "current" position.
        """
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)

class ParallelSlidingIterator(chainer.dataset.Iterator):
    """
    Dataset iterator to create sliding windows at different offsets.

    The preceeding `index:index+seq_length-1` words are used to predict
    the next word at `index+seq_length`.
    """

    def __init__(self, dataset, batch_size, seq_length, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.epoch = 0 # incremented if every word is visited at least once since prior increment
        self.is_new_epoch = False # True if the epoch is incremented at the last iteration.
        self.repeat = repeat
        length = len(dataset)

        # Offsets maintain the position of each sequence in the minibatch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]

        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0

    def __next__(self):
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size + self.seq_length >= length:
            raise StopIteration
        curr_contexts = self.get_current_contexts()
        next_words = self.get_next_words()
        self.iteration += 1

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(curr_contexts, next_words))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    def get_current_contexts(self):
        for offset in self.offsets:
            start_idx = (offset + self.iteration) % len(self.dataset)
            end_idx = start_idx + self.seq_length % len(self.dataset)
            yield self.dataset[start_idx:end_idx]

    def get_next_words(self):
        for offset in self.offsets:
            yield self.dataset[(offset + self.iteration + self.seq_length) % len(self.dataset)]

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class BPTTUpdater(training.StandardUpdater):
    "Custom updater for truncated BackProp Through Time (BPTT)."
    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            if train_iter.is_new_epoch: # reset LSTM state in between epochs
                optimizer.target.predictor.reset_state()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.zerograds()  # Initialize the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

class BPTTParallelUpdater(training.ParallelUpdater):
    "Custom updater for truncated BackProp Through Time (BPTT)."
    def __init__(self, train_iter, optimizer, bprop_len, devices):
        super(BPTTParallelUpdater, self).__init__(
            train_iter, optimizer, devices=devices)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        optimizer = self.get_optimizer('main')
        model_main = optimizer.target
        models_others = {k: v for k, v in self._models.items()
                if v is not model_main}

        for model in six.itervalues(self._models):
            model.zerograds()

        # Progress the dataset iterator for bprop_len words at each iteration.
        train_iter = self.get_iterator('main')
        losses = { model_key: 0 for model_key in self._models }
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()
            if train_iter.is_new_epoch: # reset LSTM state in between epochs
                for model in six.itervalues(self._models):
                    model.predictor.reset_state()

            # Split the batch to sub-batches.
            n = len(self._models)
            in_arrays_list = {}
            for j, key in enumerate(six.iterkeys(self._models)):
                # Concatenate the word IDs to matrices and send them to the devices
                # self.converter does this job
                # (it is chainer.dataset.concat_examples by default)
                in_arrays_list[key] = self.converter(
                    batch[j::n], self._devices[key])

            for model_key, model in six.iteritems(self._models):
                with cuda.get_device(self._devices[model_key]):
                    in_arrays = in_arrays_list[model_key]
                    loss_func = self.loss_func or model

                    if isinstance(in_arrays, tuple):
                        in_vars = tuple(chainer.Variable(x) for x in in_arrays)
                        losses[model_key] += loss_func(*in_vars)
                    elif isinstance(in_arrays[0], dict):
                        in_vars = {key: chainer.Variable(x)
                                for key, x in six.iteritems(in_arrays)}
                        losses[model_key] += loss_func(**in_vars)
                    else:
                        in_vars = chainer.Variable(in_arrays)
                        losses[model_key] += loss_func(in_vars)

        for _, loss in six.iteritems(losses):
            loss.backward() # Parallel backprop
            loss.unchain_backward()  # Truncate the graph

        for model in six.itervalues(models_others): # Accumulate gradients across GPUs
            model_main.addgrads(model)

        optimizer.update()  # Update the parameters on main GPU

        for model in six.itervalues(models_others): # Broadcast updated model to GPUs
            model.copyparams(model_main)

def compute_perplexity(result):
    "Routine to rewrite the result dictionary of LogReport to add perplexity values."
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])
    return result

def log_runtime(start_time):
    "Curried function to log runtimes."
    def _fn(result):
        result['runtime'] = (datetime.datetime.now() - start_time).total_seconds()
        return result
    return _fn

@click.command()
@click.option('--corpus', type=click.File('rb'),
        help='Training corpus. Each line should contain one unicode symbol.')
@click.option('--batchsize', '-b', type=int, default=50,
        help='Number of examples in each mini batch')
@click.option('--bproplen', '-l', type=int, default=64,
        help='Number of words in each mini batch '
        '(= length of truncated BPTT)')
@click.option('--iteration', '-i', type=int, default=1000,
        help='Number of training iterations')
@click.option('--gpu', '-g', type=int, default=0,
        help='GPU ID (negative value indicates CPU)')
@click.option('--gradclip', '-c', type=float, default=5,
        help='Gradient norm threshold to clip')
@click.option('--out', '-o', default='result',
        help='Directory to output the result')
@click.option('--resume', '-r', default='',
        help='Resume the training from snapshot')
@click.option('--quicktest', type=bool, default=False,
        help='Use tiny datasets for quick tests')
@click.option('--embed', '-e', type=int, default=64,
                    help='dimension of note vector embeddings')
@click.option('--unit', '-u', type=int, default=128,
                    help='number of units')
def train(corpus, batchsize, bproplen, iteration, gpu, gradclip, out, resume, quicktest, embed, unit):
    # Load data
    words = filter(lambda x: x != u'\n', codecs.open(SCRATCH_DIR + '/concat_corpus.txt', "r", "utf-8").read())
    if quicktest:
        words = words[:1000]
    idx_to_word = dict(enumerate(sorted(set(words))))
    word_to_idx = { v:k for k,v in idx_to_word.items() }
    n_vocab = len(word_to_idx)
    print('#vocab =', n_vocab)

    train = np.array(map(word_to_idx.get, words), dtype=np.int32)
    # train = np.array(map(word_to_idx.get, words[:int(0.8*len(words))]), dtype=np.int32)
    # val = np.array(map(word_to_idx.get, words[int(0.8*len(words)):int(0.9*len(words))]), dtype=np.int32)
    # test = np.array(map(word_to_idx.get, words[int(0.9*len(words)):]), dtype=np.int32)

    train_iter = ParallelSequentialIterator(train, batchsize)
    #train_iter = ParallelSlidingIterator(train, batchsize, bproplen)
    # val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    # test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    rnn = RNNForLM(n_vocab, embed, unit)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity

    # Set up an optimizer
    optimizer = chainer.optimizers.RMSprop()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, bproplen, gpu)
    with cuda.get_device(gpu):
        model.to_gpu()

    # devices = { 'main': 0, 'GTX660': 1, 'second': 2 } # TODO: make argument
    # updater = BPTTParallelUpdater(train_iter, optimizer, bproplen, devices=devices)

    trainer = training.Trainer(updater, (iteration, 'iteration'), out=out)

    interval = 2 if quicktest else 10
    # with cuda.get_device(gpu):
    #     eval_model = model.copy()  # Model with shared params and distinct states
    #     eval_model.to_gpu()
    # eval_rnn = eval_model.predictor
    # eval_rnn.train = False
    # trainer.extend(extensions.Evaluator(
    #     val_iter, eval_model, device=gpu,
    #     # Reset the RNN state at the beginning of each evaluation
    #     eval_hook=lambda _: eval_rnn.reset_state()
    # ), trigger=(interval, 'iteration'))

    start_time = datetime.datetime.now()
    trainer.extend(extensions.LogReport(
        postprocess=lambda result: log_runtime(start_time)(compute_perplexity(result)),
        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'runtime', 'main/loss', 'perplexity', 'val_perplexity']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.snapshot(),
        trigger=(interval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'),
        trigger=(interval, 'iteration'))
    if resume:
        chainer.serializers.load_npz(resume, trainer)

    print('Running trainer')
    trainer.run()

    # Evaluate the final model
    # print('test')
    # eval_rnn.reset_state()
    # evaluator = extensions.Evaluator(test_iter, eval_model, device=gpu)
    # result = evaluator()
    # print('test perplexity:', np.exp(float(result['main/loss'])))

@click.command()
@click.option('--corpus', type=click.File('rb'),
        help='Training corpus. Each line should contain one unicode symbol.')
@click.option('--modelpath', '-m', type=str, required=True,
                    help='model data, saved by train_ptb.py')
@click.option('--primetext', '-p', type=str, required=True,
                    default='',
                    help='base text data, used for text generation')
@click.option('--seed', '-s', type=int, default=42,
                    help='random seeds for text generation')
@click.option('--embed', '-e', type=int, default=64,
                    help='dimension of note vector embeddings')
@click.option('--unit', '-u', type=int, default=128,
                    help='number of units')
@click.option('--sample', type=int, default=1,
                    help='negative value indicates NOT use random choice')
@click.option('--length', type=int, default=20,
                    help='length of the generated text')
@click.option('--quicktest', type=bool, default=False,
        help='Use tiny datasets for quick tests')
@click.option('--gpu', type=int, default=0,
                    help='GPU ID (negative value indicates CPU)')
@click.option('--temperature', type=float, default=1.0,
                    help='Temperature of sampling softmax')
def sample(corpus, modelpath, primetext, seed, embed, unit, sample, length, quicktest, gpu, temperature):
    np.random.seed(seed)

    xp = cuda.cupy if gpu >= 0 else np

    # Load data
    words = filter(lambda x: x != u'\n', codecs.open(SCRATCH_DIR + '/concat_corpus.txt', "r", "utf-8").read())
    if quicktest:
        words = words[:1000]
    idx_to_word = dict(enumerate(sorted(set(words))))
    word_to_idx = { v:k for k,v in idx_to_word.items() }
    n_vocab = len(word_to_idx)

    n_units = unit
    n_embed = embed

    lm = RNNForLM(n_vocab, n_embed, n_units, train=False)
    model = L.Classifier(lm)
    serializers.load_npz(modelpath, model)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    model.predictor.reset_state()

    primetext = primetext
    if isinstance(primetext, six.binary_type):
        primetext = primetext.decode('utf-8')

    if primetext in word_to_idx:
        prev_word = chainer.Variable(xp.array([word_to_idx[primetext]], xp.int32))
    else:
        print('ERROR: Unfortunately ' + primetext + ' is unknown.')
        exit()

    sys.stdout.write(primetext + '\n')

    for i in six.moves.range(length):
        prob = F.softmax(model.predictor(prev_word) / temperature)
        if sample > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))

        sys.stdout.write(idx_to_word[index] + '\n')

        prev_word = chainer.Variable(xp.array([index], dtype=xp.int32))

    sys.stdout.write('\n')

map(chainer_model.add_command, [
    train,
    sample
])
