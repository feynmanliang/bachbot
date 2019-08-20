import click
import codecs
import json
import numpy as np
import random
import subprocess

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

from music21 import note, stream

from constants import *
import decode

@click.command()
@click.option('-i', '--infile', type=click.File('rb'), default=SCRATCH_DIR + '/concat_corpus.txt')
@click.option('-o', '--outdir', type=click.Path(exists=True), default=SCRATCH_DIR)
def make_h5(infile, outdir):
    """Encodes corpus for torch-rnn."""
    fileName=os.path.splitext(os.path.basename(infile.name))[0]

    # preprocess data (tokenize store in hdf5)
    infile_path = os.path.abspath(infile.name)
    print('Processing corpus at: ' + infile_path)
    print('Outputting to: ' + outdir + '/' + fileName + '{h5,json}')
    subprocess.call(' '.join([
        'python',
        '~/torch-rnn/scripts/preprocess.py',
        '--input_txt', infile_path,
        '--output_h5', outdir + '/' + fileName + '.h5',
        '--output_json', outdir + '/' + fileName + '.json'
    ]), shell=True)

class BachBot(nn.Module):
    def __init__(self, vocab_size, wordvec_size, num_layers, hidden_size, dropout):
        assert num_layers >= 1

        super(BachBot, self).__init__()

        self.embedding = nn.Embedding(vocab_size, wordvec_size).cuda()
        self.rnn = nn.LSTM(input_size=wordvec_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=(0 if num_layers == 1 else dropout)).cuda()
        self.linear = nn.Linear(hidden_size, vocab_size).cuda()

    def forward(self, inputs, lengths, hidden=None):
        embedded = self.embedding(inputs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded,
                                                   lengths=lengths,
                                                   enforce_sorted=False)
        out_packed, hidden = self.rnn(packed, hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_packed)
        scores = self.linear(out)

        return scores, hidden

@click.command()
@click.option('-i', '--concat_corpus_txt', type=click.Path(exists=True), default=SCRATCH_DIR + '/concat_corpus.txt')
@click.option('-c', '--checkpoint_path', type=click.Path())
def train(concat_corpus_txt, checkpoint_path):
    """Trains torch-rnn model. Alias to bachbot/scripts/torchrnn/train.zsh."""
    # subprocess.call(' '.join([
    #     BACHBOT_DIR + '/scripts/torchrnn/train.zsh',
    #     input_h5,
    #     checkpoint_dir
    # ]), shell=True)

    vocab = json.load(open(f"{SCRATCH_DIR}/utf_to_txt.json", "r"))
    encoding = { k:i for i,k in enumerate(vocab.keys()) }
    decoding = { v:k for k,v in encoding.items() }

    corpus = open(concat_corpus_txt, "r").read()
    example_tensors = []
    example = []
    for c in corpus:
        example.append(encoding[c])
        if vocab[c] == 'END':
            example_tensors.append(torch.Tensor(example))
            example = []

    seq_lengths = torch.Tensor(list(map(lambda x: x.size(0), example_tensors))).long()
    padded = torch.nn.utils.rnn.pad_sequence(example_tensors).long().cuda()

    n_layers = 3
    wordvec_size = 32
    hidden_size = 256
    dropout = 0.3

    bachbot = BachBot(len(vocab),
                      wordvec_size,
                      n_layers,
                      hidden_size,
                      dropout)
    learning_rate = 2e-3
    optimizer = torch.optim.Adam(
        bachbot.parameters(),
        lr=learning_rate)

    epoch = 0
    try:
        checkpoint = torch.load(checkpoint_path)
        bachbot.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Loaded from checkpoint {checkpoint_path}")
    except:
        print("Initialiizing from scratch")
        epoch = 0
        pass

    batch_size = 50
    n_epoch = 100
    bachbot.train()
    for _ in range(n_epoch):
        for _ in range(1 + int(len(example_tensors) / batch_size)):
            optimizer.zero_grad()

            batch = []
            targets = []
            idxs = [random.randint(0, len(example_tensors)-1) for _ in range(batch_size)]
            batch = padded[:-1, idxs] # batch on 1st axis, RNN convention
            lengths = seq_lengths[idxs] - 1 # -1 because we are staggering inputs and outputs by 1
            targets = padded[1:, idxs].permute([1, 0]) # batch on 0th axis

            scores, _ = bachbot(batch, lengths)

            criterion = nn.CrossEntropyLoss()
            loss = 0
            for i, seq_len in enumerate(lengths):
                loss += criterion(scores[i,:seq_len,:], targets[:seq_len,i])
            print(loss)

            loss.backward()
            optimizer.step()

        epoch += 1
        model_path = f"{SCRATCH_DIR}/bachbot-{epoch}.pt"
        print(f"Checkpointing {model_path}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': bachbot.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True), required=True)
@click.option('-t', '--temperature', type=float, default=0.9)
@click.option('-s', '--start-text-file', type=click.Path(exists=True), help='Primer UTF file')
def sample(checkpoint_path, temperature, start_text_file):
    """Samples torch-rnn model. Calls bachbot/scripts/torchrnn/sample.zsh."""
    # if not start_text_file:
    #     start_text = START_DELIM
    # else:
    #     start_text = codecs.open(start_text_file, 'r', 'utf8').read()[:320]
    # subprocess.call(
    #         BACHBOT_DIR + '/scripts/torchrnn/sample.zsh {0} {1} {2}'.format(checkpoint, temperature, start_text),
    #         shell=True)

    vocab = json.load(open(f"{SCRATCH_DIR}/utf_to_txt.json", "r"))
    encoding = { k:i for i,k in enumerate(vocab.keys()) }
    decoding = { v:k for k,v in encoding.items() }

    output_len = 5000
    min_length = 100
    try:
        n_layers = 3
        wordvec_size = 32
        hidden_size = 256
        dropout = 0.3
        bachbot = BachBot(len(vocab),
                          wordvec_size,
                          n_layers,
                          hidden_size,
                          dropout)
        checkpoint = torch.load(checkpoint_path)
        bachbot.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded from checkpoint {checkpoint_path}")
    except:
        raise Exception("Could not load checkpoint!")

    bachbot.eval()
    hidden = None
    output = [ encoding[START_DELIM] ]
    for _ in range(output_len):
        prev_char = output[-1]
        scores, hidden = bachbot(
            torch.tensor([prev_char])[None,:].cuda(),
            lengths=torch.ones(1).cuda(),
            hidden=hidden)
        if temperature < 1e-9:
            next_char = torch.argmax(scores).item()
        else:
            weights = torch.exp(scores / temperature).detach().cpu().numpy().squeeze()
            weights /= weights.sum()
            next_char = np.random.choice(len(vocab), p=weights)
        output.append(next_char)

    utf_data = list(map(lambda k: decoding[k], output))
    # utf_scores = ''.join(utf_data).split(END_DELIM)[1:]
    utf_scores = [utf_data]

    i = 0
    for utf_score in utf_scores:
        score = decode.decode_utf_single(vocab, utf_score)
        if score and len(score) >= min_length:
            print('Writing {0}'.format(SCRATCH_DIR + '/out-{0}'.format(i)))
            with open(SCRATCH_DIR + '/out-{0}.txt'.format(i), 'w') as fd:
                fd.write('\n'.join(decode.to_text(score)))
            decode.to_musicxml(score).write('musicxml', SCRATCH_DIR + '/out-{0}.xml'.format(i))
            i += 1
