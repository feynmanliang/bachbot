import click
import glob
import os.path
import shutil
import subprocess

@click.command()
@click.argument('checkpoint-dir', type=click.Path(exists=True), default=os.path.expanduser('~/torch-rnn/checkpoints'))
def clean_checkpoints(checkpoint_dir):
    if click.confirm('This will remove everything in {0}/*, ok?'.format(checkpoint_dir)):
        shutil.rmtree(checkpoint_dir)
        os.mkdir(checkpoint_dir)
        print('Cleaned!')
    else:
        print('Aborted!')
