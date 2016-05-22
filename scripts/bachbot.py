import click

from kerntools.scrape_humdrum import scrape_humdrum
from kerntools.concatenate_corpus import concatenate_corpus
from kerntools.renumber_measures import renumber_measures

@click.group()
def cli():
    pass

@click.command()
def extract_melody():
    """TODO: finish this"""
    click.echo("hi")

cli.add_command(scrape_humdrum)
cli.add_command(concatenate_corpus)
cli.add_command(extract_melody)
cli.add_command(renumber_measures)
