import click

from kerntools.scrape_humdrum import scrape_humdrum
from kerntools.concatenate_corpus import concatenate_corpus

@click.group()
def cli():
    pass

@click.command()
def extract_melody():
    click.echo("hi")

cli.add_command(scrape_humdrum)
cli.add_command(concatenate_corpus)
cli.add_command(extract_melody)
