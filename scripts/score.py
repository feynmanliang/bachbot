import click

@click.group()
def score():
    """Interface for scoring MusicXML outputs."""
    pass

@click.command()
@click.option('-i', '--input-path', required=True, type=click.File('rb'))
def pca_metric(input_path):
    """Computes distance to Bach centroid in 2D PCA."""

map(score.add_command, [
    pca_metric
])
