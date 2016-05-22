import click
import copy
from lxml import html
import os
import re
import requests
import shutil
import urllib
import urlparse

from constants import BACHBOT_DIR

@click.command()
@click.argument('query', default='Bach+Johann')
@click.option('--out-dir', help='Directory to save results under.')
def scrape_humdrum(query, out_dir):
    """Scrapes kern.humdrum.org to a local directory."""
    if not out_dir:
        out_dir = BACHBOT_DIR + '/corpus/' + query

    click.echo('Saving scrape results to {0}'.format(out_dir))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    search_url='http://kern.humdrum.org/search?s=t&keyword={0}'.format(query)
    click.echo('Requesting {0}'.format(search_url))
    tree = html.fromstring(requests.get(search_url).content)
    all_urls = tree.xpath('//a/@href')
    info_page_urls = filter(
            lambda url: re.search("file=.*format=info", url),
            all_urls)

    for url in info_page_urls:
        parsed_url = urlparse.urlparse(url)
        parsed_query = urlparse.parse_qs(parsed_url.query)

        filename = parsed_query['file'][0]
        new_query = parsed_url.query[:-12] + "&f=kern"

        kern_url = urlparse.urlunparse(parsed_url[:4] + (new_query,) + parsed_url[5:])
        click.echo('Saving {0}'.format(kern_url))
        response = requests.get(kern_url, stream=True)
        with open(out_dir + '/' + filename, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

