#!/usr/bin/env python

from lxml import html
import urllib
import urlparse
import requests
import re
import copy
import shutil
import os

# Specify crawler params here
KEYWORD="Bach+Johann"
URL='http://kern.humdrum.org/search?s=t&keyword={0}'.format(KEYWORD)

# Create corpus directory
directory = './corpus/{0}/'.format(KEYWORD)
if not os.path.exists(directory):
    os.makedirs(directory)

# Query with KEYWORD and crawl for info pages
tree = html.fromstring(requests.get(URL).content)
all_urls = tree.xpath('//a/@href')
info_page_urls = filter(
        lambda url: re.search("file=.*format=info", url),
        all_urls)

# Download kern file from each info page
for url in info_page_urls:
    parsed_url = urlparse.urlparse(url)
    parsed_query = urlparse.parse_qs(parsed_url.query)

    filename = parsed_query['file'][0]
    new_query = parsed_url.query[:-12] + "&f=kern"

    kern_url = urlparse.urlunparse(parsed_url[:4] + (new_query,) + parsed_url[5:])
    print kern_url

    response = requests.get(kern_url, stream=True)
    with open(directory + filename, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

