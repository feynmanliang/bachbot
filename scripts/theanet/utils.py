import climate
import pickle
import gzip
import numpy as np
import os
import pickle
import sys
import tarfile
import tempfile
import urllib

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.critical('please install matplotlib to run the examples!')
    raise

logging = climate.get_logger(__name__)

climate.enable_default_logging()

DATASETS = os.path.join(tempfile.gettempdir(), 'theanets-datasets')


def find(dataset, url):
    '''Find the location of a dataset on disk, downloading if needed.'''
    fn = os.path.join(DATASETS, dataset)
    dn = os.path.dirname(fn)
    if not os.path.exists(dn):
        logging.info('creating dataset directory: %s', dn)
        os.makedirs(dn)
    if not os.path.exists(fn):
        if sys.version_info < (3, ):
            urllib.urlretrieve(url, fn)
        else:
            urllib.request.urlretrieve(url, fn)
    return fn


def load_mnist(flatten=True, labels=False):
    '''Load the MNIST digits dataset.'''
    fn = find('mnist.pkl.gz', 'http://deeplearning.net/data/mnist/mnist.pkl.gz')
    h = gzip.open(fn, 'rb')
    if sys.version_info < (3, ):
        (timg, tlab), (vimg, vlab), (simg, slab) = pickle.load(h)
    else:
        (timg, tlab), (vimg, vlab), (simg, slab) = pickle.load(h, encoding='bytes')
    h.close()
    if not flatten:
        timg = timg.reshape((-1, 28, 28, 1))
        vimg = vimg.reshape((-1, 28, 28, 1))
        simg = simg.reshape((-1, 28, 28, 1))
    if labels:
        return ((timg, tlab.astype('i')),
                (vimg, vlab.astype('i')),
                (simg, slab.astype('i')))
    return (timg, ), (vimg, ), (simg, )


def load_cifar(flatten=True, labels=False):
    '''Load the CIFAR10 image dataset.'''
    def extract(name):
        logging.info('extracting data from %s', name)
        h = tar.extractfile(name)
        if sys.version_info < (3, ):
            d = pickle.load(h)
        else:
            d = pickle.load(h, encoding='bytes')
            for k in list(d):
                d[k.decode('utf8')] = d[k]
        h.close()
        img = d['data'].reshape(
            (-1, 3, 32, 32)).transpose((0, 2, 3, 1)).astype('f') / 128 - 1
        if flatten:
            img = img.reshape((-1, 32 * 32 * 3))
        d['data'] = img
        return d

    fn = find('cifar10.tar.gz', 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    tar = tarfile.open(fn)

    imgs = []
    labs = []
    for i in range(1, 6):
        d = extract('cifar-10-batches-py/data_batch_{}'.format(i))
        imgs.extend(d['data'])
        labs.extend(d['labels'])
    timg = np.asarray(imgs[:40000])
    tlab = np.asarray(labs[:40000], 'i')
    vimg = np.asarray(imgs[40000:])
    vlab = np.asarray(labs[40000:], 'i')

    d = extract('cifar-10-batches-py/test_batch')
    simg = d['data']
    slab = d['labels']

    tar.close()

    if labels:
        return (timg, tlab), (vimg, vlab), (simg, slab)
    return (timg, ), (vimg, ), (simg, )


def plot_images(imgs, loc, title=None, channels=1):
    '''Plot an array of images.

    We assume that we are given a matrix of data whose shape is (n*n, s*s*c) --
    that is, there are n^2 images along the first axis of the array, and each
    image is c squares measuring s pixels on a side. Each row of the input will
    be plotted as a sub-region within a single image array containing an n x n
    grid of images.
    '''
    n = int(np.sqrt(len(imgs)))
    assert n * n == len(imgs), 'images array must contain a square number of rows!'
    s = int(np.sqrt(len(imgs[0]) / channels))
    assert s * s == len(imgs[0]) / channels, 'images must be square!'

    img = np.zeros(((s+1) * n - 1, (s+1) * n - 1, channels), dtype=imgs[0].dtype)
    for i, pix in enumerate(imgs):
        r, c = divmod(i, n)
        img[r * (s+1):(r+1) * (s+1) - 1,
            c * (s+1):(c+1) * (s+1) - 1] = pix.reshape((s, s, channels))

    img -= img.min()
    img /= img.max()

    ax = plt.gcf().add_subplot(loc)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.imshow(img.squeeze(), cmap=plt.cm.gray)
    if title:
        ax.set_title(title)


def plot_layers(weights, tied_weights=False, channels=1):
    '''Create a plot of weights, visualized as "bottom-level" pixel arrays.'''
    if hasattr(weights[0], 'get_value'):
        weights = [w.get_value() for w in weights]
    k = min(len(weights), 9)
    imgs = np.eye(weights[0].shape[0])
    for i, weight in enumerate(weights[:-1]):
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs,
                    100 + 10 * k + i + 1,
                    channels=channels,
                    title='Layer {}'.format(i+1))
    weight = weights[-1]
    n = weight.shape[1] / channels
    if int(np.sqrt(n)) ** 2 != n:
        return
    if tied_weights:
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs,
                    100 + 10 * k + k,
                    channels=channels,
                    title='Layer {}'.format(k))
    else:
        plot_images(weight,
                    100 + 10 * k + k,
                    channels=channels,
                    title='Decoding weights')


def plot_filters(filters):
    '''Create a plot of conv filters, visualized as pixel arrays.'''
    imgs = filters.get_value()

    N, channels, x, y = imgs.shape
    n = int(np.sqrt(N))
    assert n * n == N, 'filters must contain a square number of rows!'
    assert channels == 1 or channels == 3, 'can only plot grayscale or rgb filters!'

    img = np.zeros(((y+1) * n - 1, (x+1) * n - 1, channels), dtype=imgs[0].dtype)
    for i, pix in enumerate(imgs):
        r, c = divmod(i, n)
        img[r * (y+1):(r+1) * (y+1) - 1,
            c * (x+1):(c+1) * (x+1) - 1] = pix.transpose((1, 2, 0))

    img -= img.min()
    img /= img.max()

    ax = plt.gcf().add_subplot(111)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.imshow(img.squeeze(), cmap=plt.cm.gray)
