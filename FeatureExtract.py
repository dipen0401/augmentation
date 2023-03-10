from six.moves import cPickle
import numpy as np
import scipy.misc
import itertools
import os
import imageio
from math import sqrt
n_bin   = 12        # histogram bins
n_slice = 3         # slice image
h_type  = 'region'  # global or region
d_type  = 'd1'      # distance type

n_slice    = 2
n_orient   = 8
step       = 10
radius     = 30
rings      = 2
histograms = 6
h_type     = 'region'
d_type     = 'd1'
depth      = 3
R = (rings * histograms + 1) * n_orient

stride = (1, 1)
n_slice  = 10
h_type   = 'region'
d_type   = 'cosine'

depth    = 5

edge_kernels = np.array([
  [
   # vertical
   [1,-1], 
   [1,-1]
  ],
  [
   # horizontal
   [1,1], 
   [-1,-1]
  ],
  [
   # 45 diagonal
   [sqrt(2),0], 
   [0,-sqrt(2)]
  ],
  [
   # 135 diagnol
   [0,sqrt(2)], 
   [-sqrt(2),0]
  ],
  [
   # non-directional
   [2,-2], 
   [-2,2]
  ]
])


class Color(object):

  def findFW(self, input, n_bin=n_bin, type=h_type, n_slice=n_slice, normalize=True):
    ''' count img color histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray
        n_bin    : number of bins for each channel
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram
  
      return
        type == 'global'
          a numpy array with size n_bin ** channel
        type == 'region'
          a numpy array with size n_slice * n_slice * (n_bin ** channel)
    '''
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = imageio.imread(input)
    height, width, channel = img.shape
    bins = np.linspace(0, 256, n_bin+1, endpoint=True)  # slice bins equally for each channel
  
    if type == 'global':
      hist = self._count_hist(img, n_bin, bins, channel)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, n_bin ** channel))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._count_hist(img_r, n_bin, bins, channel)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _count_hist(self, input, n_bin, bins, channel):
    print('.')
    
    img = input.copy()
    bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
    hist = np.zeros(n_bin ** channel)
  
    # cluster every pixels
    for idx in range(len(bins)-1):
      img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
    # add pixels into bins
    height, width, _ = img.shape
    for h in range(height):
      for w in range(width):
        b_idx = bins_idx[tuple(img[h,w])]
        hist[b_idx] += 1
  
    return hist
  
  
  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "histogram_cache-{}-n_bin{}".format(h_type, n_bin)
    elif h_type == 'region':
      sample_cache = "histogram_cache-{}-n_bin{}-n_slice{}".format(h_type, n_bin, n_slice)
    
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=h_type, n_bin=n_bin, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
  
    return samples

class Daisy(object):

  def histogram(self, input, type=h_type, n_slice=n_slice, normalize=True):
    ''' count img histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram
  
      return
        type == 'global'
          a numpy array with size R
        type == 'region'
          a numpy array with size n_slice * n_slice * R
  
        #R = (rings * histograms + 1) * n_orient#
    '''
    print('.')
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = imageio.imread(input)
    height, width, channel = img.shape
  
    P = math.ceil((height - radius*2) / step) 
    Q = math.ceil((width - radius*2) / step)
    assert P > 0 and Q > 0, "input image size need to pass this check"
  
    if type == 'global':
      hist = self._daisy(img)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, R))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._daisy(img_r)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _daisy(self, img, normalize=True):
    from skimage import color
    from skimage.feature import daisy
    image = color.rgb2gray(img)
    descs = daisy(image, step=step, radius=radius, rings=rings, histograms=histograms, orientations=n_orient)
    descs = descs.reshape(-1, R)  # shape=(N, R)
    hist  = np.mean(descs, axis=0)  # shape=(R,)
  
    if normalize:
      hist = np.array(hist) / np.sum(hist)
  
    return hist
  
  
  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "daisy-{}-n_orient{}-step{}-radius{}-rings{}-histograms{}".format(h_type, n_orient, step, radius, rings, histograms)
    elif h_type == 'region':
      sample_cache = "daisy-{}-n_slice{}-n_orient{}-step{}-radius{}-rings{}-histograms{}".format(h_type, n_slice, n_orient, step, radius, rings, histograms)
  
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
      for sample in samples:
        sample['hist'] /= np.sum(sample['hist'])  # normalize
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
  
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=h_type, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
  
    return samples

class Edge(object):

  def histogram(self, input, stride=(2, 2), type=h_type, n_slice=n_slice, normalize=True):
    ''' count img histogram
  
      arguments
        input    : a path to a image or a numpy.ndarray
        stride   : stride of edge kernel
        type     : 'global' means count the histogram for whole image
                   'region' means count the histogram for regions in images, then concatanate all of them
        n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
        normalize: normalize output histogram
  
      return
        type == 'global'
          a numpy array with size len(edge_kernels)
        type == 'region'
          a numpy array with size len(edge_kernels) * n_slice * n_slice
    '''
    print('.')
    if isinstance(input, np.ndarray):  # examinate input type
      img = input.copy()
    else:
      img = imageio.imread(input)
    height, width, channel = img.shape
  
    if type == 'global':
      hist = self._conv(img, stride=stride, kernels=edge_kernels)
  
    elif type == 'region':
      hist = np.zeros((n_slice, n_slice, edge_kernels.shape[0]))
      h_silce = np.around(np.linspace(0, height, n_slice+1, endpoint=True)).astype(int)
      w_slice = np.around(np.linspace(0, width, n_slice+1, endpoint=True)).astype(int)
  
      for hs in range(len(h_silce)-1):
        for ws in range(len(w_slice)-1):
          img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
          hist[hs][ws] = self._conv(img_r, stride=stride, kernels=edge_kernels)
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist.flatten()
  
  
  def _conv(self, img, stride, kernels, normalize=True):
    H, W, C = img.shape
    conv_kernels = np.expand_dims(kernels, axis=3)
    conv_kernels = np.tile(conv_kernels, (1, 1, 1, C))
    assert list(conv_kernels.shape) == list(kernels.shape) + [C]  # check kernels size
  
    sh, sw = stride
    kn, kh, kw, kc = conv_kernels.shape
  
    hh = int((H - kh) / sh + 1)
    ww = int((W - kw) / sw + 1)
  
    hist = np.zeros(kn)
  
    for idx, k in enumerate(conv_kernels):
      for h in range(hh):
        hs = int(h*sh)
        he = int(h*sh + kh)
        for w in range(ww):
          ws = w*sw
          we = w*sw + kw
          hist[idx] += np.sum(img[hs:he, ws:we] * k)  # element-wise product
  
    if normalize:
      hist /= np.sum(hist)
  
    return hist
  
  
  def make_samples(self, db, verbose=True):
    if h_type == 'global':
      sample_cache = "edge-{}-stride{}".format(h_type, stride)
    elif h_type == 'region':
      sample_cache = "edge-{}-stride{}-n_slice{}".format(h_type, stride, n_slice)
  
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
      for sample in samples:
        sample['hist'] /= np.sum(sample['hist'])  # normalize
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
  
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        d_hist = self.histogram(d_img, type=h_type, n_slice=n_slice)
        samples.append({
                        'img':  d_img, 
                        'cls':  d_cls, 
                        'hist': d_hist
                      })
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
  
    return samples


def findFourierWavelet(img) :
    color = Color()
    hist = color.findFW(img, type='global')
    
    return hist

def findDCTConv(img) :
    from skimage import color
    import cv2
    img = cv2.imread(img)
    
    daisy = Daisy()
    
    image = color.rgb2gray(img)
    descs = daisy._daisy(image)
    descs = descs.reshape(-1, R)  # shape=(N, R)
    hist  = np.mean(descs, axis=0)  # shape=(R,)
  
    hist = np.array(hist) / np.sum(hist)
  
    return hist

def findMFCC(img) :
    import cv2
    from skimage import color
    
    img = cv2.imread(img)
    
    edge = Edge()
    
    image = img
    descs = edge._conv(image, stride, edge_kernels)

    #hist  = np.mean(descs, axis=0)  # shape=(R,)
    #hist = np.array(hist) / np.sum(hist)
  
    return descs
    