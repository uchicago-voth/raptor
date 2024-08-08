# -*- encoding: utf-8 -*-
'''
Python version   : python 3.10.12
Filename         : evbampdist.py
Description      : 
Time             : 2023/09/08 15:01:52
Author           : Sijia Chen
Version          : 1.0
Email            : sijiachen@uchicago.edu
'''

import numpy as np
import copy

def _find_finest_resolution(bin_size, n_bins, max_amp, min_amp):
    if bin_size == None and n_bins == None:
        # use the default bin_size
        bin_size = 0.02
        n_bins = round((max_amp - min_amp) / bin_size + 0.5) # round up to the integer
        return n_bins
    elif bin_size != None and n_bins == None:
        return round((max_amp - min_amp) / bin_size + 0.5) # round up to the integer
    elif bin_size == None and n_bins != None:
        return n_bins
    else:
        if bin_size > (max_amp - min_amp) / n_bins:
            return n_bins
        else:
            return round((max_amp - min_amp) / bin_size + 0.499999) # round up to the integer



def evb_amplitude_distribution(_eigen_vectors, bin_size=None, n_bins=None, max_amp=1.0, min_amp=0.0, density=True):
    """
    Calculate the distribution of the first two largest EVB amplitude
    
    Parameters
    ----------
    _eigen_vectors : list
        eigen vectors ((# of cec, # of frames, # of states in one frame), dtype: float). Due to the fact that the length of each eigen vector of a single frame maybe more than 2, this function only considers the first two largest EVB amplitude and all other EVB amplitudes are ignored. The last dimension of the input eigen vectors should be equal to or larger than 2 and may be different for different frames. It is also acceptable that the input eigen vectors only have two dimensions (shape of (# of frames, # of states in a frame)), and the code will assume these is only one cec. Otherwise, the code will raise an error.
    bin_size : float, optional
        the size of the bin, by default is None
    n_bins : int, optional
        the number of bins, by default is None
    max_amp : float, optional
        the maximum value of the EVB amplitude, by default 1.0
    min_amp : float, optional
        the minimum value of the EVB amplitude, by default 0.0
    density : bool, optional
        if True, the result is the probability density, otherwise the result is the number of samples in each bin, by default True

    Returns
    -------
    amp_dist : np.array
        the distribution of the first two largest EVB amplitude ((# of cec, 2, n_bins), dtype: float)
    bin_edges : np.array
        the bin edges ((n_bins+1), dtype: float)

    Notes
    -----
    if both bin_size and n_bins are None, the default value of n_bins will be the smallest number of bins which gives bin_size no larger than 0.02; if both bin_size and n_bins are given, the code will choose which one has a finer resolution and use that one to calculate the distribution.
    """
    _eigen_vectors = copy.deepcopy(_eigen_vectors)
    _eigen_vectors = np.array(_eigen_vectors,dtype=np.object_)

    # check the shape of the input
    if len(_eigen_vectors.shape) == 1: # in case the input is a single cec and different frames have different number of states
        if isinstance(_eigen_vectors[0][0], float):
            _eigen_vectors = _eigen_vectors[np.newaxis, np.newaxis :]
        else:
            raise ValueError("The input eigen_vectors has a wrong shape. The shape of the input should be (# of cec, # of frames, (*) # of states in one frame) or (# of frames, (*) # of states in one frame), now is {}.".format(_eigen_vectors.shape))
    elif len(_eigen_vectors.shape) == 2: 
        if isinstance(_eigen_vectors[0,0], float):# in case the input is a single cec and all frames have the same number of states
            _eigen_vectors = _eigen_vectors[np.newaxis, :]
        elif isinstance(_eigen_vectors[0,0], list): # in case the input is multiple cecs and but frames have different number of states
            pass
        else:
            raise ValueError("The input eigen_vectors has a wrong shape. The shape of the input should be (# of cec, # of frames, (*) # of states in one frame) or (# of frames, (*) # of states in one frame), now is {}.".format(_eigen_vectors.shape))
    elif len(_eigen_vectors.shape) == 3: # in case the input is a multiple cecs and all frames have the same number of states
        pass
    else:
        raise ValueError("The input eigen_vectors has a wrong shape. The shape of the input should be (# of cec, # of frames, (*) # of states in one frame) or (# of frames, (*) # of states in one frame), now is {}.".format(_eigen_vectors.shape))
    
    # grab only the first two largest EVB amplitude
    # note that the first two eigen vector values are not necessarily the first two largest EVB amplitude, sorting is needed
    # first, deal with the states with only one eigen vector value
    for cec in _eigen_vectors:
        for frame in cec:
            if len(frame) == 1:
                frame.append(0.0)
    eigen_vectors = np.array([[sorted(frame,reverse=True)[:2] for frame in cec] for cec in _eigen_vectors], dtype=np.float64)
    amp_vectors = np.power(eigen_vectors, 2)

    # calculate the number of cec and the number of frames
    n_cecs = amp_vectors.shape[0]

    # find the finest resolution
    nbins = _find_finest_resolution(bin_size, n_bins, max_amp, min_amp)

    amp_dists = np.zeros((n_cecs, 2, nbins), dtype=np.float64)
    # calculate the distribution of the first two largest EVB amplitude
    for icec in range(n_cecs):
        amp_dists[icec,0], bin_edges = np.histogram(amp_vectors[icec,:,0], bins=nbins, range=(min_amp, max_amp),density=density)
        # amp_dists[icec,0] = amp_dist
        amp_dists[icec,1], bin_edges = np.histogram(amp_vectors[icec,:,1], bins=nbins, range=(min_amp, max_amp),density=density)
        # amp_dists[icec,1] = amp_dist


    return amp_dists, bin_edges

def evb_amplitude_distribution_with_parser(filename, bin_size=None, n_bins=None, max_amp=1.0, min_amp=0.0, density=True):
    """
    Calculate the distribution of the first two largest EVB amplitude from the raptor output file.

    Parameters
    ----------
    filename : str
        the path of the raptor output file
    bin_size : float, optional
        the size of the bin, by default is None
    n_bins : int, optional
        the number of bins, by default is None
    max_amp : float, optional
        the maximum value of the EVB amplitude, by default 1.0
    min_amp : float, optional
        the minimum value of the EVB amplitude, by default 0.0
    density : bool, optional
        if True, the result is the probability density, otherwise the result is the number of samples in each bin, by default True

    Returns
    -------
    amp_dist : np.array
        the distribution of the first two largest EVB amplitude ((# of cec, 2, n_bins), dtype: float)
    bin_edges : np.array
        the bin edges ((n_bins+1), dtype: float)
    """

    from ..parser.raptorreader import RaptorReader
    from ..parser.raptorframe import RaptorFrame
    raptor_reader = RaptorReader(filename)
    eigen_vectors = [[] for _ in range(raptor_reader.n_complexes)]
    for frame in raptor_reader:
        frame_eigen_vectors = frame.eigen_vectors
        for icec in range(raptor_reader.n_complexes):
            eigen_vectors[icec].append(frame_eigen_vectors[icec][:2])
    return evb_amplitude_distribution(eigen_vectors, bin_size, n_bins, max_amp, min_amp, density)


def evb_amp_difference(_eigen_vectors, bin_size=None, n_bins=None, max_amp=1.0, min_amp=0.0, density=True):
    """
    Calculate the distribution of the difference between first two largest EVB amplitude's square
    
    Parameters
    ----------
    _eigen_vectors : list
        eigen vectors ((# of cec, # of frames, # of states in one frame), dtype: float). Due to the fact that the length of each eigen vector of a single frame maybe more than 2, this function only considers the first two largest EVB amplitude and all other EVB amplitudes are ignored. The last dimension of the input eigen vectors should be equal to or larger than 2 and may be different for different frames. It is also acceptable that the input eigen vectors only have two dimensions (shape of (# of frames, # of states in a frame)), and the code will assume these is only one cec. Otherwise, the code will raise an error.
    bin_size : float, optional
        the size of the bin, by default is None
    n_bins : int, optional
        the number of bins, by default is None
    max_amp : float, optional
        the maximum value of the EVB amplitude, by default 1.0
    min_amp : float, optional
        the minimum value of the EVB amplitude, by default 0.0
    density : bool, optional
        if True, the result is the probability density, otherwise the result is the number of samples in each bin, by default True

    Returns
    -------
    amp_dist : np.array
        the distribution of the difference ((# of cec, n_bins), dtype: float)
    bin_edges : np.array
        the bin edges ((n_bins+1), dtype: float)

    Notes
    -----
    if both bin_size and n_bins are None, the default value of n_bins will be the smallest number of bins which gives bin_size no larger than 0.02; if both bin_size and n_bins are given, the code will choose which one has a finer resolution and use that one to calculate the distribution.
    """

    _eigen_vectors = np.array(_eigen_vectors,dtype=np.object_)

    # check the shape of the input
    if len(_eigen_vectors.shape) == 1: # in case the input is a single cec and different frames have different number of states
        if isinstance(_eigen_vectors[0][0], float):
            _eigen_vectors = _eigen_vectors[np.newaxis, np.newaxis :]
        else:
            raise ValueError("The input eigen_vectors has a wrong shape. The shape of the input should be (# of cec, # of frames, (*) # of states in one frame) or (# of frames, (*) # of states in one frame), now is {}.".format(_eigen_vectors.shape))
    elif len(_eigen_vectors.shape) == 2: 
        if isinstance(_eigen_vectors[0,0], float):# in case the input is a single cec and all frames have the same number of states
            _eigen_vectors = _eigen_vectors[np.newaxis, :]
        elif isinstance(_eigen_vectors[0,0], list): # in case the input is multiple cecs and but frames have different number of states
            pass
        else:
            raise ValueError("The input eigen_vectors has a wrong shape. The shape of the input should be (# of cec, # of frames, (*) # of states in one frame) or (# of frames, (*) # of states in one frame), now is {}.".format(_eigen_vectors.shape))
    elif len(_eigen_vectors.shape) == 3: # in case the input is a multiple cecs and all frames have the same number of states
        pass
    else:
        raise ValueError("The input eigen_vectors has a wrong shape. The shape of the input should be (# of cec, # of frames, (*) # of states in one frame) or (# of frames, (*) # of states in one frame), now is {}.".format(_eigen_vectors.shape))
    
    # grab only the first two largest EVB amplitude
    # note that the first two eigen vector values are not necessarily the first two largest EVB amplitude, sorting is needed
    eigen_vectors = np.array([[sorted(frame,reverse=True)[:2] for frame in cec] for cec in _eigen_vectors], dtype=np.float64)
    amp_vectors = np.power(eigen_vectors, 2)
    amp_vectors = amp_vectors[:,:,0] - amp_vectors[:,:,1]

    # calculate the number of cec and the number of frames
    n_cecs = amp_vectors.shape[0]

    # find the finest resolution
    nbins = _find_finest_resolution(bin_size, n_bins, max_amp, min_amp)

    amp_dists = np.zeros((n_cecs, nbins), dtype=np.float64)
    # calculate the distribution of the first two largest EVB amplitude
    for icec in range(n_cecs):
        amp_dists[icec], bin_edges = np.histogram(amp_vectors[icec,:], bins=nbins, range=(min_amp, max_amp),density=density)


    return amp_dists, bin_edges

def test():
    print("test 1")
    c=[[0.9,0.4,0.3],[0.88,0.3,0.2],[0.89,0.4,0.25],[0.95,0.47,0.333],[0.87,0.45,0.1]]
    dist3,edges3=evb_amplitude_distribution(c)
    print(dist3)
    print("test 2")
    a=[[0.9,0.4,0.3],[0.88,0.3,0.2],[0.89,0.4,0.25],[0.95,0.47,0.333,0.1],[0.87,0.45]]
    dist1,edges1=evb_amplitude_distribution(a)
    print(dist1)
    print("test 3")
    b=[a]
    dist2,edges2=evb_amplitude_distribution(b)
    print(dist2)

if __name__ == "__main__":
    test()