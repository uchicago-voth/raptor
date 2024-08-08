# -*- encoding: utf-8 -*-
'''
Python version   : python 3.10.12
Filename         : protonhopping.py
Description      : calculate the proton forward hopping index from the pivot ids
Time             : 2023/09/08 12:58:52
Author           : Sijia Chen
Version          : 1.0
Email            : sijiachen@uchicago.edu
'''

import numpy as np

def proton_hopping_indexes(pivot_ids):
    """
    Calculate the proton hopping events index from the pivot ids.
    In paper (J. Phys. Chem. B, Vol. 112, No. 2, 2008), equation 15 and 16 define the so-called "forward hopping" accumulation function as follows:

    \\begin{equation}
    \\tag{15}
    h(\delta t) = h(\delta t - 1) + \delta h(\delta t), where h(0) = 0
    \end{equation}

    \\begin{equation}
    \\tag{16}
    \delta h(\delta t) = 0, if no proton hop
    \delta h(\delta t) = 1, if proton hops to a new acceptor
    \delta h(\delta t) = -1, if proton hops back to the previous donor
    \end{equation}

    As complement to the above definition by paper (J. Chem. Phys. 154, 194506 (2021); doi: 10.1063/5.0040758), a more detailed definition of the proton forward hopping index is given as follows: 
    If the proton hops from water A → B → C, the index is changed from 0 → 1 → 2. Conversely, if the proton hopped from water C back to B, the index would be changed back to 1. If the proton subsequently hopped from B to a new water D, the index is updated to 2, since waters C and D are at the same level of displacement from the initial water A.

    
    Parameters
    ----------
    pivot_ids : np.array or list
        pivot ids (numpy array or list, shape: (# of cec, # of frames), dtype: int)

    Returns
    -------
    hopping_indexes : np.array
        the pivot levels (numpy array, shape: (# of cec, # of frames), dtype: int)
    """
    pivot_ids = np.array(pivot_ids,dtype=np.int64)

    # check the shape of the input
    if len(pivot_ids.shape) == 1: # in case the input is a single cec
        pivot_ids = pivot_ids[np.newaxis, :]
    elif len(pivot_ids.shape) == 2: # in case the input is a trajectory of cec
        pass
    else:
        raise ValueError("The input pivot_ids has a wrong shape. The shape of the input should be (# of cec, # of frames).")
    
    # calculate the number of cec and the number of frames
    n_cecs = pivot_ids.shape[0]
    n_frames = pivot_ids.shape[1]

    # calculate the number of proton hopping events
    # the below code will introduce errors when the pivot id changes between A and B for more than one time
    # >>>>>>>>>>>>>>>>> begin
    # pivot_changes = np.insert(np.diff(pivot_ids, axis=1),0,0,axis=1) # 0 means pivot id doesn't change (no proton hopping), non-zero means pivot id changes (proton hopping); insert 0 at the beginning of the array to make the shape of the array the same as pivot_ids
    # events = np.nonzero(pivot_changes) # the indices of the proton hopping events
    # proton_hopping = np.zeros_like(pivot_ids) # the array to store the proton hopping events
    # for i,j in zip(events[0], events[1]):
    #     if j == 0:
    #         # the first hopping event, always hops to a new acceptor
    #         proton_hopping[i,j] = 1
    #     else:
    #         proton_hopping[i,j] = -1 if pivot_changes[i,j-1] + pivot_changes[i,j] == 0 else 1 # if the pivot id changes back to the previous one, then it's a backward hopping event, otherwise it's a forward hopping event
    # <<<<<<<<<<<<<<<<< end

    # a slower method, perhaps better? assign a "level" parameter to each pivot
    hopping_indexes = np.zeros_like(pivot_ids)
    hopping_indexes_dict = [{} for _ in range(n_cecs)]
    for icec in range(n_cecs):
        hopping_indexes_dict[icec][pivot_ids[icec,0]] = 0
        hopping_indexes[icec,0] = 0
        for iframe in range(1,n_frames):
            if pivot_ids[icec,iframe] == pivot_ids[icec,iframe-1]:
                hopping_indexes[icec,iframe] = hopping_indexes[icec,iframe-1]
            else:
                if pivot_ids[icec,iframe] in hopping_indexes_dict[icec]:
                    hopping_indexes[icec,iframe] = hopping_indexes_dict[icec][pivot_ids[icec,iframe]]
                else:
                    hopping_indexes_dict[icec][pivot_ids[icec,iframe]] = hopping_indexes[icec,iframe-1] + 1
                    hopping_indexes[icec,iframe] = hopping_indexes_dict[icec][pivot_ids[icec,iframe]]
    
    return hopping_indexes


def proton_hopping_indexes_with_parser(filename):
    """
    Calculate the proton hopping events index from the raptor output file.

    Parameters
    ----------
    filename : str
        the path of the raptor output file
    """

    from ..parser.raptorreader import RaptorReader
    raptor_reader = RaptorReader(filename)
    pivot_ids = np.array([[] for _ in range(raptor_reader.n_complexes)])
    for frame in raptor_reader:
        frame_pivot_ids = np.array(frame.pivot_ids)
        pivot_ids = np.append(pivot_ids,frame_pivot_ids[:,np.newaxis],axis=1)
    return proton_hopping_indexes(pivot_ids)
        

def proton_identity_correlation(pivot_ids, tau_max, window_step=1,intermittency=0):
    """
    
    Parameters
    ----------
    pivot_ids : np.array or list
        pivot ids (numpy array or list, shape: (# of frames, ), dtype: int)
    tau_max : int
        the maximum value of tau
    window_step : int, optional
        the number of frames between each window, default is 1.
    intermittency : int, optional
        a filter to filter out the rattling effect, default is 0.
    """
    from .correlation import ProtonIdentityCorrelation
    correlation = ProtonIdentityCorrelation(pivot_ids, tau_max, window_step=window_step,intermittency=intermittency)
    taus, timeseries, timeseries_data = correlation.calc_autocorrelation()
    return taus, timeseries, timeseries_data

def proton_continuous_correlation(pivot_ids, tau_max, window_step=1,intermittency=0):
    """
    
    Parameters
    ----------
    pivot_ids : np.array or list
        pivot ids (numpy array or list, shape: (# of frames, ), dtype: int)
    tau_max : int
        the maximum value of tau
    window_step : int, optional
        the number of frames between each window, default is 1.
    intermittency : int, optional
        a filter to filter out the rattling effect, default is 0.

    Returns
    -------
    taus : numpy.ndarray
        the tau values
    timeseries : numpy.ndarray
        the timeseries of the correlation function
    timeseries_data : numpy.ndarray
        the timeseries of the correlation function, with each row corresponding to a window
    """
    from .correlation import ContinuousProtonCorrelation
    correlation = ContinuousProtonCorrelation(pivot_ids, tau_max, window_step=window_step,intermittency=intermittency)
    taus, timeseries, timeseries_data = correlation.calc_autocorrelation()
    return taus, timeseries, timeseries_data

def test():
    pivot_ids = np.array([[1,1,1,2,2,2,3,2,4,4,5,5,5,4,5,5],[1,1,1,2,2,2,1,1,1,1,2,2,3,3,3,3]])
    print(proton_hopping_indexes(pivot_ids))

if __name__ == "__main__":
    test()