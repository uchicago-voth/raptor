# -*- encoding: utf-8 -*-
'''
Python version   : python 3.10.12
Filename         : cecmsd.py
Description      : Calculate the mean square displacement of the center of excess charge (cec), as defined in `J. Phys. Chem. B 2006, 110, 37, 18594–18600`
Time             : 2023/09/04 11:52:23
Author           : Sijia Chen
Version          : 1.0
Email            : sijiachen@uchicago.edu
'''

"""
dependent packages:
1. numpy
2. tidynamics

current thoughts:
1. read the cec information (position, pivot id) from raptor parser output. The input is a trajectory of cec (numpy array, shape: (# of cec, # of frames, 3), dtype: np.float64) and the pivot id (numpy array, shape: (# of cec, # of frames), dtype: int)
2. use np.diff to calculate the displacement between two neighboring frames
3. use np.mask to mask out the change of pivot id
4. copy the array in step 2 to two numpy arrays to store the two different kinds of displacements: discrete and continuous (l_d, and l_c).
5. For discrete displacement, substitute the values in l_d where the change of pivot id doesn't happen with 0.
6. For continuous displacement, substitute the values in l_c where the change of pivot id happens with 0.
7. use np.cumsum to calculate the cumulative sum of the two arrays.
8. add zeros to the beginning of the two arrays to make the length of the two arrays equal to the length of the trajectory.
9. calculate the mean square displacement of the two arrays using tidynamics.msd
"""

# import packages
import numpy as np
import tidynamics

# define functions
def calc_cec_msd(positions: np.array, pivot_ids: np.array):
    """
    Calculate the mean square displacement of the center of excess charge (cec), as defined in the paper (J. Phys. Chem. B 2006, 110, 37, 18594–18600, https://doi.org/10.1021/jp062719k). 

    \math{r_{CEC_i}(t) - r_{CEC_i}(0) = \sum_{j=1}^{t} \Delta r_{CEC_i}(j) = \sum_{j=1}^{t} (\Delta r_{CEC_i}^{d}(j) + \Delta r_{CEC_i}^{c}(j))}

    Make sure that the time interval between two neighbouring frames is constant, otherwise the result doesn't make sense. 
    
    Parameters
    ----------
    positions : np.array
        trajectory of cec (numpy array, shape: (# of cec, # of frames, 3), dtype: np.float64)
    pivot_ids : np.array
        pivot id (numpy array, shape: (# of cec, # of frames), dtype: int)

    Returns
    -------
    msd_d : np.array, shape: (# of cec, # of frames, 3 + 1)
        mean square displacement of the discrete displacement, where the third dimension stores the displancement in the x, y, and z directions, as well as the total displacement
    msd_c : np.array, shape: (# of cec, # of frames, 3 + 1)
        mean square displacement of the continuous displacement
    msd : np.array, shape: (# of cec, # of frames, 3 + 1)
        mean square displacement of the total displacement
    """
    # check the shape of the input
    if len(positions.shape) == 2 and len(pivot_ids) == 1: # in case the input is a single cec
        positions = positions[np.newaxis, :, :]
        pivot_ids = pivot_ids[np.newaxis, :]
    assert positions.shape[0] == pivot_ids.shape[0], "The number of cec in the trajectory and the number of cec in the pivot id should be equal."
    assert positions.shape[1] == pivot_ids.shape[1], "The number of frames in the trajectory and the number of frames in the pivot id should be equal."
    num_cecs = positions.shape[0]
    num_frames = positions.shape[1]
    num_dimensions = positions.shape[2]

    # calculate the displacement
    delta_positions = np.diff(positions, axis=1)
    delta_pivot_ids = np.diff(pivot_ids, axis=1).astype(bool) # if pivot id changed, then True, else False

    # assign the displacement to the two different kinds of displacements
    delta_positions_d = np.where(delta_pivot_ids[:,:,np.newaxis], np.zeros_like(delta_positions),delta_positions) 
    delta_positions_c = np.where(np.invert(delta_pivot_ids)[:,:,np.newaxis], np.zeros_like(delta_positions), delta_positions)

    # calculate the cumulative sum of the two arrays, and add zeros to the beginning of the two arrays to make the length of the two arrays equal to the length of the trajectory (the number of frames).
    positions_d = np.insert(np.cumsum(delta_positions_d, axis=1), 0, 0.0, axis=1)
    positions_c = np.insert(np.cumsum(delta_positions_c, axis=1), 0, 0.0, axis=1)

    # calculate the mean square displacement of the three arrays
    msd_d = np.zeros((num_cecs, num_frames, num_dimensions + 1))
    msd_c = np.zeros((num_cecs, num_frames, num_dimensions + 1))
    msd = np.zeros((num_cecs, num_frames, num_dimensions + 1))
    for i in range(num_cecs):
        for k in range(num_dimensions):
            msd_d[i, :, k] = tidynamics.msd(positions_d[i, :, k])
            msd_c[i, :, k] = tidynamics.msd(positions_c[i, :, k])
            msd[i, :, k] = tidynamics.msd(positions[i, :, k])
        msd_d[i, :, -1] = np.sum(msd_d[i, :, :-1], axis=1)
        msd_c[i, :, -1] = np.sum(msd_c[i, :, :-1], axis=1)
        msd[i, :, -1] = np.sum(msd[i, :, :-1], axis=1)
    return msd_d, msd_c, msd
    

if __name__ == "main":
    pass