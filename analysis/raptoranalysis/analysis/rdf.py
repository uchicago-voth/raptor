# -*- encoding: utf-8 -*-
'''
Python version   : python 3.10.12
Filename         : rdf.py
Description      : 
Time             : 2023/09/13 14:03:23
Author           : Sijia Chen
Version          : 1.0
Email            : sijiachen@uchicago.edu
'''

"""
TODO:
1. add the option to calculate the rdf between the same group of atoms
2. multiprocessing (shouldn't be too hard, as we now have the `offset` option in `RaptorReader` and `LammpsReader`)
"""

import numpy as np
from ..parser.raptorreader import RaptorReader
from ..parser.lammpsreader import LammpsReader
import multiprocessing as mp 

def minimum_image_distance(coords1, coords2, boundary: np.array):
    """
    Calculate the minimum image distance between two sets of coordinates
    
    Parameters
    ----------
    coords1 : np.array
        coordinates of the first set of atoms ((# of atoms 1, # of dimensions), dtype: float)
    coords2 : np.array
        coordinates of the second set of atoms ((# of atoms 2, # of dimensions), dtype: float)
    boundary : np.array
        the boundary of the simulation box ((# of dimensions, ) or (# pf dimensions, 2)), dtype: float)
            
    
    Returns
    -------
    distance : np.array
        the minimum image distance between two sets of atoms ((# of atoms 1, # of atoms 2), dtype: float)
    disance_dim : np.array
        the minimum image distance between two sets of atoms in each dimension ((# of dimensions, # of atoms 1, # of atoms 2), dtype: float)
    """
    n_dims = len(boundary)
    if len(boundary.shape) == 2:
        boundary = np.abs(boundary[:,1] - boundary[:,0]).reshape(n_dims)
    
    if coords1.shape[1] != n_dims or coords2.shape[1] != n_dims:
        raise ValueError("The number of dimensions of the coordinates is not equal to the number of dimensions of the boundary")
    
    if coords1.shape[0] == 0 or coords2.shape[0] == 0:
        raise ValueError("The number of atoms in one of the coordinates is zero")
    
    disance_dim = []
    
    for i_dim in range(n_dims):
        dis = np.abs(coords1[:,i_dim].reshape(-1,1) - coords2[:,i_dim].reshape(1,-1)) # results in (# of atoms 1, # of atoms 2)
        dis = dis % boundary[i_dim] # wrap the distance to the boundary
        dis = np.where(dis > boundary[i_dim]/2, dis - boundary[i_dim], dis) # minimum image distance
        disance_dim.append(dis)

    distance = np.linalg.norm(np.array(disance_dim), axis=0) # results in (# of atoms 1, # of atoms 2)
    return distance, disance_dim


class RaptorRdf(object):
    """
    Calculate the radial distribution function (RDF) of a system involving cecs.
    Raptor output file and Lammps trajectory file are required.

    Make sure the raptor output frequency is the same as the lammps trajectory output frequency.  This class doesn't check this.
    To make sure this, the setting `output frequency, 0 means obey lammps setting` in the file `in.evb` should be `0` (obey lammps setting) or a positive integer equal to lammps dump.
    
    If you turn on `output the reaction` in `in.evb`, you need to give the output frequency `output_freq` explicitly in this class to filter out the reaction frames.

    Parameters
    ----------
    raptor_filename : str
        the name of the raptor output file
    traj_filename : str
        the name of the lammps trajectory file
    group1 : list of int
        a list of the indexes of the first group of atoms in the raptor output file (the indexes start from 0 and less than the number of cecs in the system).
        if None, all cecs in the system will be used;
        if the system has 10 cecs, then the indexes should be in the range of 0 (inclusive) to 9 (inclusive).
    group2 : list of int
        the indexes of the second group of atoms in the lammps trajectory file (the indexes start from 0 and less than the number of atoms in the system).
        if None, all atoms in the system will be used; (which doesn't make much sense)
        if the system has 1000 atoms, then the indexes should be in the range of 0 (inclusive) to 999 (inclusive).
    group2_types : list of int
        the types of the second group of atoms in the lammps trajectory file (the types start from 1 and no larger than the number of atom types in the system).
        if None, all atoms in the system will be used; (which doesn't make much sense)
        if the system has 3 atom types, then the types should be in the range of 1 (inclusive) to 3 (inclusive).
        You cannot use `group2` and `group2_types` at the same time.
    n_bins : int, optional
        the number of bins, by default 100
    max_r : float, optional
        the maximum value of r, by default 10.0
    min_r : float, optional
        the minimum value of r, by default 0.0
    density : bool, optional
        whether to normalize the rdf to density, by default True
    output_freq : int, optional
        the output frequency of the raptor/lammps output file, by default None
    tell_time : int, optional
        whether to print the time when already calculating another `tell_time` frames, by default is 0 (no printing)
    
    Methods
    ------
    calculate: 
        start the calculation. results can be accessed by the properties `rdf`, `bin_centers`, `bin_edges`.

    Properties
    ----------
    rdf : np.array
        the rdf of the system ((# of bins, ), dtype: float)
    bin_centers : np.array
        the centers of the bins ((# of bins, ), dtype: float)
    bin_edges : np.array
        the edges of the bins ((# of bins + 1, ), dtype: float)
    """
    def __init__(self, raptor_filename, traj_filename, group1 = None, group2 = None, group2_types = None, n_bins = 100, max_r = 10.0, min_r = 0.0, density = True, output_freq = None, tell_time = 0):
        super().__init__()
        self.raptor_filename = raptor_filename
        self.traj_filename = traj_filename
        self.group1 = group1
        self.group2 = group2
        self.group2_types = group2_types
        self.n_bins = n_bins
        self.max_r = max_r
        self.min_r = min_r
        self.density = density
        self.output_freq = output_freq
        self.tell_time = tell_time

        self._prepare()

        self._has_calculated = False # whether the `calculate` method has been called

    def _prepare(self):
        self._raptor_reader = RaptorReader(self.raptor_filename,self.output_freq)
        self._lammps_reader = LammpsReader(self.traj_filename)
        self._n_frames = min(self._raptor_reader.n_frames, self._lammps_reader.n_frames)
        if self.group1 is None:
            self.group1 = list(range(self._raptor_reader.n_complexes))
        if self.group2 is None and self.group2_types is None:
            self.group2 = list(range(self._lammps_reader.n_atoms))
        elif self.group2 is None and self.group2_types is not None:
            self._group2_by_types = True
            self._num_atoms_in_group2 = 0
        elif self.group2 is not None and self.group2_types is None:
            self._group2_by_types = False
        else:
            raise ValueError("You cannot use `group2` and `group2_types` at the same time.")

        self._cumulative_volumes = 0.0
        self._distances = [] # store all the distances of pairs between group1 and group2

    def _main_engine(self):
        """
        Calculate the rdf
        
        TODO: 
        1. add the option to calculate the rdf between the same group of atoms
        2. multiprocessing
        """
        for iframe in range(self._n_frames):
            if self.tell_time > 0 and iframe % self.tell_time == 0:
                print(f"Calculating frame {iframe}...")
            raptor_frame = self._raptor_reader[iframe]
            lammps_frame = self._lammps_reader[iframe]
            box_boundaries = lammps_frame.box_boundaries
            self._cumulative_volumes += np.prod(box_boundaries[:,1] - box_boundaries[:,0])
            cec_coords = np.array(raptor_frame.cec_coordinates)[self.group1,:]
            atom_coords = np.array(lammps_frame.positions)[:,:]
            if self._group2_by_types:
                atom_types = lammps_frame.atom_types
                atom_coords = atom_coords[np.isin(atom_types, self.group2_types),:]
                self._num_atoms_in_group2 += atom_coords.shape[0]
            else:
                atom_coords = atom_coords[self.group2,:]
            distances,_ = minimum_image_distance(cec_coords,atom_coords, box_boundaries)
            self._distances.extend(distances)
        
    def _postprocess(self):
        self._distances = np.concatenate(self._distances)
        self._rdf, self._bin_edges = np.histogram(self._distances, bins=self.n_bins, range=(self.min_r, self.max_r))
        self._rdf = self._rdf.astype(np.float64)
        self._bin_centers = (self._bin_edges[1:] + self._bin_edges[:-1])/2

        # normalize the counts
        n_group1 = len(self.group1)
        if self._group2_by_types:
            n_group2 = self._num_atoms_in_group2
        else:
            n_group2 = len(self.group2) * self._n_frames
        norm = n_group1 * n_group2 * self._n_frames / self._cumulative_volumes
        vols = 4/3 * np.pi * (self._bin_edges[1:]**3 - self._bin_edges[:-1]**3)
        norm *= vols

        self._rdf /= norm

    def calculate(self):
        print("Calculating rdf...")
        self._main_engine()
        self._postprocess()
        self._has_calculated = True

    @property
    def rdf(self):
        if not self._has_calculated:
            raise ValueError("You need to call the `calculate` method first.")
        return self._rdf
    
    @property
    def bin_centers(self):
        if not self._has_calculated:
            raise ValueError("You need to call the `calculate` method first.")
        return self._bin_centers
    
    @property
    def bin_edges(self):
        if not self._has_calculated:
            raise ValueError("You need to call the `calculate` method first.")
        return self._bin_edges
    