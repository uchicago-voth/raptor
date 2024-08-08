# -*- encoding: utf-8 -*-
'''
Python version   : python 3.10.12
Filename         : correlation.py
Description      : 
Time             : 2023/09/18 16:13:01
Author           : Sijia Chen
Version          : 1.0
Email            : sijiachen@uchicago.edu
'''



# This module is modified from MDAnalysis.lib.correlations
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Begin of copyrignt of MDAnalysis.lib.correlations
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# MDAnalysis --- https://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
# doi: 10.25080/majora-629e541a-00e
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
# :Authors: Paul Smith & Mateusz Bieniek
# :Year: 2020
# :Copyright: GNU Public License v2
#
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# End of copyrignt of MDAnalysis.lib.correlations
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

"""Correlations utilities --- :mod:`MDAnalysis.lib.correlations`
=================================================================================


:Authors: Paul Smith & Mateusz Bieniek
:Year: 2020
:Copyright: GNU Public License v2

.. versionadded:: 1.0.0

This module is primarily for internal use by other analysis modules. It
provides functionality for calculating the time autocorrelation function
of a binary variable (i.e one that is either true or false at each
frame for a given atom/molecule/set of molecules). This module includes
functions for calculating both the time continuous autocorrelation and
the intermittent autocorrelation. The function :func:`autocorrelation`
calculates the continuous autocorrelation only. The data may be
pre-processed using the function :func:`intermittency` in order to
acount for intermittency before passing the results to
:func:`autocorrelation`.

This module is inspired by seemingly disparate analyses that rely on the same
underlying calculation, including the survival probability of water around
proteins :footcite:p:`ArayaSecchi2014`, hydrogen bond lifetimes
:footcite:p:`Gowers2015,ArayaSecchi2014`, and the rate of cholesterol
flip-flop in lipid bilayers :footcite:p:`Gu2019`.
"""

from copy import deepcopy
import numpy as np

class ContinuousProtonCorrelation:
    def __init__(self,indexes,tau_max,window_step=1,intermittency=0,tell_time=None) -> None:
        """Implementation of the continuous proton correlation function (defined in equation 2 of [J. Chem. Phys. 154, 194506 (2021); doi: 10.1063/5.0040758]).

        The proton identity correlation function is defined as:

        $$C(t) =  \frac{ \langle H(t)H(0) \rangle} {  \langle H \rangle }$$
        
        where $H(t)$ is 1 as long as the hydronium-like identity has not changed from that of H(0) and 0 once it has changed.

        Parameters
        ----------
        indexes : list of int
            the pivot ids of the cec in each frame
        tau_max : int
            the maximum value of tau for which the autocorrelation is calculated
        window_step : int, optional
            the number of frames between each window, default is 1
        intermittency : int, optional
            a filter to filter out the rattling effect, default is 0.

        Methods
        -------
        calc_autocorrelation
            calculates the autocorrelation function for the given indexes
        
        Static Methods
        --------------
        _correct_intermittency
            mitigates the effect of rattling


        """
        self.indexes=indexes
        self.tau_max=tau_max
        self.window_step=window_step
        self.intermittency=intermittency
        self._tell_time = tell_time


    @staticmethod
    def _correct_intermittency(indexes,intermittency=0):
        """
        mitigates the effect of rattling
        if A(t1)->B(t2)->A(t3) and (t3-t1)<=(intermittency+1), then all data between t1 and t3 will be changed to A
        """

        if intermittency == 0:
            return indexes

        num_indexes = len(indexes)
        """
        todo: any simpler method?
        """
        for i,index in enumerate(indexes):
            for j in range(1, intermittency+2 if i+intermittency+2 <= num_indexes else num_indexes - i):
                if indexes[i+j] == index:
                    # correct all data between indexes[i] and indexes[i+j] to be indexes[i]
                    indexes[i:i+j] = index
                    break

        
        return indexes

    def calc_autocorrelation(self):
        """Implementation of a discrete autocorrelation function.

        Returns
        --------
        taus : list of int
            the values of tau for which the autocorrelation was calculated
        timeseries : list of float
            the autocorelation values for each of the tau values
        timeseries_data : list of list of int
            the raw data from which the autocorrelation is computed, i.e :math:`S(\tau)` at each window.
            This allows the time dependant evolution of :math:`S(\tau)` to be investigated.

        TODO
        ----
        increase the speed of the calculation, this method is too slow. Any idea?

        """

        taus=list(range(1,self.tau_max+1))
        timeseries_data=[ [] for _ in range(self.tau_max)]

        self.indexes=self._correct_intermittency(self.indexes,intermittency=self.intermittency)

        # calculate autocorrelation
        for t in range(0,len(self.indexes),self.window_step):
            if (self._tell_time!= None) and t % self._tell_time == 0:
                print("Already calculate {} frames...".format(t))

            for tau in taus:
                if tau + t + 1 >= len(self.indexes):
                    break

                n = 1 if len(set(self.indexes[t:t + tau + 1])) == 1 else 0 # 1 if all the indexes are the same which means pivot id of the cec doesn't change during the time, 0 otherwise
                timeseries_data[tau-1].append(n)

        timeseries = [np.mean(x) for x in timeseries_data]

        taus.insert(0,0)
        timeseries.insert(0,1)

        return taus,timeseries,timeseries_data


class ProtonIdentityCorrelation:
    def __init__(self,indexes,tau_max,window_step=1,intermittency=0,tell_time=None) -> None:
        """Implementation of the proton identity correlation function (defined in equation 1 of [J. Chem. Phys. 154, 194506 (2021); doi: 10.1063/5.0040758]).

        The proton identity correlation function is defined as:

        $$C(t) =  \frac{ \langle h(t)h(0) \rangle} {  \langle h \rangle }$$
        
        where $h(t)$ is 1 if it is equal to $h(0)$ and 0 if it is not.

        Parameters
        ----------
        indexes : list of int
            the pivot ids of the cec in each frame
        tau_max : int
            the maximum value of tau for which the autocorrelation is calculated
        window_step : int, optional
            the number of frames between each window, default is 1
        intermittency : int, optional
            a filter to filter out the rattling effect, default is 0.

        Methods
        -------
        calc_autocorrelation
            calculates the autocorrelation function for the given indexes
        
        Static Methods
        --------------
        _correct_intermittency
            mitigates the effect of rattling


        """
        self.indexes=indexes
        self.tau_max=tau_max
        self.window_step=window_step
        self.intermittency=intermittency
        self._tell_time = tell_time


    @staticmethod
    def _correct_intermittency(indexes,intermittency=0):
        """
        mitigates the effect of rattling
        if A(t1)->B(t2)->A(t3) and (t3-t1)<=(intermittency+1), then all data between t1 and t3 will be changed to A
        """

        if intermittency == 0:
            return indexes

        num_indexes = len(indexes)
        """
        todo: any simpler method?
        """
        for i,index in enumerate(indexes):
            for j in range(1, intermittency+2 if i+intermittency+2 <= num_indexes else num_indexes - i):
                if indexes[i+j] == index:
                    # correct all data between indexes[i] and indexes[i+j] to be indexes[i]
                    indexes[i:i+j] = index
                    break

        
        return indexes

    def calc_autocorrelation(self):
        """Implementation of a discrete autocorrelation function.

        Returns
        --------
        taus : list of int
            the values of tau for which the autocorrelation was calculated
        timeseries : list of float
            the autocorelation values for each of the tau values
        timeseries_data : list of list of int
            the raw data from which the autocorrelation is computed, i.e :math:`S(\tau)` at each window.
            This allows the time dependant evolution of :math:`S(\tau)` to be investigated.

        TODO
        ----
        increase the speed of the calculation, this method is too slow. Any idea?

        """

        taus=list(range(1,self.tau_max+1))
        timeseries_data=[ [] for _ in range(self.tau_max)]

        self.indexes=self._correct_intermittency(self.indexes,intermittency=self.intermittency)

        # calculate autocorrelation
        for t in range(0,len(self.indexes),self.window_step):
            if (self._tell_time!= None) and t % self._tell_time == 0:
                print("Already calculate {} frames...".format(t))

            for tau in taus:
                if tau + t +1 >= len(self.indexes):
                    break

                n = 1 if self.indexes[t] == self.indexes[t + tau + 1] else 0 # 1 if all the indexes are the same which means pivot id of the cec doesn't change during the time, 0 otherwise
                timeseries_data[tau-1].append(n)

        timeseries = [np.mean(x) for x in timeseries_data]

        taus.insert(0,0)
        timeseries.insert(0,1)

        return taus,timeseries,timeseries_data