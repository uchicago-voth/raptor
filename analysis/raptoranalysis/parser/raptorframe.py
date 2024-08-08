# -*- encoding: utf-8 -*-
# python 3.10.12
'''
Filename         : RaptorFrame.py
Description      : A class to store the information of one frame in RAPTOR output file
Time             : 2023/08/30 16:23:51
Last Modified    : 2023/08/31 00:43:20
Author           : Sijia Chen
Version          : 1.0
Email            : sijiachen@uchicago.edu
'''

class RaptorFrame(object):
    """
    RaptorFrame stores the information of one frame in RAPTOR output file.

    Attributes
    ----------
    n_complexes : int, the number of complexes in the simulation
    index : int, the index of current frame, starting from 0
    timestep : int, the timestep of current frame
    energy : dict[str, float], energy information of this frame if there is energy, otherwise `None`; keys of this dict: 'environment', 'complex', 'total'
    decomposed_env_energy: dict[str, float], decomposed energy of the environment if there is decomposed energy, otherwise `None`; keys of this dict: 'vdw', 'coul', 'bond', 'angle', 'dihedral', 'improper', 'kspace'
    center_location: a `numpy.ndarray` with dtype `numpy.int32`, center location data of this frame if there is center location, otherwise `None` with shape ``(n_complexes, 3)`` for all complexes
    complexes: list[dict], each dict stores the information of one complex

    Methods
    -------
    get_complex(complex_id): get the information of a specific complex in this frame, ``complex_id`` starts from 1.
    set_complex(new_complex): set the information of a specific complex in this frame, complex_id starts from 1.

    Notes
    ------
    The content of the attribute `complexes` dict is as follows by default:
    {"id": None, "n_states": None, "states": None, "decomposed_complex_energy_diagonal": None, "decomposed_complex_energy_offdiagonal": None, "extra_coupling": None,"eigen_vector": None, "cec_coordinate": None,"cec_v2_coordinate":None,"next_pivot_state":None}

    If there is corresponding information in the RAPTOR output file, the format of the information will be updated as follows:
    - "id": int, the id of the complex, starting from 1
    - "n_states": int, the number of states in this complex
    - "states": list[dict[str, int]], each dict stores the information of one state. The keys of the dict are "id", "parent", "shell", "mol_A", "mol_B", "react", "path", "extra_cpl"
    - "decomposed_complex_energy_diagonal": list[dict[str, float]], the diagonal elements of the decomposed complex energy matrix. The keys of the dict are "id", "total", "vdw", "coul", "bond", "angle", "dihedral", "improper", "kspace", "repulsive". There is no gurantee that the keys are all present or match with above, since the output format of RAPTOR maybe not consistent for different version.
    - "decomposed_complex_energy_offdiagonal": list[dict[str, float]], the off-diagonal elements of the decomposed complex energy matrix. The keys of the dict should be "id", "energy",  "A_Rq", "Vij_const",  "Vij". There is no gurantee that the keys are all present or match with above, since the output format of RAPTOR maybe not consistent for different version.
    - "extra_coupling": list[dict[str, float]], the extra coupling energy between the states. The keys of the dict should be "I", "J", "energy". "I" and "J" are the indexes of the extra coupled states. There is no gurantee that the keys are all present or match with above, since the output format of RAPTOR maybe not consistent for different version.
    """
    def __init__(self, n_complexes: int, **kwargs) -> None:
        """
        Initialize the RaptorFrame object, which stores the information of one frame
        
        Parameters
        ----------
        n_complexes : int
        
        """
        self._n_complexes = n_complexes
        self._index = -1
        self._timestep = -1
        self._complexes = [dict() for _ in range(self._n_complexes)]

        self._has_center_location = kwargs.get('center_location', False) # False
        self._has_n_states = kwargs.get('n_states', False) # False
        self._has_states_info = kwargs.get('states_info', False) # False
        self._has_energy = kwargs.get('energy', False) # False
        self._has_decomposed_energy = kwargs.get('decomposed_energy', False) # False

    @property
    def n_complexes(self) -> int:
        """
        The number of complexes in the simulation.
        
        Returns
        -------
        - n_complexes : int
        
        """
        return self._n_complexes
        
    @n_complexes.setter
    def n_complexes(self, new_n_complexes):
        raise ValueError("Cannot change the number of complexes after initilization!")

    @property
    def index(self) -> int:
        """
        The index of current frame, starting from 0.
        
        Returns
        -------
        - index : int
        
        """
        return self._index
        
    @index.setter
    def index(self, new_index):
        self._index = new_index

    @property
    def timestep(self) -> int:
        """
        The timestep of current frame.
        
        Returns
        -------
        - timestep : int
        
        """
        return self._timestep
        
    @timestep.setter
    def timestep(self, new_timestep):
        self._timestep = new_timestep

    @property
    def energy(self):
        """
        Store the energy information in this frame.
        
        Returns
        -------
        energy : dict[str, float], energy information of this frame; keys of this dict: 'environment', 'complex', 'total'
        """
        if self._has_energy:
            return self._energy
        else:
            raise ValueError("This frame doesn't have energy information")
        
    @energy.setter
    def energy(self, new_energy):
        self._has_energy = True
        self._energy = new_energy

    @property
    def decomposed_env_energy(self):
        """
        Store the decomposed environment energy information in this frame.
        
        Returns
        -------
        - decomposed_env_energy: dict[str, float], decomposed energy of the environment; keys of this dict: 'vdw', 'coul', 'bond', 'angle', 'dihedral', 'improper', 'kspace'
        
        """
        if self._has_decomposed_energy:
            return self._decomposed_env_energy
        else:
            raise ValueError("This frame doesn't have decomposed energy information")
        
    @decomposed_env_energy.setter
    def decomposed_env_energy(self, new_decomposed_env_energy):
        self._has_decomposed_energy = True
        expected_keys = ('vdw', 'coul', 'bond', 'angle', 'dihedral', 'improper', 'kspace')
        check_sanity = all((x in new_decomposed_env_energy) for x in expected_keys)
        if not check_sanity:
            raise ValueError("Incomprehensive information of decomposed environment energy.")
        else:
            self._decomposed_env_energy = new_decomposed_env_energy

    
    @property
    def center_location(self):
        """
        Store the center location information in this frame.
        For each complex, there are three data: center_id, molecule_id, and location_rank
        
        Returns
        -------
        center_location: a numpy.ndarray with dtype numpy.int32
                center location data of this frame with shape ``(n_complexes, 3)`` for all complexes
        """
        if self._has_center_location:
            return self._center_location
        else:
            raise ValueError("This frame doesn't have center location information")
        
    @center_location.setter
    def center_location(self, new_center_location):
        self._has_center_location = True
        self._center_location = new_center_location

    @property
    def complexes(self):
        """
        Get all the complexes in this frame.
        
        Returns
        -------
        complexes: list[dict], each dict stores the information of one complex

        Notes
        ------
        The format of the complex dict is as follows by default:
        {"id": None, "n_states": None, "states": None, "decomposed_complex_energy_diagonal": None, "decomposed_complex_energy_offdiagonal": None, "extra_coupling": None,"eigen_vector": None, "cec_coordinate": None,"cec_v2_coordinate":None,"next_pivot_state":None}

        If there is corresponding information in the RAPTOR output file, the format of the information will be updated as follows:
        - "id": int, the id of the complex, starting from 1 to n_complexes
        - "n_states": int, the number of states in this complex
        - "states": list[dict[str, int]], each dict stores the information of one state. The keys of the dict are "id", "parent", "shell", "mol_A", "mol_B", "react", "path", "extra_cpl"
        - "decomposed_complex_energy_diagonal": list[dict[str, float]], the diagonal elements of the decomposed complex energy matrix. The keys of the dict are "id", "total", "vdw", "coul", "bond", "angle", "dihedral", "improper", "kspace", "repulsive". There is no gurantee that the keys are all present or match with above, since the output format of RAPTOR maybe not consistent for different version.
        - "decomposed_complex_energy_offdiagonal": list[dict[str, float]], the off-diagonal elements of the decomposed complex energy matrix. The keys of the dict should be "id", "energy",  "A_Rq", "Vij_const",  "Vij". There is no gurantee that the keys are all present or match with above, since the output format of RAPTOR maybe not consistent for different version.
        - "extra_coupling": list[dict[str, float]], the extra coupling energy between the states. The keys of the dict should be "I", "J", "energy". "I" and "J" are the indexes of the extra coupled states. There is no gurantee that the keys are all present or match with above, since the output format of RAPTOR maybe not consistent for different version.
        - "eigen_vector": list[float], the eigen vector of the complex
        - "cec_coordinate": list[float], the coordinate of the center of excess charge
        - "cec_v2_coordinate": list[float], the coordinate of the center of excess charge v2
        - "next_pivot_state": int, the index of the next pivot state (0 means pivot state doesn't change, 1 means proton hops to the next state)
        """
        return self._complexes
    
    @complexes.setter
    def complexes(self, new_complexes):
        self._complexes = new_complexes

    def get_complex(self, complex_id):
        """
        Get the information of a specific complex in this frame.


        Parameters
        ----------
        complex_id : int
            The id of the complex, starting from 1.

        Returns
        -------
        complex: dict, stores the information of the complex
        """
        return self._complexes[complex_id-1]
    
    def set_complex(self, new_complex):
        self._complexes[new_complex['id'] - 1] = new_complex

    @property
    def pivot_ids(self):
        """
        Get the pivot ids of all the complexes in this frame.

        Returns
        -------
        pivot_ids: list[int]
                pivot ids (mol ids) of all the complexes in this frame with shape ``(n_complexes,)`` for all complexes
        """
        if self._has_center_location:
            return self._center_location[:,1]
        elif self._has_states_info:
            return [complex['states'][0]['mol_B'] for complex in self._complexes]
        else:
            raise ValueError("This frame doesn't have pivot id information or states information")
        
    @property
    def eigen_vectors(self):
        """
        Get the eigen vectors of all the complexes in this frame.
        
        Returns
        -------
        eigen_vectors: list[list[float]]
                eigen vectors of all the complexes in this frame with shape ``(n_complexes, (*)n_states)`` for all complexes, the second dimension (n_states) is the number of states in the complex and may be different for different complexes
        """
        return [complex['eigen_vector'] for complex in self._complexes]
    
    @property
    def cec_coordinates(self):
        """
        Get the cec coordinates of all the complexes in this frame.
        
        Returns
        -------
        cec_coordinates: list[list[float]]
                cec coordinates of all the complexes in this frame with shape ``(n_complexes, 3)`` for all complexes
        """
        return [complex['cec_coordinate'] for complex in self._complexes]