# -*- encoding: utf-8 -*-
'''
Python version   : python 3.10.12
Filename         : lammpsframe.py
Description      : 
Time             : 2023/09/13 15:27:03
Author           : Sijia Chen
Version          : 1.0
Email            : sijiachen@uchicago.edu
'''



import numpy as np

class LammpsFrame(object):
    """
    RaptorFrame stores the information of one frame in RAPTOR output file.

    Attributes
    ----------
    n_atoms : int
        The number of complexes in the simulation.
    index : int
        The index of current frame, starting from 0 for each trajectory file.
    timestep : int
        The timestep of current frame.
    box_boundaries : np.array, the box boundaries of this frame with shape (3, 2)
        Store the box boundaries in this frame. Only support orthorhombic box.
    charges : np.array, the charges of atoms in this frame with shape (n_atoms, )
        Get the charges of atoms in this frame.
    wrapped_positions : np.array, the wrapped positions of atoms in this frame with shape (n_atoms, 3)
        Get the wrapped positions of atoms in this frame.
    unwrapped_positions : np.array, the unwrapped positions of atoms in this frame with shape (n_atoms, 3)
        Get the unwrapped positions of atoms in this frame.
    images : np.array, the box images of atoms in this frame with shape (n_atoms, 3)
        Get the box images of atoms in this frame.
    atom_types : np.array, the atom types of atoms in this frame with shape (n_atoms, )
        Get the atom types of atoms in this frame.
    mol_ids : np.array, the molecule ids of atoms in this frame with shape (n_atoms, )
        Get the molecule ids of atoms in this frame.
    velocities : np.array, the velocities of atoms in this frame with shape (n_atoms, 3)
        Get the velocities of atoms in this frame.
    forces : np.array, the forces of atoms in this frame with shape (n_atoms, 3)
        Get the forces of atoms in this frame.
    positions : np.array, the positions of atoms in this frame with shape (n_atoms, 3)
        Get the positions of atoms in this frame. 
        If the unwrapped positions are available, return the unwrapped positions; 
        otherwise, return the wrapped positions.

    Notes
    -----

    """
    def __init__(self, n_atoms: int, **kwargs) -> None:
        """
        Initialize the RaptorFrame object, which stores the information of one frame
        
        Parameters
        ----------
        n_atoms : int
        
        """
        self._n_atoms = n_atoms
        self._index = -1
        self._timestep = -1
        self._box_boundaries = np.zeros((3, 2), dtype=np.float32)

        self._has_charges = kwargs.get('charges', False) # False
        self._has_wrapped_positions = kwargs.get('wrapped_positions', False) # False
        self._has_unwrapped_positions = kwargs.get('unwrapped_positions', False) # False
        self._has_images = kwargs.get('images', False) # False
        self._has_atom_types = kwargs.get('atom_types', False) # False
        self._has_mol_ids = kwargs.get('mol_ids', False) # False
        self._has_velocities = kwargs.get('velocities', False) # False
        self._has_forces = kwargs.get('forces', False) # False
        self._has_masses = kwargs.get('masses', False) # False
        self._has_elements = kwargs.get('elements', False) # False

    @property
    def n_atoms(self) -> int:
        """
        The number of complexes in the simulation.
        
        Returns
        -------
        - n_atoms : int
        
        """
        return self._n_atoms
        
    @n_atoms.setter
    def n_atoms(self, new_n_atoms):
        if new_n_atoms == self._n_atoms:
            pass
        else:
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
    def box_boundaries(self):
        """
        Store the box boundaries in this frame. Only support orthorhombic box.
        
        Returns
        -------
        box_boundary : np.array, the box boundaries of this frame with shape (3, 2)
        """
        return self._box_boundaries
        
    @box_boundaries.setter
    def box_boundaries(self, new_box_boundaries):
        self._box_boundaries = new_box_boundaries

    @property
    def charges(self):
        """
        Get the charges of atoms in this frame.
        
        Returns
        -------
        charges: np.array, the charges of atoms in this frame with shape (n_atoms, )

        """
        if not self._has_charges:
            raise ValueError("This frame does not have charges!")
        else:
            return self._charges
    
    @charges.setter
    def charges(self, new_charges):
        self._has_charges = True
        self._charges = new_charges

    @property
    def wrapped_positions(self):
        """
        Get the wrapped positions of atoms in this frame.
        
        Returns
        -------
        wrapped_positions: np.array, the wrapped positions of atoms in this frame with shape (n_atoms, 3)

        """
        if not self._has_wrapped_positions:
            raise ValueError("This frame does not have wrapped positions!")
        else:
            return self._wrapped_positions
        
    @wrapped_positions.setter
    def wrapped_positions(self, new_wrapped_positions):
        self._has_wrapped_positions = True
        self._wrapped_positions = new_wrapped_positions

    @property
    def unwrapped_positions(self):
        """
        Get the unwrapped positions of atoms in this frame.
        
        Returns
        -------
        unwrapped_positions: np.array, the unwrapped positions of atoms in this frame with shape (n_atoms, 3)

        """
        if not self._has_unwrapped_positions:
            raise ValueError("This frame does not have unwrapped positions!")
        else:
            return self._unwrapped_positions
        
    @unwrapped_positions.setter
    def unwrapped_positions(self, new_unwrapped_positions):
        self._has_unwrapped_positions = True
        self._unwrapped_positions = new_unwrapped_positions

    @property
    def images(self):
        """
        Get the box images of atoms in this frame.
        
        Returns
        -------
        images: np.array, the box images of atoms in this frame with shape (n_atoms, 3)

        """
        if not self._has_images:
            raise ValueError("This frame does not have box images!")
        else:
            return self._images
    
    @images.setter
    def images(self, new_images):
        self._has_images = True
        self._images = new_images

    @property
    def atom_types(self):
        """
        Get the atom types of atoms in this frame.
        
        Returns
        -------
        atom_types: np.array, the atom types of atoms in this frame with shape (n_atoms, )

        """
        if not self._has_atom_types:
            raise ValueError("This frame does not have atom types!")
        else:
            return self._atom_types
        
    @atom_types.setter
    def atom_types(self, new_atom_types):
        self._has_atom_types = True
        self._atom_types = new_atom_types

    @property
    def mol_ids(self):
        """
        Get the molecule ids of atoms in this frame.
        
        Returns
        -------
        mol_ids: np.array, the molecule ids of atoms in this frame with shape (n_atoms, )

        """
        if not self._has_mol_ids:
            raise ValueError("This frame does not have molecule ids!")
        else:
            return self._mol_ids
        
    @mol_ids.setter
    def mol_ids(self, new_mol_ids):
        self._has_mol_ids = True
        self._mol_ids = new_mol_ids

    @property
    def velocities(self):
        """
        Get the velocities of atoms in this frame.
        
        Returns
        -------
        velocities: np.array, the velocities of atoms in this frame with shape (n_atoms, 3)

        """
        if not self._has_velocities:
            raise ValueError("This frame does not have velocities!")
        else:
            return self._velocities
        
    @velocities.setter
    def velocities(self, new_velocities):
        self._has_velocities = True
        self._velocities = new_velocities

    @property
    def forces(self):
        """
        Get the forces of atoms in this frame.
        
        Returns
        -------
        forces: np.array, the forces of atoms in this frame with shape (n_atoms, 3)

        """
        if not self._has_forces:
            raise ValueError("This frame does not have forces!")
        else:
            return self._forces
        
    @forces.setter
    def forces(self, new_forces):
        self._has_forces = True
        self._forces = new_forces

    @property
    def masses(self):
        """
        Get the masses of atoms in this frame.
        
        Returns
        -------
        masses: np.array, the masses of atoms in this frame with shape (n_atoms, )

        """
        if not self._has_masses:
            raise ValueError("This frame does not have masses!")
        else:
            return self._masses
        
    @masses.setter
    def masses(self, new_masses):
        self._has_masses = True
        self._masses = new_masses

    @property
    def elements(self):
        """
        Get the elements of atoms in this frame.
        
        Returns
        -------
        elements: np.array, the elements of atoms in this frame with shape (n_atoms, )

        """
        if not self._has_elements:
            raise ValueError("This frame does not have elements!")
        else:
            return self._elements
        
    @elements.setter
    def elements(self, new_elements):
        self._has_elements = True
        self._elements = new_elements

    @property
    def positions(self):
        """
        Get the positions of atoms in this frame. 
        If the unwrapped positions are available, return the unwrapped positions; 
        otherwise, return the wrapped positions.
        
        Returns
        -------
        positions: np.array, the positions of atoms in this frame with shape (n_atoms, 3)

        """
        if self._has_unwrapped_positions:
            return self._unwrapped_positions
        elif self._has_wrapped_positions:
            return self._wrapped_positions
        else:
            raise ValueError("This frame does not have positions!")