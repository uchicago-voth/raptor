# -*- encoding: utf-8 -*-
# python 3.10.12
# -*- encoding: utf-8 -*-
'''
Filename         : LammpsReader.py
Description      : 
Time             : 2023/09/05 15:17:42
Author           : Sijia Chen
Version          : 1.0
Email            : sijiachen@uchicago.edu
'''

#---------Import modules-------------

import os
import numpy as np
import functools
from .lammpsframe import LammpsFrame



class LammpsReader(object):
    """
    LammpsReader reads in the trajectory file and stores the information of all frames in the trajectory file.
    
    Parameters
    ----------
    filename : str
        Name of the RAPTOR output file
    verbose : bool, optional
        If True, print out more information. Mostly for debugging purpose.

    Attributes
    ----------
    n_frames : int
        The number of frames in the trajectory file.
    n_atoms : int
        The number of atoms in the simulation.
    timestep : int

    """
    # todo: add support for reading in multiple trajectory files; "_n_frames" should be the total number of frames in all trajectory files, therefore it should be a class property
    # maybe "_n_frames", "_offsets", "_cached" should be class properties? a experiment for now
    # a example of class property: https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
    _n_frames = 0
    _frame = LammpsFrame
    _offsets = []

    def __init__(self, filename, verbose = False,**kwargs) -> None:

        # initialize filename and check sanity
        self.filename = filename
        self._check_sanity_filename()
        self._verbose = verbose # if True, print out more information. Mostly for debugging purpose

        # initialize frame class-related arguments
        self._frame_kwargs = self._parse_frame_kwargs(kwargs)
        # initialize cached dictionary
        self._cached = dict()

        self._reopen()
        self._has_initialized = False
        self._read_first_time()

    def _reopen(self):
        self.close()
        self._file = open(self.filename, 'r')
        self.frame= self._frame(self.n_atoms, **self._frame_kwargs)
        self.frame.index = -1
        
    def _check_sanity_filename(self):
        # check if the input file exists
        if isinstance(self.filename, str) and os.path.isfile(self.filename):
            pass
        elif not isinstance(self.filename, str):
            raise TypeError("Input file name must be a string!")
        elif not os.path.isfile(self.filename):
            raise FileNotFoundError("File {} does not exist!".format(self.filename))
        else:
            raise TypeError("Unknown error when parsing the argument {}!".format(self.filename))

        # check if the input file is empty
        if os.stat(self.filename).st_size == 0:            
            raise ValueError("Input file {} is empty!".format(self.filename))
        
    @staticmethod
    def _parse_frame_kwargs(kwargs) -> dict:
        frame_kwargs = dict()
        return frame_kwargs


    # @property
    # def time(self):
    #     return self.frame.time

    # @property
    # def frame(self) -> LammpsFrame:
    #     return self._frame
    
    # @frame.setter
    # def frame(self, new_frame):
    #     self._frame = new_frame
    
    @property
    def timestep(self):
        return self.frame.timestep
    
    @property
    def index(self) -> int:
        """
        return the index of the current frame, 0-based

        Returns
        -------
        - index : int
        """
        return self.frame.index
    
    @functools.cached_property
    def n_atoms(self):
        with open(self.filename, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            n_atoms = int(f.readline())
        self._cached["n_atoms"] = n_atoms
        return n_atoms

    @functools.cached_property
    def n_frames(self):
        lines_per_frame = 9 + self.n_atoms
        counter = 0
        offsets = []
        with open(self.filename, 'r') as f:
            line = True
            while line:
                if counter % lines_per_frame == 0 :
                    offsets.append(f.tell())
                counter += 1
                line = f.readline()
        self._offsets = offsets[:-1] # last position is EOF
        self._n_frames += len(self._offsets)
        self._cached["n_frames"] = self._n_frames
        return self._n_frames
    
    def __len__(self):
        return self.n_frames
    
    def next(self) -> LammpsFrame:
        try:
            frame = self._read_next_frame()
        except EOFError:
            self.rewind()
            raise StopIteration from None
        else:
            pass
        
        return frame


    def __next__(self):
        return self.next()

    def close(self):
        if hasattr(self, '_file'):
            self._file.close()

    def _read_next_frame(self) -> LammpsFrame:
        # initialize 
        if not self._has_initialized:
            raise RuntimeError("Please initialize the reader first!")
        
        f = self._file
        frame = self.frame
        frame.index += 1
        if frame.index >= len(self):
            raise EOFError("End of file reached!")
        

        # go to the TIMESTEP line of this frame
        f.seek(self._offsets[frame.index])

        # read the TIMESTEP line and grap the timestep
        f.readline() # skip the "ITEM: TIMESTEP" line
        frame.timestep = int(f.readline().strip())

        # read number of atoms
        f.readline() # skip the "ITEM: NUMBER OF ATOMS" line
        frame.n_atoms = int(f.readline().strip())

        # read the box boundaries
        f.readline() # skip the "ITEM: BOX BOUNDS" line
        box_boundaries = []
        for i in range(3):
            box_boundaries.append([float(x) for x in f.readline().strip().split()])
        frame.box_boundaries = np.array(box_boundaries)
        
        # read the atom attributes
        attrs = f.readline().strip().split()[2:]
        if attrs != self.attributes:
            raise ValueError("Inconsistent attributes in frame {} at timestep {}, compared to the first frame's attributes.".format(frame.index, frame.timestep, self.attributes))
        
        # read the remaing lines of this frame
        atom_info=[]
        for _ in range(self.n_atoms):
            atom_info.append(f.readline().strip().split())
        atom_info = np.array(atom_info)

        # parse the atom information
        if self._has_atom_ids:
            atom_ids = np.array([int(x[self.attrs_to_cols["id"]]) for x in atom_info], dtype=np.int64)
            sortedargs = np.argsort(atom_ids)
        else:
            sortedargs = np.arange(frame.n_atoms)
        if self._has_charges:
            frame.charges = np.array([float(x[self.attrs_to_cols["q"]]) for x in atom_info], dtype=np.float64)[sortedargs]
        if self._has_wrapped_positions:
            frame.wrapped_positions = np.array([[float(y) for y in x[self.attrs_to_cols_wrapped_positions]] for x in atom_info], dtype=np.float64)[sortedargs]
        if self._has_unwrapped_positions:
            frame.unwrapped_positions = np.array([[float(y) for y in x[self.attrs_to_cols_unwrapped_positions]] for x in atom_info], dtype=np.float64)[sortedargs]
        if self._has_images:
            frame.images = np.array([[int(y) for y in x[self.attrs_to_cols_images]] for x in atom_info], dtype=np.int64)[sortedargs]
        if self._has_atom_types:
            try:
                frame.atom_types = np.array([int(x[self.attrs_to_cols["type"]]) for x in atom_info], dtype=np.int64)[sortedargs]
            except TypeError:
                frame.atom_types = np.array([x[self.attrs_to_cols["type"]] for x in atom_info], dtype=np.string_)[sortedargs]
        if self._has_mol_ids:
            frame.mol_ids = np.array([int(x[self.attrs_to_cols["mol"]]) for x in atom_info], dtype=np.int64)[sortedargs]
        if self._has_velocities:
            frame.velocities = np.array([[float(y) for y in x[self.attrs_to_cols_velocities]] for x in atom_info], dtype=np.float64)[sortedargs]
        if self._has_forces:
            frame.forces = np.array([[float(y) for y in x[self.attrs_to_cols_forces]] for x in atom_info], dtype=np.float64)[sortedargs]
        if self._has_elements:
            frame.elements = np.array([x[self.attrs_to_cols["element"]] for x in atom_info], dtype=np.string_)[sortedargs]




        return frame
        
        
    def rewind(self) -> LammpsFrame:
        self._reopen()
        self.next()

    def _read_frame(self, index) -> LammpsFrame:
        """
        Read a specific frame `index` from the trajectory and return it.
        
        Parameters
        ----------
        index : int
            Index of the frame to read. 0-based.

        Returns
        -------
        frame : RaptorFrame
            Frame object.
        """
        self._index = index-1
        self._file.seek(self._offsets[index])
        return self._read_next_frame()
    
    def __getitem__(self, index) -> LammpsFrame:
        """Return the frame with index `index`.

        Parameters
        ----------
        index : int
            Index of the frame to read. 0-based.

        """
        if isinstance(index, int) and index >= 0:
            return self._read_frame(index)
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self)
            step = index.step or 1
            return [self._read_frame(i) for i in range(start, stop, step)]
        elif isinstance(index, (list, np.ndarray)):
            return [self._read_frame(i) for i in index]
        else:
            raise TypeError("Invalid index {} with type {}!".format(index,type(index)))

    def __del__(self):
        self.close()

    def __iter__(self):
        """ Iterate over trajectory frames. """
        self._reopen()
        return self

    def _read_first_time(self) -> None:
        """
        Check the information of the first frame and initialize the frame class
        """
        frame = self.frame
        content = []

        with open(self.filename,'r') as file:
            trigger = False
            while True:
                line = file.readline()
                if line.startswith("ITEM: TIMESTEP") and trigger:
                    break
                elif line.startswith("ITEM: TIMESTEP") and not trigger:
                    trigger = True
                    content.append(line)
                else:
                    content.append(line)
        
        # parse the frame information
        attributes = content[8].strip().split()[2:]
        self.attributes = attributes
        self.attrs_to_cols = {x: i for i, x in enumerate(attributes)}
        self._has_unwrapped_positions = False
        self._has_wrapped_positions = False
        self._has_images = False
        self._has_atom_types = False
        self._has_mol_ids = False
        self._has_velocities = False
        self._has_forces = False
        self._has_charges = False
        self._has_atom_ids = False
        self._has_masses = False
        self._has_elements = False

        if "xu" in attributes and "yu" in attributes and "zu" in attributes:
            self._has_unwrapped_positions = True
            self.attrs_to_cols_unwrapped_positions = [self.attrs_to_cols["xu"], self.attrs_to_cols["yu"], self.attrs_to_cols["zu"]]
        if "x" in attributes and "y" in attributes and "z" in attributes:
            self._has_wrapped_positions = True
            self.attrs_to_cols_wrapped_positions = [self.attrs_to_cols["x"], self.attrs_to_cols["y"], self.attrs_to_cols["z"]]
        if "ix" in attributes and "iy" in attributes and "iz" in attributes:
            self._has_images = True
            self.attrs_to_cols_images = [self.attrs_to_cols["ix"], self.attrs_to_cols["iy"], self.attrs_to_cols["iz"]]
        if "type" in attributes:
            self._has_atom_types = True
        if "mol" in attributes:
            self._has_mol_ids = True
        if "vx" in attributes and "vy" in attributes and "vz" in attributes:
            self._has_velocities = True
            self.attrs_to_cols_velocities = [self.attrs_to_cols["vx"], self.attrs_to_cols["vy"], self.attrs_to_cols["vz"]]
        if "fx" in attributes and "fy" in attributes and "fz" in attributes:
            self._has_forces = True
            self.attrs_to_cols_forces = [self.attrs_to_cols["fx"], self.attrs_to_cols["fy"], self.attrs_to_cols["fz"]]
        if "q" in attributes:
            self._has_charges = True
            self.attrs_to_cols_charges = self.attrs_to_cols["q"]
        if "id" in attributes:
            self._has_atom_ids = True
        if "mass" in attributes:
            self._has_masses = True
        if "element" in attributes:
            self._has_elements = True


        print("Initializing LammpsReader class for file {}...".format(self.filename))

        if self._verbose:
            print("n_atoms: ", self.n_atoms, ", n_frames: ", self.n_frames, ", frame attributes: ", attributes)
            print("has_atom_ids: ", self._has_atom_ids, \
                  "\nhas_charges: ", self._has_charges, \
                  "\nhas_wrapped_positions:", self._has_wrapped_positions, \
                  "\nhas_unwrapped_positions:", self._has_unwrapped_positions, \
                  "\nhas_images:", self._has_images, \
                  "\nhas_atom_types:", self._has_atom_types, \
                  "\nhas_mol_ids:", self._has_mol_ids, \
                  "\nhas_velocities:", self._has_velocities, \
                  "\nhas_forces:", self._has_forces, 
                  "\nhas masses:", self._has_masses,\
                  "\nhas elements:", self._has_elements \
                 )
        
        self._has_initialized = True

def test(filename):
    reader = LammpsReader(filename,verbose=True)
    i = 0
    for frame in reader:
        if i < 1:
            print(frame.index, frame.timestep, frame.n_atoms, frame.box_boundaries)
            print("frame._has_atom_ids:", reader._has_atom_ids, ", frame._has_charges:", frame._has_charges, ", frame._has_wrapped_positions:", frame._has_wrapped_positions, ", frame._has_unwrapped_positions:", frame._has_unwrapped_positions, ", frame._has_images:", frame._has_images, ", frame._has_atom_types:", frame._has_atom_types, ", frame._has_mol_ids:", frame._has_mol_ids, ", frame._has_velocities:", frame._has_velocities, ", frame._has_forces:", frame._has_forces, ", frame._has_masses:", frame._has_masses)
            if frame._has_wrapped_positions:
                print("wrapped positions:", frame.wrapped_positions[:10,:])
            if frame._has_unwrapped_positions:
                print("unwrapped positions:", frame.unwrapped_positions[:10,:])
            if frame._has_images:
                print("image",frame.images)
            if frame._has_atom_types:
                print("atom types",frame.atom_types)
            if frame._has_mol_ids:
                print("mol ids",frame.mol_ids)
            if frame._has_velocities:
                print("velocities: ",frame.velocities[:10,:])
            if frame._has_forces:
                print("forces: ",frame.forces[:10,:])
            if frame._has_charges:
                print("charges",frame.charges)
            if frame._has_masses:
                print("masses",frame.masses)
            if frame._has_elements:
                print("elements",frame.elements)
        else:
            break
        i += 1

if __name__ == "__main__":
    file="/beagle3/gavoth/sijiachen/Project/RTIL_Water_Jesse/3.production/JP-2InterfaceS-1/prod1_trj.lammpstrj"
    test(file)