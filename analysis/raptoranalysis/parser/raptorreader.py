# -*- encoding: utf-8 -*-
# python 3.10.12
'''
Filename         : RaptorReader.py
Description      : A post-processing code to read RAPTOR output file, the default name of this output file
                   is `evb.out`. Please refer to RAPTOR manual for more information regarding
                   MS-EVB calculation and the meaning of the output.
Time             : 2023/08/29 12:46:37
Last Modified    : 2023/08/31 00:43:20
Author           : Sijia Chen
Version          : 1.0
Email            : sijiachen@uchicago.edu
'''

import os
import numpy as np
import functools
from .raptorframe import RaptorFrame



class RaptorReader(object):
    """
    class RaptorReader for reading RAPTOR outputs. One should notice that if you turn on "output for reaction" in `in.evb`, then you should not expect that the timestep difference between every two neighboring frames is the same. When turning on "output for reaction", RAPTOR will write out a frame information when 1) it meets the "output frequency, 0 means obey lammps setting", 2) a proton hopping happens. 

    Parameters
    ----------
    filename : str
        Name of the RAPTOR output file
    outputfreq : int, optional, default=1
        The output frequency of RAPTOR. 
        In some cases, one only wants to read the frames with a specific output frequency (to calculate msd, for example)
        (the real output frequency of RAPTOR could be nonconsistent if you turn on `if output for reaction`), 
        then one can set this argument to the desired output frequency. 
        If not set, then the reader will read all the frames in the output file.

    """
    n_frames: int
    _frame = RaptorFrame

    def __init__(self, filename, outputfreq=1, **kwargs) -> None:

        # initialize filename and check sanity
        self.filename = filename
        self._output_freq = int(outputfreq)
        self._check_sanity_filename()

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
        self.frame= self._frame(self.n_complexes, **self._frame_kwargs)
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
    # def frame(self) -> RaptorFrame:
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
    def n_complexes(self) -> int:
        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith("COMPLEX_COUNT"):
                    n_complexes= int(line.split()[1])
                    break
        self._cached["n_complexes"] = n_complexes
        return n_complexes
    
    @functools.cached_property
    def n_frames(self):
        counter = 0
        offsets = []
        with open(self.filename, 'r') as f:
            line = True
            while line:
                pos=f.tell()
                line = f.readline()
                if line.startswith("TIMESTEP"):
                    if int(line.strip().split()[1]) % self._output_freq != 0:
                        continue
                    counter += 1
                    offsets.append(pos)
        self._offsets = offsets
        self._cached["n_frames"] = counter 
        return len(self._offsets)
    
    def __len__(self):
        return self.n_frames
    
    def next(self) -> RaptorFrame:
        try:
            frame = self._read_next_frame()
        except EOFError:
            self.rewind()
            raise StopIteration from None
        else:
            pass
        
        return frame


    def __next__(self) -> RaptorFrame:
        return self.next()

    def close(self):
        if hasattr(self, '_file'):
            self._file.close()

    def _read_next_frame(self) -> RaptorFrame:
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
        line=f.readline() 
        timestep = int(line.strip().split()[1])
        frame.timestep = timestep

        # read the complex_count line and check sanity
        line=f.readline()
        n_complexes = int(line.strip().split()[1])
        if n_complexes != self.n_complexes:
            raise ValueError("Inconsistent number of complexes between the default and the current frame {} at timestep {}!".format(frame.index, frame.timestep))
        
        # read the remaing lines of this frame
        lines = []
        line = True
        while line:
            # if reach the last frame
            if frame.index == len(self)-1:
                line = f.readline()
                # I found a new possible error: 
                # if the last frame we want is 1000, but the actual last frame is 1001, then the reader will possible read two frames into the "last frame"
                # to avoid this, we need to check if the line is empty
                # then we need to check if the line is the "END_OF_COMPLEX {n_complexes}" line
                # if is, then we need to break the loop
                if line: # The readline() method doesn't trigger the end-of-file condition. Instead, when data is exhausted, it returns an empty string.
                    if line.startswith("END_OF_COMPLEX {}".format(n_complexes)):
                        lines.append(line)
                        break
                    else:
                        lines.append(line)
            # if not reach the last frame
            elif line == True: # handle the first line of the frame
                line = f.readline()
                lines.append(line)
            else: # handle the remaining lines of the frame
                if line.startswith("***************************"): 
                    break
                elif f.tell() < self._offsets[frame.index+1] : 
                    line = f.readline()
                    lines.append(line)
                else:
                    break
        
        # parse the frame overall information, record the start line # and stop line # of each complex for next step, but do not the complex information at this time
        start_complex_line = -1
        start_end_complex_lines = []
        for i, line in enumerate(lines):
            if line.startswith("REACTION_CENTER_LOCATION"):
                center_info=[]
                for j in range(i+1, i+1+n_complexes):
                    center_info.append([int(x) for x in lines[j].strip().split()])
                frame.center_location = np.array(center_info)
            if line.startswith("ENERGY_SUMMARY"):
                energy_info=dict()
                for j in range(i+1, i+1+3):
                    if lines[j].startswith("ENE_ENVIRONMENT"):
                        energy_info["environment"] = float(lines[j].strip().split()[1])
                    elif lines[j].startswith("ENE_COMPLEX"):
                        energy_info["complex"] = float(lines[j].strip().split()[1])
                    elif lines[j].startswith("ENE_TOTAL"):
                        energy_info["total"] = float(lines[j].strip().split()[1])
                frame.energy = energy_info
            if line.startswith("DECOMPOSED_ENV_ENERGY"):
                if lines[i+1].startswith("ENVIRONMENT"):
                    decomposed_env_energy = dict()
                    keys = lines[i+1].strip().replace('[','').replace(']','').replace('|',' ').split()[1:]
                    values = [float(x) for x in lines[i+2].strip().split()]
                    for key, value in zip(keys, values):
                        decomposed_env_energy[key] = value
                    frame.decomposed_env_energy = decomposed_env_energy
                else:
                    raise ValueError("Expect decomposed environment energy component names after the line of \"DECOMPOSED_ENV_ENERGY\" in frame {} at timestep {}.".format(frame.index, frame.timestep))
            if line.startswith("LOOP_ALL_COMPLEX"):
                start_complex_line = i
            if line.startswith("START_OF_COMPLEX"):
                start_end_complex_lines.append(i)
            if line.startswith("END_OF_COMPLEX"):
                start_end_complex_lines.append(i)
        
        # handle possible errors
        if start_complex_line == -1:
            raise ValueError("Cannot find the line of \"LOOP_ALL_COMPLEX\" in frame {} at timestep {}.".format(frame.index, frame.timestep))
        if len(start_end_complex_lines) != 2*n_complexes:
            print(start_end_complex_lines)
            raise ValueError("Cannot find the start and end line of each complex in frame {} at timestep {}.".format(frame.index, frame.timestep))
        
        # reshape the start_end_complex_lines for better understanding
        start_end_complex_lines = np.array(start_end_complex_lines).reshape(n_complexes, 2)
        # initialize the complexes list
        frame.complexes = []
        # parse the complex information
        for start, end in start_end_complex_lines:
            complex_info = {"id": None, "n_states": None, "states": None, "decomposed_complex_energy_diagonal": None, "decomposed_complex_energy_offdiagonal": None, "extra_coupling": None,"eigen_vector": None, "cec_coordinate": None,"cec_v2_coordinate":None,"next_pivot_state":None}
            complex_info["id"] = int(lines[start].strip().split()[1])
            for i in range(start+2, end):
                if lines[i].startswith("COMPLEX"):
                    complex_info["n_states"] = int(lines[i].strip().split()[2])
                    n_states = complex_info["n_states"]
                elif lines[i].startswith("STATE"):
                    complex_info["states"] = []
                    states_keys=lines[i].strip().replace('[','').replace(']','').replace('|',' ').split()[1:]
                    states_values=[[int(x) for x in lines[i+k].strip().split()] for k in range(1, n_states+1)]
                    for state_value in states_values:
                        complex_info["states"].append(dict(zip(states_keys, state_value)))
                elif lines[i].startswith("DECOMPOSED_COMPLEX_ENERGY"):
                    complex_info["decomposed_complex_energy_diagonal"] = []
                    complex_info["decomposed_complex_energy_offdiagonal"] = []
                elif lines[i].startswith("DIAGONAL ["): # must have " [" to avoid the "DIAGONALIZATION" line
                    diagonal_keys = lines[i].strip().replace('[','').replace(']','').replace('|',' ').split()[1:]
                    diagonal_values = [[float(x) for x in lines[i+k].strip().split()] for k in range(1, n_states+1)]
                    for diagonal_value in diagonal_values:
                        complex_info["decomposed_complex_energy_diagonal"].append(dict(zip(diagonal_keys, diagonal_value)))
                elif lines[i].startswith("OFF-DIAGONAL"):
                    offdiagonal_keys = lines[i].strip().replace('[','').replace(']','').replace('|',' ').split()[1:]
                    offdiagonal_values = [[float(x) for x in lines[i+k].strip().split()] for k in range(1, n_states)]
                    for offdiagonal_value in offdiagonal_values:
                        complex_info["decomposed_complex_energy_offdiagonal"].append(dict(zip(offdiagonal_keys, offdiagonal_value)))
                elif lines[i].startswith("EXTRA-COUPLING"):
                    complex_info["extra_coupling"] = []
                    n_extra_couplings = int(lines[i].strip().replace('[','').replace(']','').replace('|',' ').split()[1])
                    extra_coupling_keys = lines[i].strip().replace('[','').replace(']','').replace('|',' ').split()[2:]
                    extra_coupling_values = [[float(x) for x in lines[i+k].strip().split()] for k in range(1, 1+n_extra_couplings)]
                    for extra_coupling_value in extra_coupling_values:
                        complex_info["extra_coupling"].append(dict(zip(extra_coupling_keys, extra_coupling_value)))
                elif lines[i].startswith("EIGEN_VECTOR"):
                    complex_info["eigen_vector"] = [float(x) for x in lines[i+1].strip().split()]
                    if complex_info["n_states"] is not None and len(complex_info["eigen_vector"]) != n_states:
                        raise ValueError("Inconsistent number of eigen vectors with the number of states in frame {} at timestep {}.".format(frame.index, frame.timestep))
                elif lines[i].startswith("NEXT_PIVOT_STATE"):
                    complex_info["next_pivot_state"] = int(lines[i].strip().split()[1])
                elif lines[i].startswith("CEC_COORDINATE"):
                    complex_info["cec_coordinate"] = [float(x) for x in lines[i+1].strip().split()]
                elif lines[i].startswith("CEC_V2_COORDINATE"):
                    complex_info["cec_v2_coordinate"] = [float(x) for x in lines[i+1].strip().split()]
            frame.complexes.append(complex_info)


        return frame
        
        
    def rewind(self) -> RaptorFrame:
        self._reopen()
        self.next()

    def __del__(self):
        self.close()

    def __iter__(self):
        """ Iterate over trajectory frames. """
        self._reopen()
        return self
    
    def _read_frame(self, index) -> RaptorFrame:
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
    
    def __getitem__(self, index) -> RaptorFrame:
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
                if line.startswith("TIMESTEP") and trigger:
                    break
                elif line.startswith("TIMESTEP") and not trigger:
                    trigger = True
                    content.append(line)
                else:
                    content.append(line)
        
        # parse the frame information
        content = "".join(content)
        if "REACTION_CENTER_LOCATION" in content:
            frame._has_center_location = True
        if "ENERGY_SUMMARY" in content:
            frame._has_energy = True
        if ("DECOMPOSED_ENV_ENERGY" in content) and ("DECOMPOSED_COMPLEX_ENERGY" in content):
            frame._has_decomposed_energy = True
        if ("STATE_SEARCH" in content) and ("state(s) =" in content):
            frame._has_n_states = True
        if "STATES [ id | parent | shell | mol_A | mol_B | react | path | extra_cpl ]" in content:
            frame._has_states_info = True

        self._has_initialized = True

        print("Initialized RaptorReader class for file {}...".format(self.filename))
        print("n_complexes: ", self.n_complexes, ", n_frames: ", self.n_frames, ", has_center_location: ", self.frame._has_center_location, ", has_n_states: ", self.frame._has_n_states, ", has_states_info: ", self.frame._has_states_info, ", has_energy: ", self.frame._has_energy, ", has_decomposed_energy: ", self.frame._has_decomposed_energy)

def test(filename):
    reader = RaptorReader(filename)
    for frame in reader:
        print("frame index: ", frame.index, "frame timestep: ",frame.timestep, "frame number of complexes: ", frame.n_complexes, "frame all complexes information: ", frame.complexes, "\nframe complex 1: ", frame.get_complex(0))
        # print(frame.index, frame.timestep, frame.n_complexes, frame.center_location, frame.energy, frame.decomposed_env_energy, frame.complexes, frame.get_complex(0))

if __name__ == "__main__":
    file="44_10frames.evb"
    test(file)