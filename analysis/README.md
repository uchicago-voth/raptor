# raptor_analysis_suite
Development of a tool box **`raptoranalysis`** to help analyze RAPTOR outputs.
The tool box is developed in python 3.10.12. Any python version newer than 3.8 (the tool box use a feature in the build-in module `functools` introduced in python 3.8) should work, however, it is not tested.

## Author
Sijia Chen (@Alexandrina-Chen)

## Dependencies
1. [numpy](https://numpy.org/install/)
2. [tidynamics](https://pypi.org/project/tidynamics/)

All the dependencies can be installed using pip or conda.

## Installation
1. Clone the repository to your local machine, suppose the path is `[parent directory]/raptor_analysis_suite`
2. Enter the directory
    ```bash
    cd [parent directory]/raptor_analysis_suite
    ```
3. Add the path to your environment variable so that python can find the module
    ```bash
    export PYTHONPATH=$PYTHONPATH:[parent directory]/raptor_analysis_suite
    ```
4. Use the module in your python script
    ```python
    import raptoranalysis as ra
    ```

## Supported Functions

**Please refer to the example scripts in the `examples` directory for more details.**

### 1. Read the RAPTOR output

### 2. Calculated the proton forward hopping index (J. Chem. Phys. 154, 194506 (2021); doi: 10.1063/5.0040758)

### 3. Calculated proton identity correlation

### 4. Calculated proton continuous identity correlation

### 5. Calculate RDF between CEC and other normal atoms

### 6. Calculate MSD of CECs, from both discrete displacement and continuous displacement

### 7. Calculate EVB matrix eigenvalues distribution