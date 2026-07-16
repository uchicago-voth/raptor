# RAPTOR package

RAPTOR (Rapid Approach for Proton Transport and Other Reactions) is the Voth
group's implementation of multistate empirical valence bond (MS-EVB, also
called MS-RMD, multiscale reactive molecular dynamics) simulations in LAMMPS.
It enables reactive events such as proton transport to be simulated with
classical force fields by diagonalizing an EVB Hamiltonian built from
multiple bonding topologies ("states") at every timestep.

Copyright by the Voth Group (University of Chicago) since 2009. See the
`ACKNOWLEGEMENT` file in this directory for the full list of authors and
contributors, and `MANUAL.pdf` in the repository root for the user manual.

> **Minimal package distribution:** In the `uchicago-voth/raptor`
> `develop-2025` branch, this directory is distributed as `EXTRA-RAPTOR` and
> should be copied to `src/RAPTOR` in the pinned LAMMPS source tree. Optional
> CMake and GPU support described below also requires changes to core LAMMPS
> files from the external integration fork; those files are intentionally not
> bundled in this repository. The traditional CPU make build is the
> self-contained installation path provided here.

## What the package provides

* `fix evb` - the main command. It hijacks the force computation, builds the
  EVB states around each reaction center (complex), computes the Hamiltonian
  matrix elements (diagonal terms, off-diagonal couplings, repulsive
  corrections), diagonalizes it, and combines the forces with the
  ground-state eigenvector. Configuration is read from a `.cfg`/`.evb`
  parameter file and a `.top` topology file:

  ```
  fix ID all evb <cfg_file> <output_file> <top_file> [partial_offload=yes|no] [kspace_cplx=yes|no]
  ```

* `kspace_style evb_pppm` - an EVB-aware PPPM solver used internally by
  `fix evb`. Users specify a standard `kspace_style pppm` in the input; the
  fix replaces it with the matching `evb_pppm` variant at setup. Do NOT use
  `kspace_style evb_pppm` directly in an input script.

* `evb_pppm/gpu` (in `src/GPU`, built when `_RAPTOR_GPU` is enabled) - GPU
  variant that offloads the environment charge spreading to the GPU. It is
  selected automatically when running with `-sf gpu` and `kspace_cplx=yes`.

* GPU-aware pair styles: the RAPTOR-modified `lj/charmm/coul/long/gpu` and
  `lj/cut/coul/long/gpu` in `src/GPU` support toggling GPU offload per EVB
  force pass, so the environment diagonal term can run on the GPU while the
  (much smaller) complex-state terms run on the CPU.

Required LAMMPS packages: MOLECULE and KSPACE (enforced by CMake).

## Building

CMake is the primary build system. From the LAMMPS top-level directory:

CPU-only build:

```bash
cmake -S cmake -B build-raptor \
      -D PKG_MOLECULE=on -D PKG_KSPACE=on -D PKG_RAPTOR=on
cmake --build build-raptor -j 8
```

Build with the GPU package, using the preset provided in
`cmake/presets/raptor_gpu.cmake` (enables KSPACE, MANYBODY, MOLECULE, RIGID,
RAPTOR, GPU, and the `_RAPTOR_GPU` compile definition):

```bash
cmake -S cmake -B build-raptor -C cmake/presets/raptor_gpu.cmake \
      -D GPU_API=cuda -D GPU_PREC=mixed -D GPU_ARCH=sm_80 \
      -D CUDA_MPS_SUPPORT=on
cmake --build build-raptor -j 8
```

Notes:

* `GPU_API` can be `cuda`, `opencl`, or `hip`; set `GPU_ARCH` to match your
  GPU (e.g. `sm_80` for Ampere, `sm_60` for Pascal).
* `GPU_PREC=mixed` (single-precision forces, double-precision accumulation)
  is the recommended setting; expect relative energy differences of order
  1e-4 vs. a CPU run from the different precision and Coulomb tabulation.
* `-D CUDA_MPS_SUPPORT=on` builds the GPU library so that multiple MPI ranks
  can share one GPU through the CUDA Multi-Process Service (see below).
* `-D STATE_DECOMP=on` enables the optional multi-partition state
  decomposition (run with `-partition`); off by default.
* The traditional make build also works:
  `cd src && make yes-molecule yes-kspace yes-raptor && make mpi`.

The executable is `build-raptor/lmp` (or `src/lmp_mpi` for make builds).

## Examples (`tests` in the minimal package repository)

* `single` - water box with one hydronium/chloride pair (256 waters); the
  smallest and fastest test. Includes `lmp.in` (CPU), `lmp_gpu.in` (GPU), and
  a Polaris submission script.
* `multi`  - water box with 16 H+/Cl- pairs, exercising the self-consistent
  iterative (SCI) multi-complex MS-RMD method.
* `ClC`    - ClC chloride channel protein solvated in water (~67k atoms);
  a realistic biomolecular benchmark. `in.clc` continues a short run from
  the provided `md.restart`.

All examples use the RMD/3.2 parameters for excess protons in water.

Run on the CPU:

```bash
cd tests/ClC
mpirun -np 4 /path/to/build-raptor/lmp -in in.clc
```

Run with the GPU package:

```bash
mpirun -np 4 /path/to/build-raptor/lmp -in in.clc -sf gpu -pk gpu 1 neigh no
```

The EVB energy decomposition is written to the fix's output file (`evb.out`);
compare `ENE_ENVIRONMENT`, `ENE_COMPLEX`, and the per-state `DIAGONAL` rows
against a CPU run to validate a GPU build.

## Running with the GPU package

Requirements and restrictions enforced by `fix evb`:

* Neighbor list builds must stay on the host: use `-pk gpu N neigh no`.
  RAPTOR splits and swaps the host neighbor and topology lists between its
  environment/complex force passes, which is incompatible with GPU-side
  neighboring. For the same reason `fix evb` disables the GPU package's
  deferred ("overlapped") topology builds at setup.
* The particle split must be 1.0 (the default), i.e. all pair computations
  assigned to the GPU: do not use `-pk gpu N split <1`.
* Only a single reaction complex is supported on the GPU (`ncomplex == 1`);
  SCI multi-complex runs (e.g. the `multi` example) and `pppm/gpu` in SCI
  simulations are not supported yet.

Options of `fix evb` relevant to GPU runs:

* `partial_offload=yes` (recommended, default in the examples): only the
  environment real-space diagonal term - by far the dominant cost - is
  computed on the GPU; the pivot and other state diagonals stay on the CPU.
  With `partial_offload=no` all diagonal passes run on the GPU.
* `kspace_cplx=no` (default): a single CPU `evb_pppm` handles all k-space
  terms; the GPU accelerates only the pair style ("pair/only" behavior).
* `kspace_cplx=yes`: a separate k-space solver is created for the complex
  charges and the environment charge spreading runs on the GPU via
  `evb_pppm/gpu`.

### Sharing one GPU among MPI ranks (CUDA MPS)

RAPTOR runs typically use several MPI ranks per GPU. Throughput improves
substantially with the CUDA Multi-Process Service. See
`tests/notes-mps.txt` for a step-by-step walkthrough;
in short:

```bash
export CUDA_MPS_PIPE_DIRECTORY=$PWD
export CUDA_MPS_LOG_DIRECTORY=$PWD
nvidia-smi -i <gpu_id> -c EXCLUSIVE_PROCESS   # requires root
nvidia-cuda-mps-control -d
mpirun -np 4 /path/to/lmp -in in.txt -sf gpu -pk gpu 1 neigh no
```

The build must have been configured with `-D CUDA_MPS_SUPPORT=on`.

## Developer notes

* `EVB_cracker.h` contains hand-copied "cracked" mirrors of several core
  LAMMPS class definitions (`KSpace`, `Neighbor`, `Integrate`, `Pair`, and
  some pair styles) with protected members made public. These mirrors MUST
  be kept in sync with the real headers: a stale mirror changes the class
  layout seen by some translation units and fails silently as memory
  corruption, not as a compile error. Whenever the develop branch is merged,
  re-diff every `#ifdef _CRACKER_*` block against its header.
* `fix evb` removes `fix gpu` from the post-force call list and fetches the
  GPU forces itself (`EVB_Engine::get_gpu_data()`) at the end of each
  offloaded pass, so force collection points differ from stock GPU runs.

## Contact

For questions about the method and parameters, contact the Voth group
(see `ACKNOWLEGEMENT`). The user manual is `MANUAL.pdf` in the repository
root.


## Reference

S. Kaiser, Z. Yue, Y. Peng, T. D. Nguyen, S. Chen, D. Teng, G. A. Voth,
Molecular Dynamics Simulation of Complex Reactivity with the Rapid Approach
for Proton Transport and Other Reactions (RAPTOR) Software Package,
J. Phys. Chem. B 2024, 128, 20, 4959-4974.
