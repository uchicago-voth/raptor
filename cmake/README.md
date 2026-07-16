# RAPTOR CMake integration

This directory contains the minimal CPU-only integration needed for LAMMPS to
recognize RAPTOR as a CMake package. It does not vendor or replace any LAMMPS
source files.

The patch is pinned to LAMMPS development commit
`697545ba8b25df7b83482e936738c87f93d15255` (`LAMMPS_VERSION "30 Mar 2026"`).
It adds RAPTOR to LAMMPS's standard package list and declares its dependencies
on the MOLECULE and KSPACE packages. LAMMPS's existing CMake package scan then
discovers and compiles the RAPTOR sources and style headers automatically.

## Install and build

From separate RAPTOR and LAMMPS checkouts:

```bash
git -C /path/to/lammps checkout 697545ba8b25df7b83482e936738c87f93d15255
cp -R /path/to/raptor/EXTRA-RAPTOR /path/to/lammps/src/RAPTOR

git -C /path/to/lammps apply --check \
  /path/to/raptor/cmake/lammps-697545ba8b-raptor.patch
git -C /path/to/lammps apply \
  /path/to/raptor/cmake/lammps-697545ba8b-raptor.patch

cmake -S /path/to/lammps/cmake -B /path/to/lammps/build-raptor \
  -D PKG_MOLECULE=on \
  -D PKG_KSPACE=on \
  -D PKG_RAPTOR=on
cmake --build /path/to/lammps/build-raptor -j 8
```

The resulting executable's `-h` output should list `RAPTOR` among the installed
packages and `evb` among the available fix styles.

## Scope

This integration supports the CPU package only. It does not define
`_RAPTOR_GPU`, install `evb_pppm/gpu`, or modify the LAMMPS GPU or KOKKOS
packages.

To remove the CMake integration, reverse the patch and delete the copied
package directory:

```bash
git -C /path/to/lammps apply -R \
  /path/to/raptor/cmake/lammps-697545ba8b-raptor.patch
rm -rf /path/to/lammps/src/RAPTOR
```
