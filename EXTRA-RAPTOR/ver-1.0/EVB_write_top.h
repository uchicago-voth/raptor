/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMMAND_CLASS

CommandStyle(write_top,EVBWriteTop)

#else

#ifndef EVB_WRITE_TOP_H
#define EVB_WRITE_TOP_H

#include "stdio.h"
#include "pointers.h"

namespace LAMMPS_NS {

class EVBWriteTop : protected Pointers {
 public:
  EVBWriteTop(class LAMMPS *);
  void command(int, char **);
  void write(char *);

 private:
  int me,nprocs;
  FILE *fp;

  void evb_top();
  class FixEVB * fix_evb;

};

}

#endif
#endif
