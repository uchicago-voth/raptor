/* -------------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Authors: Chris Knight 
             Derived from USER-MULTIPRO package
------------------------------------------------------------------------- */

#ifdef INTEGRATE_CLASS

//IntegrateStyle(mp_verlet_sci,MP_Verlet_SCI)

#else

#ifndef LMP_MP_VERLET_SCI_H
#define LMP_MP_VERLET_SCI_H

#include "integrate.h"

namespace LAMMPS_NS {

class MP_Verlet_SCI : public Integrate {
 public:
  MP_Verlet_SCI(class LAMMPS *, int narg, char **arg) : Integrate(lmp, narg, arg) {};
  ~MP_Verlet_SCI() {};
  void init() {};
  void setup() {};
  void setup_minimal(int) {};
  void run(int) {};
  void cleanup() {};

  /***********************************/
  
  int is_master;          // Partition 0
  int is_master2;         // Partition 1; defaults to 0 if 1-partition calculation
  int is_master3;         // Partition 2; defaults to 0 if less than 3-partition calculation

  MPI_Comm block;
  
  void comm_forward(); // Update coordinates on slave partitions
  
  // class FixEVB* fix_evb;
  
  /***********************************/
  
 private:
  int triclinic;                    // 0 if domain is orthog, 1 if triclinic
  int torqueflag,extraflag;

  void force_clear() {};
};

}

#endif
#endif
