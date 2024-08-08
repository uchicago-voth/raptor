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
   Package: MULTIPRO
   Purpose: Improve the parallel effeciency of PPPM in MD simulations
   Authors: Yuxing Peng and Chris Knight 
            Voth Group, Department of Chemistry, University of Chicago
            
   Note: This file is Modified from verlet.h by Yuxing
------------------------------------------------------------------------- */

#ifdef INTEGRATE_CLASS

//IntegrateStyle(mp_verlet,MP_Verlet)

#else

#ifndef LMP_MP_VERLET_H
#define LMP_MP_VERLET_H

#include "integrate.h"

namespace LAMMPS_NS {

class MP_Verlet : public Integrate {
 public:
  MP_Verlet(class LAMMPS *, int narg, char **arg) : Integrate(lmp, narg, arg) {};
  ~MP_Verlet() {};
  void init() {};
  void setup() {};
  void setup_minimal(int) {};
  void run(int) {};

  /***********************************/
  
  int is_master, iblock, rank_block,nprocs_per_block;
  MPI_Comm block;
  
  int nlocal, nlocal_block;
  
  int *qsize, *xsize, *qdisp, *xdisp;

  double **f_kspace;
  int max_nlocal;
  
  void comm_init() {};
  void comm_box() {};
  void comm_forward() {};
  void comm_backward() {};
  
  //class FixEVB* fix_evb;
  
  /***********************************/
  
 private:
  int triclinic;                    // 0 if domain is orthog, 1 if triclinic
  int torqueflag;                   // zero out array every step

  void force_clear() {};
};

}

#endif
#endif
