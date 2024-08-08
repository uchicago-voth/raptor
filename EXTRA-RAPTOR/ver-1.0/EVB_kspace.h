/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

    Modified for MS-EVB by Tianying Yan
------------------------------------------------------------------------- */

#ifndef EVB_KSPACE_H
#define EVB_KSPACE_H

#include "pointers.h"
#include "kspace.h"

#include "EVB_pointers.h"

namespace LAMMPS_NS {

class EVB_KSpace : public KSpace {
 public:
  EVB_KSpace(class LAMMPS *);
  virtual ~EVB_KSpace() {}

  /**********************************************
   * Function EVB_KSpace::compute(int,int)
   * This function is disabled in MS-EVB module
   **********************************************/
  virtual void init();
  void compute(int,int);

  
  virtual void evb_setup() = 0;
  virtual void compute_env(int) = 0;
  virtual void compute_env_density(int) = 0;
  virtual void compute_cplx(int) = 0;
  virtual void compute_cplx_eff(int) {};
  virtual void compute_exch(int) = 0;
  virtual void compute_eff(int) = 0;

  virtual void sci_setup_iteration() = 0;
  virtual void sci_setup_init() = 0;
  virtual void sci_compute_env(int)  = 0;
  virtual void sci_compute_cplx(int) = 0;
  virtual void sci_compute_exch(int) = 0;
  virtual void sci_compute_eff(int)  = 0;
  virtual void sci_compute_eff_cplx(int) {};
  virtual void sci_compute_eff_mp(int) = 0;
  virtual void sci_compute_eff_cplx_mp(int) {};

  double env_energy;

  bool bEff;
  double A_Rq;
  int *is_exch_chg;

  double off_diag_energy;
  double off_diag_virial[6];
  
  class EVB_Engine* evb_engine;
  class EVB_Timer* evb_timer;
};

}

#endif
