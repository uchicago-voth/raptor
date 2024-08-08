/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   modified for MS-EVB by Tianying Yan
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "EVB_kspace.h"
#include "error.h"
#include "comm.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

EVB_KSpace::EVB_KSpace(LAMMPS *lmp) : KSpace(lmp) 
{
  bEff = false;
  energy = 0.0;
  order = 5;
  gridflag = 0;
  gewaldflag = 0;
  slabflag = 0;
  slab_volfactor = 1;
  evb_engine = NULL;
}

/* ---------------------------------------------------------------------- */

void EVB_KSpace::init()
{
  if(!evb_engine)
  {
    char errline[255];
    sprintf(errline,"[EVB_KAPCE] Please don't use EVB KSpace in \"kspace\" command.");
    error->all(FLERR,errline);
  }
}

/* ---------------------------------------------------------------------- */

void EVB_KSpace::compute(int eflag, int vflag)
{

}

/* ---------------------------------------------------------------------- */
