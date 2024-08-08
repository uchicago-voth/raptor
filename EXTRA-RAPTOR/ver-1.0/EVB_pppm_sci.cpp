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

/* ----------------------------------------------------------------------
   Contributing authors: Roy Pollock (LLNL), Paul Crozier (SNL)

   Splitted for MS-EVB by: Tianying, Chris and Yuxing
------------------------------------------------------------------------- */

#include "mpi.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "atom.h"
#include "comm.h"
#define _CRACKER_GRIDCOMM
#include "EVB_cracker.h"
#undef _CRACKER_GRIDCOMM
#include "neighbor.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "domain.h"
#include "math_const.h"
#include "fft3d_wrap.h"
#include "remap_wrap.h"
#include "memory.h"
#include "error.h"
#include "mp_verlet_sci.h"
#include "universe.h"

#include "EVB_pppm.h"
#include "EVB_engine.h"
#include "EVB_offdiag.h"
#include "EVB_complex.h"
#include "EVB_effpair.h"
#include "EVB_timer.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define MAXORDER 7
#define OFFSET 4096
#define SMALL 0.00001
#define LARGE 10000.0
#define EPS_HOC 1.0e-7

enum{REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

#define Q_ATOM 0
#define Q_EFFECTIVE 1

#define KSPACE_DEFAULT    0 // Hellman-Feynman forces for Ewald
#define PPPM_HF_FORCES    1 // Hellman-Feynman forces for PPPM
#define PPPM_ACC_FORCES   2 // Approximate (acc) forces for PPPM. 
#define PPPM_POLAR_FORCES 3 // ACC forces plus an additional polarization force on complex atoms for PPPM.

/* ----------------------------------------------------------------------
   SCI functions
------------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void EVB_PPPM::sci_compute_env(int vflag)
{
  TIMER_STAMP(EVB_PPPM, sci_compute_env);

  nlocal = atom->nlocal;
  
  q = atom->q;
  x = atom->x;
  f = atom->f;
  
  int* is_cplx_atom = evb_engine->complex_atom;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int has_cplx_atom = evb_engine->has_complex_atom;

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  // Calculate the ENV density map;
  FFT_SCALAR ***save_density = density_brick;
  density_brick = env_density_brick;
  clear_density();
  
  // Make the density all at once
  make_rho(); 
  
  for(int i=0; i<nlocal; ++i) if(is_cplx_atom[i]) map2density_one_subtract(i);
  
  density_brick = save_density;
  
  load_env_density();

  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  poisson_energy(vflag);
  
  qsqsum = evb_engine->qsqsum_sys;
  for(int i=0; i<evb_engine->ncomplex; i++) qsqsum -= evb_engine->all_complex[i]->qsqsum;
  evb_engine->qsqsum_env = qsqsum;
  
  reduce_ev(vflag,true);

  env_energy = energy;
  if (vflag) for (int i = 0; i < 6; i++) virial[i] = 0.0;

  // Environment contribution to dipole for slab correction
  if(slabflag) {
    double *q = atom->q;
    double **x = atom->x;
    
    double dipole = 0.0;
    double dipole_r2 = 0.0;
    for(int i=0; i<atom->nlocal; i++) if(!is_cplx_atom[i]) {
	dipole    += q[i] * x[i][2];
	dipole_r2 += q[i] * x[i][2] * x[i][2];
      }
    
    MPI_Allreduce(&dipole,    &dipole_env,    1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&dipole_r2, &dipole_r2_env, 1, MPI_DOUBLE, MPI_SUM, world);
  }

  // Safe place to allocate some SCI_MP specific memory for arrays used in sci_iteration.
  if(!do_sci_compute_cplx_other) {
    memory->create(do_sci_compute_cplx_other, evb_engine->ncomplex, "EVB_PPPM::do_sci_compute_cplx_other");
    memory->create(energy_sci_compute_cplx_other, evb_engine->ncomplex, "EVB_PPPM::energy_sci_compute_cplx_other");

    memory->create(do_sci_compute_cplx_self, evb_engine->ncomplex*MAX_STATE, "EVB_PPPM::do_sci_compute_cplx_self");
    memory->create(energy_sci_compute_cplx_self, evb_engine->ncomplex*MAX_STATE, "EVB_PPPM::energy_sci_compute_cplx_self");    
  }

  // Preparing to start iteration phase
  sci_first_iteration_test = 1;

  TIMER_CLICK(EVB_PPPM,sci_compute_env);
}

/* ---------------------------------------------------------------------- */

void EVB_PPPM::sci_setup_iteration()
{
  for(int i=0; i<evb_engine->ncomplex; i++) do_sci_compute_cplx_other[i] = 1;
  if(sci_first_iteration_test) {
     for(int i=0; i<evb_engine->ncomplex*MAX_STATE; i++) do_sci_compute_cplx_self[i] = 1;
     sci_first_iteration_test = 0;
  }
}

/* ---------------------------------------------------------------------- */

void EVB_PPPM::sci_compute_cplx(int vflag)
{
  TIMER_STAMP(EVB_PPPM,sci_compute_cplx);

  energy = 0.0;
  double energy_cplx = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;

  int *is_cplx_atom = evb_engine->complex_atom;
  int cplx_id = evb_engine->evb_complex->id;

  // Load all complexes
  energy = 0.0;
  clear_density();
  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i]) {
      if(is_cplx_atom[i] == cplx_id) map2density_one(i,Q_ATOM);
      else map2density_one(i,Q_EFFECTIVE);
    }

  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();
  poisson_energy(vflag);
  energy_cplx = energy;

  // Subtract energy from all other complexes
  if(do_sci_compute_cplx_other[cplx_id-1]) {
    energy = 0.0;
    clear_density();
    for(int i=0; i<nlocal; i++) if(is_cplx_atom[i] && is_cplx_atom[i] != cplx_id) map2density_one(i,Q_EFFECTIVE);
    
    cg->reverse_comm(this,REVERSE_RHO);
    brick2fft();
    poisson_energy(vflag);
    energy_sci_compute_cplx_other[cplx_id-1] = energy;
    do_sci_compute_cplx_other[cplx_id-1] = 0;
  } else energy = energy_sci_compute_cplx_other[cplx_id-1];

  energy_cplx -= energy;
  
  const int state = evb_engine->evb_complex->current_status;
  const int indx = (cplx_id-1)*MAX_STATE + state;

  // Subtract self energy
  if(do_sci_compute_cplx_self[indx]) {
    energy = 0.0;
    clear_density();
    for(int i=0; i<nlocal; i++) if(is_cplx_atom[i] == cplx_id) map2density_one(i,Q_ATOM);
    
    cg->reverse_comm(this,REVERSE_RHO);
    brick2fft();
    poisson_energy(vflag);
    energy_sci_compute_cplx_self[indx] = energy;
    do_sci_compute_cplx_self[indx] = 0;
  } else energy = energy_sci_compute_cplx_self[indx];

  energy_cplx -= energy;

  energy = energy_cplx * 0.5 * volume * qqrd2e; // Interaction energy with all other complexes

  if(slabflag) slabcorr_sci_cplx();

  TIMER_CLICK(EVB_PPPM,sci_compute_cplx)
}

/* ---------------------------------------------------------------------- */

void EVB_PPPM::sci_compute_exch(int vflag)
{
  off_diag_energy = 0.0;
  error->universe_all(FLERR,"Must use real space approx. for off-diagonals with SCI + PPPM");
}

/* ---------------------------------------------------------------------- */

void EVB_PPPM::sci_compute_eff(int vflag)
{
  TIMER_STAMP(EVB_PPPM, sci_compute_eff);

  int ncomplex = evb_engine->ncomplex;

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  // load full-grid density for all atoms

  load_env_density();
  
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int *cplx_list = evb_engine->evb_complex->cplx_list;
  int *is_cplx_atom = evb_engine->complex_atom;

  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i]) map2density_one(i);
  
  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  poisson(true,vflag);

  if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
  else cg->forward_comm(this,FORWARD_IK);

  int SCI_KSPACE_flag = evb_engine->SCI_KSPACE_flag;

  if(SCI_KSPACE_flag == KSPACE_DEFAULT) fieldforce();
  else if(SCI_KSPACE_flag == PPPM_HF_FORCES) fieldforce_env();
  else if(SCI_KSPACE_flag == PPPM_ACC_FORCES) fieldforce();
  else if(SCI_KSPACE_flag == PPPM_POLAR_FORCES) fieldforce();

  if(slabflag) slabcorr_sci_eff(); // The slab correction should just apply to diagonal charges when using real-space approximation for off-diagonals, right?

  TIMER_CLICK(EVB_PPPM,sci_compute_eff)
}

/* ---------------------------------------------------------------------- */

void EVB_PPPM::sci_compute_eff_cplx(int vflag)
{
  TIMER_STAMP(EVB_PPPM, sci_compute_eff_cplx);

  f = atom->f;
  q = atom->q;

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  int ncomplex = evb_engine->ncomplex;
  int *is_cplx_atom = evb_engine->complex_atom;

  // Loop over complexes, calculating force due to other complexes.
  for(int i=0; i<ncomplex; i++) {
    clear_density();
    for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] && is_cplx_atom[j] != i+1) map2density_one(j);
  
    cg->reverse_comm(this,REVERSE_RHO);
    brick2fft();

    poisson(true,vflag);

    if (differentiation_flag == 1) {
      cg->forward_comm(this,FORWARD_AD);
      for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == i+1) field2force_one_ad(j,false);
    } else {
      cg->forward_comm(this,FORWARD_IK);
      for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == i+1) field2force_one_ik(j,false);
    }
  }

  TIMER_CLICK(EVB_PPPM,sci_compute_eff_cplx)
}

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void EVB_PPPM::sci_compute_eff_cplx_mp(int vflag)
{
  TIMER_STAMP(EVB_PPPM, sci_compute_eff_cplx);

  f = atom->f;
  q = atom->q;

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  int ncomplex = evb_engine->ncomplex;
  int *is_cplx_atom = evb_engine->complex_atom;

  /*
    if nworlds >= 3*ncomplex+3, then distribute subsystem (cplx and env) and dim (x, y, and z) to partitions
    else if nworlds >= 2*ncomplex+3, then distribute subsystem (cplx and env) and dim (x+y and z) to partitions
    else if nworlds>= ncomplex+3, then distribute subsystem (cplx and env) to partitions
    else partition masters compute forces
   */

  if(universe->nworlds >= 3*ncomplex+3) {

    if(universe->iworld < 3) return;
    int part_indx = universe->iworld - 3; // First three busy with sci_compute_eff_mp().

    // 3 partitions assigned to each complex
    int nw = 3;
    int cplx_indx = part_indx / nw;
    int dim = part_indx % nw;

    if(cplx_indx >= ncomplex) return;

    clear_density();
    for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] && is_cplx_atom[j] != cplx_indx+1) map2density_one(j);

    cg->reverse_comm(this,REVERSE_RHO);
    brick2fft();

    poisson_mp(true,vflag,dim,nw);

    if(dim > 0) return; // Only master of complex computes forces

    if (differentiation_flag == 1) {
      cg->forward_comm(this,FORWARD_AD);
      for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == cplx_indx+1) field2force_one_ad(j,false);
    } else {
      cg->forward_comm(this,FORWARD_IK);
      for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == cplx_indx+1) field2force_one_ik(j,false);
    }

  } else if(universe->nworlds >= 2*ncomplex+3) {

    if(universe->iworld < 3) return;
    int part_indx = universe->iworld - 3; // First three busy with sci_compute_eff_mp().

    // 2 partitions assigned to each complex
    int nw = 2;
    int cplx_indx = part_indx / nw;
    int dim = part_indx % nw;

    if(cplx_indx >= ncomplex) return;

    clear_density();
    for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] && is_cplx_atom[j] != cplx_indx+1) map2density_one(j);

    cg->reverse_comm(this,REVERSE_RHO);
    brick2fft();

    poisson_mp(true,vflag,dim,nw);

    if(dim > 0) return; // Only master of complex computes forces

    if (differentiation_flag == 1) {
      cg->forward_comm(this,FORWARD_AD);
      for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == cplx_indx+1) field2force_one_ad(j,false);
    } else {
      cg->forward_comm(this,FORWARD_IK);
      for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == cplx_indx+1) field2force_one_ik(j,false);
    }

  } else if(universe->nworlds >= ncomplex+3) {

    if(universe->iworld < 3) return;
    int num_part_working = universe->nworlds - 3; // First three busy with sci_compute_eff_mp().
    int part_indx = universe->iworld - 3;
    
    // Loop over complexes, calculating force due to other complexes.
    for(int i=0; i<ncomplex; i++) {
      if(i % num_part_working != part_indx) continue; // Skip if partition is not master of complex
      
      clear_density();
      for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] && is_cplx_atom[j] != i+1) map2density_one(j);
      
      cg->reverse_comm(this,REVERSE_RHO);
      brick2fft();
      
      poisson(true,vflag);
      
      if (differentiation_flag == 1) {
	cg->forward_comm(this,FORWARD_AD);
	for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == i+1) field2force_one_ad(j,false);
      } else {
	cg->forward_comm(this,FORWARD_IK);
	for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == i+1) field2force_one_ik(j,false);
      }
    }

  } else {
    
    // Loop over complexes, calculating force due to other complexes.
    for(int i=0; i<ncomplex; i++) {
      if(evb_engine->lb_cplx_master[i] != universe->iworld) continue; // Skip if partition is not master of complex
      
      clear_density();
      for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] && is_cplx_atom[j] != i+1) map2density_one(j);
      
      cg->reverse_comm(this,REVERSE_RHO);
      brick2fft();
      
      poisson(true,vflag);
      
      if (differentiation_flag == 1) {
	cg->forward_comm(this,FORWARD_AD);
	for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == i+1) field2force_one_ad(j,false);
      } else {
	cg->forward_comm(this,FORWARD_IK);
	for(int j=0; j<nlocal; j++) if(is_cplx_atom[j] == i+1) field2force_one_ik(j,false);
      }
    }

    // Otherwise assign complexes to partitions in round-robin style starting with partition #4
  }

  TIMER_CLICK(EVB_PPPM,sci_compute_eff_cplx)
}

/* ---------------------------------------------------------------------- */

void EVB_PPPM::sci_compute_eff_mp(int vflag)
{
  // Only the first three partitions do work
  if(universe->iworld > 2) return;

  TIMER_STAMP(EVB_PPPM, sci_compute_eff);

  int ncomplex = evb_engine->ncomplex;

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  // load full-grid density for all atoms

  load_env_density();
  
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int *cplx_list = evb_engine->evb_complex->cplx_list;
  int *is_cplx_atom = evb_engine->complex_atom;

  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i]) map2density_one(i);
  
  cg->reverse_comm(this,REVERSE_RHO);
  brick2fft();

  int iw = universe->iworld;
  int nw = universe->nworlds;
  if(nw > 3) nw = 3;
  poisson_mp(true,vflag,iw,nw);

  // Only master partition calculates forces
  if(!evb_engine->mp_verlet_sci->is_master) return;

  if (differentiation_flag == 1) cg->forward_comm(this,FORWARD_AD);
  else cg->forward_comm(this,FORWARD_IK);

  int SCI_KSPACE_flag = evb_engine->SCI_KSPACE_flag;

  if(SCI_KSPACE_flag == PPPM_HF_FORCES) fieldforce_env();
  else fieldforce();

  if(slabflag) slabcorr_sci_eff(); // The slab correction should just apply to diagonal charges when using real-space approximation for off-diagonals, right?

  TIMER_CLICK(EVB_PPPM,sci_compute_eff)
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if 
   adequate empty space is left between repeating slabs (J. Chem. Phys. 
   111, 3155).  Slabs defined here to be parallel to the xy plane. 
------------------------------------------------------------------------- */

void EVB_PPPM::slabcorr_sci_cplx()
{
  // compute local cplx contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  int cplx_id = evb_engine->evb_complex->id;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int * cplx_list = evb_engine->evb_complex->cplx_list;
  int * is_cplx_atom = evb_engine->complex_atom;

  double dipole_cplx    = 0.0;
  double dipole_r2_cplx = 0.0;
  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i]) {
    dipole_cplx    += q[i] * x[i][2];
    dipole_r2_cplx += q[i] * x[i][2] * x[i][2];
  }
  
  // sum local contributions to get global dipole moment
  
  double dipole_all    = 0.0;
  double dipole_r2_all = 0.0;
  MPI_Allreduce(&dipole_cplx,    &dipole_all,    1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&dipole_r2_cplx, &dipole_r2_all, 1, MPI_DOUBLE, MPI_SUM, world);

  dipole_all    += dipole_env; // Total System Dipole
  dipole_r2_all += dipole_r2_env;

  // compute corrections

  const double e_slabcorr = MY_2PI * (dipole_all * dipole_all - qsum * dipole_r2_all - 
				      qsum * qsum * zprd * zprd / 12.0) / volume;
  
  energy += qqrd2e * e_slabcorr / comm->nprocs / evb_engine->ncomplex;
}

void EVB_PPPM::slabcorr_sci_eff()
{
  // compute local cplx contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  int cplx_id = evb_engine->evb_complex->id;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int * cplx_list = evb_engine->evb_complex->cplx_list;
  int * is_cplx_atom = evb_engine->complex_atom;

  double dipole_cplx    = 0.0;
  for(int i=0; i<nlocal; i++) if(is_cplx_atom[i]) dipole_cplx    += q[i] * x[i][2];

  // sum local contributions to get global dipole moment
  
  double dipole_all    = 0.0;
  MPI_Allreduce(&dipole_cplx,    &dipole_all,    1, MPI_DOUBLE, MPI_SUM, world);

  dipole_all += dipole_env; // Total System Dipole

  // add on force corrections

  double ffact = -4.0 * MY_PI * qqrd2e / volume;
  double **f = atom->f;

  for(int i=0; i<nlocal; i++) f[i][2] += ffact * q[i] * (dipole_all - qsum*x[i][2]);
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver 
------------------------------------------------------------------------- */

void EVB_PPPM::poisson_mp(int eflag, int vflag, int iw, int nw)
{
  // Only master partition evaluates ad differentiation
  // Master and two worker partitions evaluate ik differentiation
  if (differentiation_flag == 1) {
    if(!evb_engine->mp_verlet_sci->is_master) return;
    poisson_ad(vflag);
  }
  else poisson_ik_mp(vflag,iw,nw);
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
------------------------------------------------------------------------- */

void EVB_PPPM::poisson_ik_mp(int vflag, int iw, int nw)
{
  int i,j,k,n;
  double eng;

  MPI_Status status;
  MPI_Comm block = evb_engine->mp_verlet_sci->block;
  
  // nw should be 1, 2, or 3
  // iw should be 0, 1, or 2
  // master_part is rank of iw==0 partition in communicator block
  const int master_part = universe->iworld - iw;

  // transform charge density (r -> k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,1);

  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  // Assign partitions to do x, y, and z gradients
  int do_x = 0;
  int do_y = 0;
  int do_z = 0;

  if(nw == 1) {
    do_x = 1;
    do_y = 1;
    do_z = 1;

  } else if(nw == 2) {
    if(iw == 0) {
      do_x = 1;
      do_y = 1;
    } else do_z = 1;

  } else {
    if(iw == 0) do_x = 1;
    else if(iw == 1) do_y = 1;
    else if(iw == 2) do_z = 1;
  }

  // compute gradients of V(r) in each of 3 dims by transformimg -ik*V(k)
  // FFT leaves data in 3d brick decomposition
  // copy it into inner portion of vdx,vdy,vdz arrays

  // x direction gradient

  if(do_x) {
    n = 0;
    for (k = nzlo_fft; k <= nzhi_fft; k++)
      for (j = nylo_fft; j <= nyhi_fft; j++)
	for (i = nxlo_fft; i <= nxhi_fft; i++) {
	  work2[n] = fkx[i]*work1[n+1];
	  work2[n+1] = -fkx[i]*work1[n];
	  n += 2;
	}
    
    fft2->compute(work2,work2,-1);
    
    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
      for (j = nylo_in; j <= nyhi_in; j++)
	for (i = nxlo_in; i <= nxhi_in; i++) {
	  vdx_brick[k][j][i] = work2[n];
	  n += 2;
	}
  }

  // y direction gradient

  if(do_y) {
    n = 0;
    for (k = nzlo_fft; k <= nzhi_fft; k++)
      for (j = nylo_fft; j <= nyhi_fft; j++)
	for (i = nxlo_fft; i <= nxhi_fft; i++) {
	  work2[n] = fky[j]*work1[n+1];
	  work2[n+1] = -fky[j]*work1[n];
	  n += 2;
	}
    
    fft2->compute(work2,work2,-1);
  }
  
  // Send y-gradient to master if need be
  // If one or two partitions, then master already has y-component
  // If three partitions, then partition 3 sends to master
  if(nw > 2) {
    if(iw == 1) MPI_Send(&(work2[0]), 2*nfft_both, MPI_FFT_SCALAR, master_part, 0, block);
    else if(iw == 0) MPI_Recv(&(work2[0]), 2*nfft_both, MPI_FFT_SCALAR, master_part+1, 0, block, &status);
  }

  // Master partition accumulates vdy_brick
  if(iw == 0) {
    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
      for (j = nylo_in; j <= nyhi_in; j++)
	for (i = nxlo_in; i <= nxhi_in; i++) {
	  vdy_brick[k][j][i] = work2[n];
	  n += 2;
	}
  }
  
  // z direction gradient

  if(do_z) {
    n = 0;
    for (k = nzlo_fft; k <= nzhi_fft; k++)
      for (j = nylo_fft; j <= nyhi_fft; j++)
	for (i = nxlo_fft; i <= nxhi_fft; i++) {
	  work2[n] = fkz[k]*work1[n+1];
	  work2[n+1] = -fkz[k]*work1[n];
	  n += 2;
	}
    
    fft2->compute(work2,work2,-1);
  }

  // Send z-gradient to master if need be
  // If one partition, master already has z-component
  // If two partitions, partition 1 sends to master
  // If three partitions, partition 2 sends to master
  if(nw > 2) {
    if(iw == 2) MPI_Send(&(work2[0]), 2*nfft_both, MPI_FFT_SCALAR, master_part, 0, block);
    else if(iw == 0) MPI_Recv(&(work2[0]), 2*nfft_both, MPI_FFT_SCALAR, master_part+2, 0, block, &status);
  } else if(nw == 2) {
    if(iw == 1) MPI_Send(&(work2[0]), 2*nfft_both, MPI_FFT_SCALAR, master_part, 0, block);
    else if(iw == 0) MPI_Recv(&(work2[0]), 2*nfft_both, MPI_FFT_SCALAR, master_part+1, 0, block, &status);
  }

  // Master partition accumulates vdz_brick
  if(iw == 0) {
    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
      for (j = nylo_in; j <= nyhi_in; j++)
	for (i = nxlo_in; i <= nxhi_in; i++) {
	  vdz_brick[k][j][i] = work2[n];
	  n += 2;
	}
  }

}
