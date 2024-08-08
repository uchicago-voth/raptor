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

#include "lmptype.h"
#include "mpi.h"
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "atom.h"
#include "comm.h"
#include "gridcomm.h"
//#define _CRACKER_GRIDCOMM
//#include "EVB_cracker.h"
//#undef _CRACKER_GRIDCOMM
#include "neighbor.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "domain.h"
#include "fft3d_wrap.h"
#include "remap_wrap.h"
#include "memory.h"
#include "error.h"
#include "timer.h"

#include "EVB_pppm.h"
#include "EVB_engine.h"
#include "EVB_effpair.h"
#include "EVB_offdiag.h"
#include "EVB_complex.h"
#include "EVB_timer.h"

#include "math_const.h"
#include "math_special.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

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

/*************************************************************************/
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/

void EVB_PPPM::evb_setup()
{
  nlocal = atom->nlocal;

  // extend size of per-atom arrays if necessary

  if (nlocal > nmax) {
    memory->destroy(part2grid);
    memory->destroy(part2grid_dr);
    nmax = atom->nmax;
    memory->create(part2grid,nmax,3,"EVB_PPPM:part2grid");
    memory->create(part2grid_dr,nmax,3,"EVB_PPPM:part2grid_dr");
  }
}

/*************************************************************************/

void EVB_PPPM::clear_density()
{
  FFT_SCALAR *vec = &density_brick[nzlo_out][nylo_out][nxlo_out];
  memset(vec, ZEROF, sizeof(FFT_SCALAR)*ngrid);
}

/*************************************************************************/

void EVB_PPPM::load_env_density()
{
  memcpy(&density_brick[nzlo_out][nylo_out][nxlo_out],
         &env_density_brick[nzlo_out][nylo_out][nxlo_out],
		 sizeof(FFT_SCALAR)*ngrid);
}

/*************************************************************************/

void EVB_PPPM::map2density_one(int id)
{
  double *q = atom->q;

  nx = part2grid[id][0];
  ny = part2grid[id][1];
  nz = part2grid[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d(part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt   

  z0 = delvolinv * q[id];
  for (int n = nlower; n <= nupper; n++) {
    mz = n+nz;
    y0 = z0*rho1d[2][n];
    
    for (int m = nlower; m <= nupper; m++) {
      my = m+ny;
      x0 = y0*rho1d[1][m];
      
      for (int l = nlower; l <= nupper; l++) {
	mx = l+nx;
	density_brick[mz][my][mx] += x0*rho1d[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
}

/*************************************************************************/

void EVB_PPPM::map2density_one(int id, int WHICH)
{
  double * q;
  if(WHICH == Q_ATOM) q = atom->q;
  else if(WHICH == Q_EFFECTIVE) q = evb_engine->evb_effpair->q;
  
  nx = part2grid[id][0];
  ny = part2grid[id][1];
  nz = part2grid[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d(part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt   

  z0 = delvolinv * q[id];
  for (int n = nlower; n <= nupper; n++) {
    mz = n+nz;
    y0 = z0*rho1d[2][n];
    
    for (int m = nlower; m <= nupper; m++) {
      my = m+ny;
      x0 = y0*rho1d[1][m];
      
      for (int l = nlower; l <= nupper; l++) {
	mx = l+nx;
	density_brick[mz][my][mx] += x0*rho1d[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
}

/*************************************************************************/

void EVB_PPPM::map2density_one_subtract(int id)
{
  double *q = atom->q;

  // Subtracts the contrinution for a given id
  nx = part2grid[id][0];
  ny = part2grid[id][1];
  nz = part2grid[id][2];
  
  // (dx,dy,dz) = distance to "lower left" grid pt
  
  compute_rho1d(part2grid_dr[id][0], part2grid_dr[id][1], part2grid_dr[id][2]);
  
  // (mx,my,mz) = global coords of moving stencil pt
  
  const FFT_SCALAR z0 = delvolinv * q[id];
  for (int n = nlower; n <= nupper; n++) {
    const FFT_SCALAR y0 = z0*rho1d[2][n];
    for (int m = nlower; m <= nupper; m++) {
      const FFT_SCALAR x0 = y0*rho1d[1][m];
      for (int l = nlower; l <= nupper; l++) {
	density_brick[n+nz][m+ny][l+nx] -= x0*rho1d[0][l];
      } // Loop mx
    } // Loop my
  } // Loop mz
}

/*************************************************************************/

void EVB_PPPM::poisson_energy(int vflag)
{
  int n;
  double eng;

  // transform charge density (r -> k) 

  n = 0;
  for (int i=0; i<nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
    
  }
  
  fft1->compute(work1,work1,1);
  
  // if requested, compute energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  n = 0;
  if (vflag) {
    for (int i=0; i<nfft; ++i) {
      eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
      for (int j=0; j<6; ++j) virial[j] += eng*vg[i][j];
      energy += eng;
      n += 2;
    }
  } else {
    for (int i=0; i<nfft; ++i) {
      eng = greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
      energy += eng;
      n += 2;
    }
    energy *= s2;
  }

}

/*************************************************************************/

void EVB_PPPM::field2force_one_ik(int id, bool Aflag)
{  
  nx = part2grid[id][0];
  ny = part2grid[id][1];
  nz = part2grid[id][2];
  
  compute_rho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
  
  ekx = eky = ekz = ZEROF;
  for (int n = nlower; n <= nupper; n++) {
    mz = n+nz;
    z0 = rho1d[2][n];
    for (int m = nlower; m <= nupper; m++) {
      my = m+ny;
      y0 = z0*rho1d[1][m];
      for (int l = nlower; l <= nupper; l++) {
	mx = l+nx;
	x0 = y0*rho1d[0][l];
	ekx -= x0*vdx_brick[mz][my][mx];;
	eky -= x0*vdy_brick[mz][my][mx];;
	ekz -= x0*vdz_brick[mz][my][mx];;
      }
    }
  }
  
  // convert E-field to force
  double qfactor = qqrd2e * scale * q[id];
  if(Aflag) qfactor *= A_Rq;
  
  f[id][0] += qfactor * ekx;
  f[id][1] += qfactor * eky;
  if (slabflag != 2) f[id][2] += qfactor * ekz;
}

/*************************************************************************/

void EVB_PPPM::field2force_one_ad(int id, bool Aflag)
{
  double s1, s2, s3;
  double sf = 0.0;
  double *prd;

  if(triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;
  
  const double hx_inv = nx_pppm / prd[0];
  const double hy_inv = ny_pppm / prd[1];
  const double hz_inv = nz_pppm / prd[2];

  nx = part2grid[id][0];
  ny = part2grid[id][1];
  nz = part2grid[id][2];
  
  compute_rho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
  compute_drho1d(part2grid_dr[id][0],part2grid_dr[id][1],part2grid_dr[id][2]);
  
  ekx = eky = ekz = ZEROF;
  for (int n = nlower; n <= nupper; n++) {
    mz = n+nz;
    for (int m = nlower; m <= nupper; m++) {
      my = m+ny;
      for (int l = nlower; l <= nupper; l++) {
	mx = l+nx;
	ekx += drho1d[0][l] *  rho1d[1][m] *  rho1d[2][n] * u_brick[mz][my][mx];
	eky +=  rho1d[0][l] * drho1d[1][m] *  rho1d[2][n] * u_brick[mz][my][mx];
	ekz +=  rho1d[0][l] *  rho1d[1][m] * drho1d[2][n] * u_brick[mz][my][mx];
      }
    }
  }
  ekx *= hx_inv;
  eky *= hy_inv;
  ekz *= hz_inv;
  
  // convert E-field to force and subtract self forces
  double qfactor = qqrd2e * scale;
  if(Aflag) qfactor *= A_Rq;

  s1 = x[id][0] * hx_inv;
  s2 = x[id][1] * hy_inv;
  s3 = x[id][2] * hz_inv;
  sf = sf_coeff[0] * sin(2 * MY_PI * s1);
  sf += sf_coeff[1] * sin(4 * MY_PI * s1);
  sf *= 2 * q[id] * q[id];
  f[id][0] += qfactor * (ekx * q[id] - sf);

  sf = sf_coeff[2] * sin(2 * MY_PI * s2);
  sf += sf_coeff[3] * sin(4 * MY_PI * s2);
  sf *= 2 * q[id] * q[id];
  f[id][1] += qfactor * (eky * q[id] - sf);

  if (slabflag != 2) {
    sf = sf_coeff[4] * sin(2 * MY_PI * s3);
    sf += sf_coeff[5] * sin(4 * MY_PI * s3);
    sf *= 2 * q[id] * q[id];
    f[id][2] += qfactor * (ekz * q[id] - sf);
  }
}

/*************************************************************************/

void EVB_PPPM::reduce_ev(int vflag, bool Aflag)
{
  energy *= 0.5*volume;
  if(comm->me==0) energy -= g_ewald*qsqsum/MY_PIS + MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
  
  energy *= qqrd2e;
  
  // sum virial across procs

  if (vflag) {
    double virial_all[6];
    MPI_Allreduce(virial,virial_all,6,MPI_DOUBLE,MPI_SUM,world);
    double pre_factor = 0.5*qqrd2e*volume;
    if(Aflag) pre_factor *= A_Rq;
    for (int i=0; i<6; i++) virial[i] = pre_factor*virial_all[i];
  }
}

/*************************************************************************/

void EVB_PPPM::compute_env(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_env);
  
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

  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }
  
  // Calculate the ENV density map;
  FFT_SCALAR ***save_density = density_brick;
  density_brick = env_density_brick;
  clear_density();  

  // Make the density all at once
  make_rho(); 
  
  if (has_cplx_atom) for(int i=0; i<nlocal_cplx; ++i) map2density_one_subtract(cplx_list[i]);

  
  density_brick = save_density;  
  load_env_density();
  
  cg->reverse_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                   REVERSE_RHO,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
  
  brick2fft();

  poisson_energy(vflag);
  
  qsqsum = evb_engine->qsqsum_env = evb_engine->qsqsum_sys-evb_engine->evb_complex->qsqsum;
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
  
  if (triclinic) domain->lamda2x(atom->nlocal);
  
  TIMER_CLICK(EVB_PPPM, compute_env); 
}

/*************************************************************************/

void EVB_PPPM::compute_env_density(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_env);
  
  nlocal = atom->nlocal;

  q = atom->q;
  x = atom->x;
  
  int* is_cplx_atom = evb_engine->complex_atom;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int has_cplx_atom = evb_engine->has_complex_atom;
  
  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;

  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }
  // Calculate the ENV density map;
  FFT_SCALAR ***save_density = density_brick;
  density_brick = env_density_brick;
  clear_density();

  // Make the density all at once
  make_rho(); 
  
  if (has_cplx_atom) for(int i=0; i<nlocal_cplx; ++i) map2density_one_subtract(cplx_list[i]);
  
  density_brick = save_density;
  load_env_density();

  qsqsum = evb_engine->qsqsum_env = 0.0;
  env_energy = energy;

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
  
  if (triclinic) domain->lamda2x(atom->nlocal);
  TIMER_CLICK(EVB_PPPM, compute_env); 
}

/*************************************************************************/

void EVB_PPPM::compute_cplx(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_cplx);

  f = atom->f;

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }
  
  load_env_density();
  
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  for(int i=0; i<nlocal_cplx; i++) map2density_one(cplx_list[i]);
  
  cg->reverse_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                   REVERSE_RHO,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
  brick2fft();
 
  // Don't calculate forces here in SCI simulations during compute(), only initialize().
  if(evb_engine->ncomplex == 1) {
    poisson(true,vflag);

    if (differentiation_flag == 1) cg->forward_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                     FORWARD_AD,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
    else cg->forward_comm(GridComm::KSPACE,this,3,sizeof(FFT_SCALAR),
                     FORWARD_IK,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
    fieldforce();

  } else if( (evb_engine->SCI_KSPACE_flag == KSPACE_DEFAULT || evb_engine->SCI_KSPACE_flag == PPPM_HF_FORCES) &&
	     evb_engine->engine_indicator == ENGINE_INDICATOR_INITIALIZE) {
    poisson(true,vflag);
    
    if (differentiation_flag == 1) {
      cg->forward_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                     FORWARD_AD,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
      for(int i=0; i<nlocal_cplx; i++) field2force_one_ad(cplx_list[i],false);
    } else {
      cg->forward_comm(GridComm::KSPACE,this,3,sizeof(FFT_SCALAR),
                     FORWARD_IK,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
      for(int i=0; i<nlocal_cplx; i++) field2force_one_ik(cplx_list[i],false);
    }

  } else poisson_energy(vflag);

  qsqsum = evb_engine->qsqsum_env + evb_engine->evb_complex->qsqsum;
  reduce_ev(vflag,true);

  // Don't calculate slab correction here in SCI simulations
  if(slabflag && evb_engine->ncomplex == 1) slabcorr_cplx();

  energy -= env_energy;
  
  if (triclinic) domain->lamda2x(atom->nlocal);
  TIMER_CLICK(EVB_PPPM, compute_cplx);
}

/*************************************************************************/

void EVB_PPPM::compute_cplx_eff(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_cplx_eff);

  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  f = atom->f;
  
  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }
  
  int *is_cplx_atom = evb_engine->complex_atom;
  int cplx_id = evb_engine->evb_complex->id;  

  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;

  clear_density();
  for(int i=0; i<nlocal; i++) {
      if(is_cplx_atom[i] == cplx_id) map2density_one(i,Q_ATOM);
      else map2density_one(i,Q_EFFECTIVE);
    }
  
  cg->reverse_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                   REVERSE_RHO,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
  brick2fft();
 
  poisson(true,vflag);

  // Only accumulate forces on complex atoms
  if (differentiation_flag == 1) {
    cg->forward_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                     FORWARD_AD,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
    for(int i=0; i<nlocal_cplx; i++) field2force_one_ad(cplx_list[i],false);
  } else {
    cg->forward_comm(GridComm::KSPACE,this,3,sizeof(FFT_SCALAR),
                     FORWARD_IK,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
    for(int i=0; i<nlocal_cplx; i++) field2force_one_ik(cplx_list[i],false);
  }

 
  if (triclinic) domain->lamda2x(atom->nlocal);
  TIMER_CLICK(EVB_PPPM, compute_cplx_eff);
}

/*************************************************************************/

void EVB_PPPM::compute_exch(int vflag)
{
  TIMER_STAMP(EVB_PPPM, compute_exch);  

  double qsum_save = qsum;
 
  f = atom->f;
  double save_energy = energy;
  double save_virial[6];
  memcpy(save_virial,virial,sizeof(double)*6);
  
  /***************************************************/
  /******* Overall                             *******/
  /***************************************************/
  
  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  // convert atoms from box to lamda coords

  if (triclinic == 0) boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }
  
  load_env_density();

  int* is_cplx_atom = evb_engine->complex_atom;
  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;
  for(int i=0; i<nlocal_cplx; i++) map2density_one(cplx_list[i]);
  
  cg->reverse_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                   REVERSE_RHO,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
  brick2fft();
  poisson_energy(vflag);

  if (differentiation_flag == 1) cg->forward_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                     FORWARD_AD,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
  else cg->forward_comm(GridComm::KSPACE,this,3,sizeof(FFT_SCALAR),
                     FORWARD_IK,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
  
  qsum = qsqsum = 0.0;
  reduce_ev(vflag,true);
  
  off_diag_energy = energy;
  if (vflag) memcpy(off_diag_virial,virial,sizeof(double)*6);
 
  /***************************************************/
  /******* Mesh(Exch_chg)->Point(Non_exch_chg) *******/
  /***************************************************/
  
  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  clear_density();

  for (int i=0; i<nlocal; i++) if(is_exch_chg[i]) map2density_one(i);
  
  cg->reverse_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                   REVERSE_RHO,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
  brick2fft();
  poisson(true, vflag);

  if (differentiation_flag == 1) {
    cg->forward_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                     FORWARD_AD,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
    for (int i=0; i<nlocal; i++) if(!is_exch_chg[i]) field2force_one_ad(i,true);
  } else {
    cg->forward_comm(GridComm::KSPACE,this,3,sizeof(FFT_SCALAR),
                     FORWARD_IK,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
    for (int i=0; i<nlocal; i++) if(!is_exch_chg[i]) field2force_one_ik(i,true);
  }
  
  reduce_ev(vflag,true);

  off_diag_energy -= energy;
  if (vflag) for(int i=0; i<6; i++) off_diag_virial[i]-=virial[i];

  /***************************************************/
  /******* Mesh(Non_exch_chg)->Point(Exch_chg) *******/
  /***************************************************/
  
  energy = 0.0;
  if (vflag) for (int i=0; i<6; i++) virial[i] = 0.0;
  
  load_env_density();
  
  for (int i=0; i<nlocal; i++) if(is_cplx_atom[i] && !is_exch_chg[i]) map2density_one(i);
  
  cg->reverse_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                   REVERSE_RHO,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
  brick2fft();
  poisson(true, vflag);

  if (differentiation_flag == 1) {
    cg->forward_comm(GridComm::KSPACE,this,1,sizeof(FFT_SCALAR),
                     FORWARD_AD,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
    for (int i=0; i<nlocal; i++) if(is_exch_chg[i]) field2force_one_ad(i,true);
  } else {
    cg->forward_comm(GridComm::KSPACE,this,3,sizeof(FFT_SCALAR),
                     FORWARD_IK,cg_buf1,cg_buf2,MPI_FFT_SCALAR);
    for (int i=0; i<nlocal; i++) if(is_exch_chg[i]) field2force_one_ik(i,true);
  }
  
  reduce_ev(vflag,true);
  
  off_diag_energy -= energy;
  if (vflag) for(int i=0; i<6; i++) off_diag_virial[i]-=virial[i];

  /***************************************************************/
  /***************************************************************/
  
  // Don't calculate slab correction here in SCI simulations
  if(slabflag && evb_engine->ncomplex == 1) slabcorr_exch();

  double energy_all;
  MPI_Allreduce(&off_diag_energy,&energy_all,1,MPI_DOUBLE,MPI_SUM,world);
  off_diag_energy = energy_all;
  
  energy = save_energy;
  memcpy(virial, save_virial,sizeof(double)*6);
  
  qsum = qsum_save;
  
  if (triclinic) domain->lamda2x(atom->nlocal);
  TIMER_CLICK(EVB_PPPM, compute_exch); 
}

/*************************************************************************/

void EVB_PPPM::compute_eff(int vflag)
{

}

/*************************************************************************/
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/

/* ---------------------------------------------------------------------- */

EVB_PPPM::EVB_PPPM(LAMMPS *lmp) : EVB_KSpace(lmp)
{
  pppmflag = 1;
  group_group_enable = 0;
  triclinic = domain->triclinic;
  
  // AWGL : save pointer
  lmp_pointer = lmp;
  
  nfactors = 3;
  factors = new int[nfactors];
  factors[0] = 2;
  factors[1] = 3;
  factors[2] = 5;

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  density_brick = vdx_brick = vdy_brick = vdz_brick = NULL;
  density_fft = NULL;
  u_brick = NULL;
  greensfn = NULL;
  work1 = work2 = NULL;
  vg = NULL;
  fkx = fky = fkz = NULL;

  sf_precoeff1 = sf_precoeff2 = sf_precoeff3 = 
    sf_precoeff4 = sf_precoeff5 = sf_precoeff6 = NULL;

  gf_b = NULL;
  rho1d = rho_coeff = drho1d = drho_coeff = NULL;

  fft1 = fft2 = NULL;
  remap = NULL;
  cg = NULL;

  nmax = 0;
  part2grid = NULL;
  
  // define acons coefficients for estimation of kspace errors
  // see JCP 109, pg 7698 for derivation of coefficients
  // higher order coefficients may be computed if needed
  memory->create(acons,8,7,"EVB_PPPM:acons");

  acons[1][0] = 2.0 / 3.0;
  acons[2][0] = 1.0 / 50.0;
  acons[2][1] = 5.0 / 294.0;
  acons[3][0] = 1.0 / 588.0;
  acons[3][1] = 7.0 / 1440.0;
  acons[3][2] = 21.0 / 3872.0;
  acons[4][0] = 1.0 / 4320.0;
  acons[4][1] = 3.0 / 1936.0;
  acons[4][2] = 7601.0 / 2271360.0;
  acons[4][3] = 143.0 / 28800.0;
  acons[5][0] = 1.0 / 23232.0;
  acons[5][1] = 7601.0 / 13628160.0;
  acons[5][2] = 143.0 / 69120.0;
  acons[5][3] = 517231.0 / 106536960.0;
  acons[5][4] = 106640677.0 / 11737571328.0;
  acons[6][0] = 691.0 / 68140800.0;
  acons[6][1] = 13.0 / 57600.0;
  acons[6][2] = 47021.0 / 35512320.0;
  acons[6][3] = 9694607.0 / 2095994880.0;
  acons[6][4] = 733191589.0 / 59609088000.0;
  acons[6][5] = 326190917.0 / 11700633600.0;
  acons[7][0] = 1.0 / 345600.0;
  acons[7][1] = 3617.0 / 35512320.0;
  acons[7][2] = 745739.0 / 838397952.0;
  acons[7][3] = 56399353.0 / 12773376000.0;
  acons[7][4] = 25091609.0 / 1560084480.0;
  acons[7][5] = 1755948832039.0 / 36229939200000.0;
  acons[7][6] = 4887769399.0 / 37838389248.0;

  /***************************************/
  /***************************************/
  env_density_brick = NULL;
  part2grid_dr = NULL;
  /***************************************/
  /***************************************/

  // GridComm
  cg_buf1 = cg_buf2 = NULL;
  
  do_sci_compute_cplx_other = NULL;
  energy_sci_compute_cplx_other = NULL;

  do_sci_compute_cplx_self = NULL;
  energy_sci_compute_cplx_self = NULL;
}

void EVB_PPPM::settings(int narg, char **arg)
{
  if (narg < 1) error->all(FLERR,"Illegal kspace_style pppm command");
  accuracy_relative = fabs(utils::numeric(FLERR,arg[0],false,lmp));
}

/* ----------------------------------------------------------------------
   free all memory 
------------------------------------------------------------------------- */

EVB_PPPM::~EVB_PPPM()
{
  delete [] factors;
  deallocate();
  memory->destroy(part2grid);
  memory->destroy(acons);
  
  /***************************************/
  /***************************************/
  memory->destroy(part2grid_dr);
  /***************************************/
  /***************************************/

  memory->destroy(cg_buf1);
  memory->destroy(cg_buf2);

  memory->destroy(do_sci_compute_cplx_other);
  memory->destroy(energy_sci_compute_cplx_other);

  memory->destroy(do_sci_compute_cplx_self);
  memory->destroy(energy_sci_compute_cplx_self);
}

/* ----------------------------------------------------------------------
   called once before run 
------------------------------------------------------------------------- */

void EVB_PPPM::init()
{
  EVB_KSpace::init();  

  if (me == 0) {
    if (screen) fprintf(screen,"EVB_PPPM initialization ...\n");
    if (logfile) fprintf(logfile,"EVB_PPPM initialization ...\n");
  }

  // error check

  triclinic_check();

  if (triclinic != domain->triclinic)
    error->all(FLERR,"Must redefine kspace_style after changing to triclinic box");
  
  if (domain->triclinic && differentiation_flag == 1)
    error->all(FLERR,"Cannot (yet) use PPPM with triclinic box "
               "and kspace_modify diff ad");
  if (domain->triclinic && slabflag)
    error->all(FLERR,"Cannot (yet) use PPPM with triclinic box and "
               "slab correction");
  
  if (domain->dimension == 2) error->all(FLERR,"Cannot use EVB_PPPM with 2d simulation");

  if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute q");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use nonperiodic boundaries with EVB_PPPM");
  if (slabflag == 1) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 || 
	domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab EVB_PPPM");
  }

  if(comm->me == 0 && slabflag) {
    if(screen) fprintf(screen,"  Slab correction activated: volfactor = %f\n",slab_volfactor);
    if(logfile) fprintf(logfile,"  Slab correction activated: volfactor = %f\n",slab_volfactor);
  }

  if (order < 2 || order > MAXORDER) {
    char str[128];
    sprintf(str,"EVB_PPPM order cannot be < 2 or > than %d",MAXORDER);
    error->all(FLERR,str);
  }
  
  // compute two charge force
  two_charge();
  
  // extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;
  scale = 1.0;

  pair_check();

  qqrd2e = force->qqrd2e;
  
  int itmp;
  double *p_cutoff = (double *) force->pair->extract((char*)"cut_coul",itmp);
  if (p_cutoff == NULL)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  cutoff = *p_cutoff;

  // if kspace is TIP4P, extract TIP4P params from pair style
  // bond/angle are not yet init(), so insure equilibrium request is valid

  qdist = 0.0;

  if (tip4pflag) {
    double *p_qdist = (double *) force->pair->extract("qdist",itmp);
    int *p_typeO = (int *) force->pair->extract("typeO",itmp);
    int *p_typeH = (int *) force->pair->extract("typeH",itmp);
    int *p_typeA = (int *) force->pair->extract("typeA",itmp);
    int *p_typeB = (int *) force->pair->extract("typeB",itmp);
    if (!p_qdist || !p_typeO || !p_typeH || !p_typeA || !p_typeB)
      error->all(FLERR,"KSpace style is incompatible with Pair style");
    qdist = *p_qdist;
    typeO = *p_typeO;
    typeH = *p_typeH;
    int typeA = *p_typeA;
    typeB = *p_typeB;

    if (force->angle == NULL || force->bond == NULL)
      error->all(FLERR,"Bond and angle potentials must be defined for TIP4P");
    if (typeA < 1 || typeA > atom->nangletypes ||
	force->angle->setflag[typeA] == 0)
      error->all(FLERR,"Bad TIP4P angle type for PPPM/TIP4P");
    if (typeB < 1 || typeB > atom->nbondtypes ||
	force->bond->setflag[typeB] == 0)
      error->all(FLERR,"Bad TIP4P bond type for PPPM/TIP4P");
    double theta = force->angle->equilibrium_angle(typeA);
    double blen = force->bond->equilibrium_distance(typeB);
    alpha = qdist / (cos(0.5*theta) * blen);
  }

  // compute qsum & qsqsum and warn if not charge-neutral

  qsum = qsqsum = 0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    qsum += atom->q[i];
    qsqsum += atom->q[i]*atom->q[i];
  }

  // if kspace is CRHYDROXIDE, extract CR parameters from pair style
  if ((strcmp(force->kspace_style,"pppm/crhydroxide") == 0) ||
      (strcmp(force->kspace_style,"evb_pppm/crhydroxide") == 0)) {
    if (force->pair == NULL) error->all(FLERR,"KSpace style is incompatible with Pair style");
    double *p_qdist = (double *) force->pair->extract("qdist",itmp);
    int *p_typeO = (int *) force->pair->extract("typeO",itmp);
    int *p_typeH = (int *) force->pair->extract("typeH",itmp);
    int *p_typeB = (int *) force->pair->extract("typeB",itmp);
    if (!p_qdist || !p_typeO || !p_typeH || !p_typeB) error->all(FLERR,"KSpace style is incompatible with Pair style");
    qdist = *p_qdist;
    qdist = 0.0;
    typeO = *p_typeO;
    typeH = *p_typeH;
    typeB = *p_typeB;

    int *p_cr_N = (int *) force->pair->extract("cr_N",itmp);
    double *p_cr_height = (double *) force->pair->extract("cr_height",itmp);
    double *p_cr_diameter = (double *) force->pair->extract("cr_diameter",itmp);

    cr_N        = *p_cr_N;
    cr_height   = *p_cr_height;
    cr_diameter = *p_cr_diameter;

    if (force->bond == NULL) error->all(FLERR,"Bond potentials must be defined for CR-HYDROXIDE");
    if (typeB < 1 || typeB > atom->nbondtypes || force->bond->setflag[typeB] == 0) error->all(FLERR,"Bad hydroxide bond type for PPPM/CRHYDROXIDE");

    // Adjust qsqsum for ring particles
    cr_qsqsum = 0.0;
    for(int i=0; i<atom->nlocal; i++) {
      if(atom->type[i] == typeO) {
	double qI = atom->q[i];
	cr_qsqsum -= qI*qI;
	qI /= double(cr_N);
	cr_qsqsum += cr_N * qI * qI;
      }
    }
    qsqsum += cr_qsqsum;
  }

  double tmp;
  MPI_Allreduce(&qsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum = tmp;
  MPI_Allreduce(&qsqsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsqsum = tmp;
  MPI_Allreduce(&cr_qsqsum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  cr_qsqsum = tmp;
  q2 = qsqsum * force->qqrd2e / force->dielectric;

  if (qsqsum == 0.0)
    error->all(FLERR,"Cannot use kspace solver on system with no charge");
  if (fabs(qsum) > SMALL && me == 0) {
    char str[128];
    sprintf(str,"System is not charge neutral, net charge = %g",qsum);
    error->warning(FLERR,str);
  }

  // set accuracy (force units) from accuracy_relative or accuracy_absolute
  
  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;
  
  if(comm->me == 0) {
    printf("accuracy_relative %lf two_charge_force %lf accuracy %lf\n", accuracy_relative, two_charge_force, accuracy);
  }
  
  // free all arrays previously allocated

  deallocate();

  // setup FFT grid resolution and g_ewald
  // normally one iteration thru while loop is all that is required
  // if grid stencil does not extend beyond neighbor proc
  //   or overlap is allowed, then done
  // else reduce order and try again

  int (*procneigh)[2] = comm->procneigh;

  GridComm *cgtmp = NULL;
  int iteration = 0;

  while (order >= minorder) {
    if (iteration && me == 0)
      error->warning(FLERR,"Reducing EVB_PPPM order b/c stencil extends "
		     "beyond nearest neighbor processor");

    set_grid_global();
    set_grid_local();
    if(overlap_allowed) break;

    cgtmp = new GridComm(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                         nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                         nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out);
      
    int tmp1,tmp2;
    cgtmp->setup(tmp1,tmp2);
    if (cgtmp->ghost_adjacent()) break;
    delete cgtmp;

    order--;
    iteration++;
  }

  if (order < minorder) error->all(FLERR,"EVB_PPPM order < minimum allowed order");
  if (!overlap_allowed && !cgtmp->ghost_adjacent())
    error->all(FLERR,"EVB_PPPM grid stencil extends beyond nearest neighbor processor");
  if (cgtmp) delete cgtmp;

  // adjust g_ewald

  if(!gewaldflag) adjust_gewald();

  // calculate the final accuracy

  double estimated_accuracy = final_accuracy();

  // print stats

  int ngrid_max,nfft_both_max,nbuf_max;
  MPI_Allreduce(&ngrid,&ngrid_max,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&nfft_both,&nfft_both_max,1,MPI_INT,MPI_MAX,world);

  if (me == 0) {

#ifdef FFT_SINGLE
    const char fft_prec[] = "single";
#else
    const char fft_prec[] = "double";
#endif

    if (screen) {
      fprintf(screen,"  G vector (1/distance)= %g\n",g_ewald);
      fprintf(screen,"  grid = %d %d %d\n",nx_pppm,ny_pppm,nz_pppm);
      fprintf(screen,"  stencil order = %d\n",order);
      if(differentiation_flag == 1) fprintf(screen,"  differentiation = ad (1 FFT energies + 1 FFT forces)\n");
      else fprintf(screen,"  differentiation = ik (1 FFT energies + 3 FFT forces)\n");
      fprintf(screen,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(screen,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(screen,"  using %s precision FFTs\n",fft_prec);
      fprintf(screen,"  3d grid and FFT values/proc = %d %d\n",
              ngrid_max,nfft_both_max);
    }
    if (logfile) {
      fprintf(logfile,"  G vector (1/distance) = %g\n",g_ewald);
      fprintf(logfile,"  grid = %d %d %d\n",nx_pppm,ny_pppm,nz_pppm);
      fprintf(logfile,"  stencil order = %d\n",order);
      if(differentiation_flag == 1) fprintf(logfile,"  differentiation = ad (1 FFT energies + 1 FFT forces)\n");
      else fprintf(logfile,"  differentiation = ik (1 FFT energies + 3 FFT forces)\n");
      fprintf(logfile,"  estimated absolute RMS force accuracy = %g\n",
              estimated_accuracy);
      fprintf(logfile,"  estimated relative force accuracy = %g\n",
              estimated_accuracy/two_charge_force);
      fprintf(logfile,"  using %s precision FFTs\n",fft_prec);
      fprintf(logfile,"  3d grid and FFT values/proc = %d %d\n",
              ngrid_max,nfft_both_max);
    }
  }

  // allocate K-space dependent memory

  allocate();

  // pre-compute Green's function denomiator expansion
  // pre-compute 1d charge distribution coefficients

  compute_gf_denom();
  if (differentiation_flag == 1) compute_sf_precoeff();
  compute_rho_coeff();
}

/* ----------------------------------------------------------------------
   adjust EVB_PPPM coeffs, called initially and whenever volume has changed 
------------------------------------------------------------------------- */

void EVB_PPPM::setup()
{
  if (triclinic) {
    setup_triclinic();
    return;
  }
  
  int i,j,k,l,m,n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab EVB_PPPM
  // z dimension for 3d EVB_PPPM is zprd since slab_volfactor = 1.0

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;
    
  delxinv = nx_pppm/xprd;
  delyinv = ny_pppm/yprd;
  delzinv = nz_pppm/zprd_slab;

  delvolinv = delxinv*delyinv*delzinv;

  double unitkx = (MY_2PI/xprd);
  double unitky = (MY_2PI/yprd);
  double unitkz = (MY_2PI/zprd_slab);

  // fkx,fky,fkz for my FFT grid pts

  double per;

  for (i = nxlo_fft; i <= nxhi_fft; i++) {
    per = i - nx_pppm*(2*i/nx_pppm);
    fkx[i] = unitkx*per;
  }

  for (i = nylo_fft; i <= nyhi_fft; i++) {
    per = i - ny_pppm*(2*i/ny_pppm);
    fky[i] = unitky*per;
  }

  for (i = nzlo_fft; i <= nzhi_fft; i++) {
    per = i - nz_pppm*(2*i/nz_pppm);
    fkz[i] = unitkz*per;
  }

  // virial coefficients

  double sqk,vterm;

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++) {
    for (j = nylo_fft; j <= nyhi_fft; j++) {
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
	sqk = fkx[i]*fkx[i] + fky[j]*fky[j] + fkz[k]*fkz[k];
	if (sqk == 0.0) {
	  vg[n][0] = 0.0;
	  vg[n][1] = 0.0;
	  vg[n][2] = 0.0;
	  vg[n][3] = 0.0;
	  vg[n][4] = 0.0;
	  vg[n][5] = 0.0;
	} else {
	  vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
	  vg[n][0] = 1.0 + vterm*fkx[i]*fkx[i];
	  vg[n][1] = 1.0 + vterm*fky[j]*fky[j];
	  vg[n][2] = 1.0 + vterm*fkz[k]*fkz[k];
	  vg[n][3] = vterm*fkx[i]*fky[j];
	  vg[n][4] = vterm*fkx[i]*fkz[k];
	  vg[n][5] = vterm*fky[j]*fkz[k];
	}
	n++;
      }
    }
  }

  if (differentiation_flag == 1) compute_gf_ad();
  else compute_gf_ik();
}

/* ----------------------------------------------------------------------
   adjust PPPM coeffs, called initially and whenever volume has changed
   for a triclinic system
------------------------------------------------------------------------- */

void EVB_PPPM::setup_triclinic()
{
  int i,j,k,n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab PPPM
  // z dimension for 3d PPPM is zprd since slab_volfactor = 1.0

  prd = domain->prd;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  // use lamda (0-1) coordinates

  delxinv = nx_pppm;
  delyinv = ny_pppm;
  delzinv = nz_pppm;
  delvolinv = delxinv*delyinv*delzinv/volume;

  // fkx,fky,fkz for my FFT grid pts

  double per_i,per_j,per_k;

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++) {
    per_k = k - nz_pppm*(2*k/nz_pppm);
    for (j = nylo_fft; j <= nyhi_fft; j++) {
      per_j = j - ny_pppm*(2*j/ny_pppm);
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        per_i = i - nx_pppm*(2*i/nx_pppm);

        double unitk_lamda[3];
        unitk_lamda[0] = 2.0*MY_PI*per_i;
        unitk_lamda[1] = 2.0*MY_PI*per_j;
        unitk_lamda[2] = 2.0*MY_PI*per_k;
        x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
        fkx[n] = unitk_lamda[0];
        fky[n] = unitk_lamda[1];
        fkz[n] = unitk_lamda[2];
        n++;
      }
    }
  }

  // virial coefficients

  double sqk,vterm;

  for (n = 0; n < nfft; n++) {
    sqk = fkx[n]*fkx[n] + fky[n]*fky[n] + fkz[n]*fkz[n];
    if (sqk == 0.0) {
      vg[n][0] = 0.0;
      vg[n][1] = 0.0;
      vg[n][2] = 0.0;
      vg[n][3] = 0.0;
      vg[n][4] = 0.0;
      vg[n][5] = 0.0;
    } else {
      vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
      vg[n][0] = 1.0 + vterm*fkx[n]*fkx[n];
      vg[n][1] = 1.0 + vterm*fky[n]*fky[n];
      vg[n][2] = 1.0 + vterm*fkz[n]*fkz[n];
      vg[n][3] = vterm*fkx[n]*fky[n];
      vg[n][4] = vterm*fkx[n]*fkz[n];
      vg[n][5] = vterm*fky[n]*fkz[n];
    }
  }

  compute_gf_ik_triclinic();
}

/* ----------------------------------------------------------------------
   reset local grid arrays and communication stencils
   called by fix balance b/c it changed sizes of processor sub-domains
------------------------------------------------------------------------- */

void EVB_PPPM::setup_grid()
{
  // free all arrays previously allocated

  deallocate();
  // deallocate_peratom();
  // peratom_allocate_flag = 0;
  // deallocate_groups();
  // group_allocate_flag = 0;

  // reset portion of global grid that each proc owns

  set_grid_local();

  // reallocate K-space dependent memory
  // check if grid communication is now overlapping if not allowed
  // don't invoke allocate_peratom(), compute() will allocate when needed

  allocate();
    
  if (overlap_allowed == 0 && !cg->ghost_adjacent())
    error->all(FLERR,"PPPM grid stencil extends "
               "beyond nearest neighbor processor");

  // pre-compute Green's function denomiator expansion
  // pre-compute 1d charge distribution coefficients

  compute_gf_denom();
  if (differentiation_flag == 1) compute_sf_precoeff();
  compute_rho_coeff();

  // pre-compute volume-dependent coeffs

  setup();
}

/* ----------------------------------------------------------------------
   compute the EVB_PPPM long-range force, energy, virial 
------------------------------------------------------------------------- */

void EVB_PPPM::compute(int eflag, int vflag)
{ 
  return;
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors and order 
------------------------------------------------------------------------- */

void EVB_PPPM::allocate()
{
  /*******************************************************/
  /*******************************************************/
  memory->create3d_offset(env_density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			  nxlo_out,nxhi_out,"EVB_PPPM:env_density_brick");
  /*******************************************************/
  /*******************************************************/
  
  memory->create3d_offset(density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
				   nxlo_out,nxhi_out,"EVB_PPPM:density_brick");


  memory->create(density_fft,nfft_both,"EVB_PPPM:density_fft");
  memory->create(greensfn,nfft_both,"EVB_PPPM:greensfn");
  memory->create(work1,2*nfft_both,"EVB_PPPM:work1");
  memory->create(work2,2*nfft_both,"EVB_PPPM:work2");
  memory->create(vg,nfft_both,6,"EVB_PPPM:vg");
  
  if (triclinic == 0) {
    memory->create1d_offset(fkx,nxlo_fft,nxhi_fft,"EVB_pppm:fkx");
    memory->create1d_offset(fky,nylo_fft,nyhi_fft,"EVB_pppm:fky");
    memory->create1d_offset(fkz,nzlo_fft,nzhi_fft,"EVB_pppm:fkz");
  } else {
    memory->create(fkx,nfft_both,"EVB_pppm:fkx");
    memory->create(fky,nfft_both,"EVB_pppm:fky");
    memory->create(fkz,nfft_both,"EVB_pppm:fkz");
  }
  
  if (differentiation_flag == 1) {
    memory->create3d_offset(u_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                          nxlo_out,nxhi_out,"EVB_PPPM:u_brick");

    memory->create(sf_precoeff1,nfft_both,"EVB_PPPM:sf_precoeff1");
    memory->create(sf_precoeff2,nfft_both,"EVB_PPPM:sf_precoeff2");
    memory->create(sf_precoeff3,nfft_both,"EVB_PPPM:sf_precoeff3");
    memory->create(sf_precoeff4,nfft_both,"EVB_PPPM:sf_precoeff4");
    memory->create(sf_precoeff5,nfft_both,"EVB_PPPM:sf_precoeff5");
    memory->create(sf_precoeff6,nfft_both,"EVB_PPPM:sf_precoeff6");

  } else {
    memory->create3d_offset(vdx_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			    nxlo_out,nxhi_out,"EVB_PPPM:vdx_brick");
    memory->create3d_offset(vdy_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			    nxlo_out,nxhi_out,"EVB_PPPM:vdy_brick");
    memory->create3d_offset(vdz_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
			    nxlo_out,nxhi_out,"EVB_PPPM:vdz_brick");
  }

  // summation coeffs

  memory->create(gf_b,order,"pppm:gf_b");
  memory->create2d_offset(rho1d,3,-order/2,order/2,"EVB_PPPM:rho1d");
  memory->create2d_offset(drho1d,3,-order/2,order/2,"EVB_PPPM:drho1d");
  memory->create2d_offset(rho_coeff,order,(1-order)/2,order/2,"EVB_PPPM:rho_coeff");
  memory->create2d_offset(drho_coeff,order,(1-order)/2,order/2,"EVB_PPPM:drho_coeff");

  // create 2 FFTs and a Remap
  // 1st FFT keeps data in FFT decompostion
  // 2nd FFT returns data in 3d brick decomposition
  // remap takes data from 3d brick to FFT decomposition

  int tmp;

  fft1 = new FFT3d(lmp,world,nx_pppm,ny_pppm,nz_pppm,
		   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
		   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
		   0,0,&tmp,collective_flag);

  fft2 = new FFT3d(lmp,world,nx_pppm,ny_pppm,nz_pppm,
		   nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
		   nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
		   0,0,&tmp,collective_flag);

  remap = new Remap(lmp,world,
		    nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
		    nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
		    1,0,0,FFT_PRECISION,collective_flag);

  // create ghost grid object for rho and electric field communication

  int (*procneigh)[2] = comm->procneigh;

  cg = new GridComm(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                    nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                    nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out);

  cg->setup(ngc_buf1,ngc_buf2);
 
  if (differentiation_flag) npergrid = 1;
  else npergrid = 3;
    
  memory->create(cg_buf1,npergrid*ngc_buf1,"pppm:gc_buf1");
  memory->create(cg_buf2,npergrid*ngc_buf2,"pppm:gc_buf2");
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors and order 
------------------------------------------------------------------------- */

void EVB_PPPM::deallocate()
{
  /*******************************************************/
  /*******************************************************/
  memory->destroy3d_offset(env_density_brick,nzlo_out,nylo_out,nxlo_out);
  /*******************************************************/
  /*******************************************************/
  
  memory->destroy3d_offset(density_brick,nzlo_out,nylo_out,nxlo_out);
 
 if (differentiation_flag == 1) {
    memory->destroy3d_offset(u_brick,nzlo_out,nylo_out,nxlo_out);
    memory->destroy(sf_precoeff1);
    memory->destroy(sf_precoeff2);
    memory->destroy(sf_precoeff3);
    memory->destroy(sf_precoeff4);
    memory->destroy(sf_precoeff5);
    memory->destroy(sf_precoeff6);
  } else {
    memory->destroy3d_offset(vdx_brick,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(vdy_brick,nzlo_out,nylo_out,nxlo_out);
    memory->destroy3d_offset(vdz_brick,nzlo_out,nylo_out,nxlo_out);
  }

  memory->sfree(density_fft);
  memory->sfree(greensfn);
  memory->sfree(work1);
  memory->sfree(work2);
  memory->destroy(vg);
  
  if (triclinic == 0) {
    memory->destroy1d_offset(fkx,nxlo_fft);
    memory->destroy1d_offset(fky,nylo_fft);
    memory->destroy1d_offset(fkz,nzlo_fft);
  } else {
    memory->destroy(fkx);
    memory->destroy(fky);
    memory->destroy(fkz);
  }
  
  memory->destroy(gf_b);
  memory->destroy2d_offset(rho1d,-order/2);
  memory->destroy2d_offset(drho1d,-order/2);
  memory->destroy2d_offset(rho_coeff,(1-order)/2);
  memory->destroy2d_offset(drho_coeff,(1-order)/2);

  delete fft1;
  delete fft2;
  delete remap;
  delete cg;
}

/* ----------------------------------------------------------------------
   set global size of PPPM grid = nx,ny,nz_pppm
   used for charge accumulation, FFTs, and electric field interpolation 
------------------------------------------------------------------------- */

void EVB_PPPM::set_grid_global()
{
  // use xprd,yprd,zprd even if triclinic so grid size is the same
  // adjust z dimension for 2d slab EVB_PPPM
  // 3d EVB_PPPM just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  
  // make initial g_ewald estimate
  // based on desired error and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  double h,h_x,h_y,h_z;
  bigint natoms = atom->natoms;

  if (!gewaldflag) {
    if(accuracy <= 0.0) error->all(FLERR,"KSpace accuaracy must be > 0");
    g_ewald = accuracy*sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
    if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy)) / cutoff;
    else g_ewald = sqrt(-log(g_ewald)) / cutoff;
  } 

  // set optimal nx_pppm,ny_pppm,nz_pppm based on order and accuracy
  // nz_pppm uses extended zprd_slab instead of zprd
  // reduce it until precision target is met

  if (!gridflag) {

    if (differentiation_flag == 1) {

      h = h_x = h_y = h_z = 4.0/g_ewald;
      int count = 0;
      while (1) {

        // set grid dimension
        nx_pppm = static_cast<int> (xprd/h_x);
        ny_pppm = static_cast<int> (yprd/h_y);
        nz_pppm = static_cast<int> (zprd_slab/h_z);

        if (nx_pppm <= 1) nx_pppm = 2;
        if (ny_pppm <= 1) ny_pppm = 2;
        if (nz_pppm <= 1) nz_pppm = 2;

        //set local grid dimension
        int npey_fft,npez_fft;
        if (nz_pppm >= nprocs) {
          npey_fft = 1;
          npez_fft = nprocs;
        } else procs2grid2d(nprocs,ny_pppm,nz_pppm,&npey_fft,&npez_fft);

        int me_y = me % npey_fft;
        int me_z = me / npey_fft;

        nxlo_fft = 0;
        nxhi_fft = nx_pppm - 1;
        nylo_fft = me_y*ny_pppm/npey_fft;
        nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
        nzlo_fft = me_z*nz_pppm/npez_fft;
        nzhi_fft = (me_z+1)*nz_pppm/npez_fft - 1;

        double df_kspace = compute_df_kspace();

        count++;

        // break loop if the accuracy has been reached or
        // too many loops have been performed

        if (df_kspace <= accuracy) break;
        if (count > 500) error->all(FLERR, "Could not compute grid size");
        h *= 0.95;
        h_x = h_y = h_z = h;
      }

    } else {

      double err;
      h_x = h_y = h_z = 1/g_ewald;  
    
      nx_pppm = static_cast<int> (xprd/h_x) + 1;
      ny_pppm = static_cast<int> (yprd/h_y) + 1;
      nz_pppm = static_cast<int> (zprd_slab/h_z) + 1;

      err = estimate_ik_error(h_x,xprd,natoms);
      while (err > accuracy) {
	err = estimate_ik_error(h_x,xprd,natoms);
	nx_pppm++;
	h_x = xprd/nx_pppm;
      }

      err = estimate_ik_error(h_y,yprd,natoms);
      while (err > accuracy) {
	err = estimate_ik_error(h_y,yprd,natoms);
	ny_pppm++;
	h_y = yprd/ny_pppm;
      }
      
      err = estimate_ik_error(h_z,zprd_slab,natoms);
      while (err > accuracy) {
	err = estimate_ik_error(h_z,zprd_slab,natoms);
	nz_pppm++;
	h_z = zprd_slab/nz_pppm;
      }
    }
    
    // scale grid for triclinic skew

    if (triclinic) {
      double tmp[3];
      tmp[0] = nx_pppm/xprd;
      tmp[1] = ny_pppm/yprd;
      tmp[2] = nz_pppm/zprd;
      lamda2xT(&tmp[0],&tmp[0]);
      nx_pppm = static_cast<int>(tmp[0]) + 1;
      ny_pppm = static_cast<int>(tmp[1]) + 1;
      nz_pppm = static_cast<int>(tmp[2]) + 1;
    }
  }

  // boost grid size until it is factorable

  while (!factorable(nx_pppm)) nx_pppm++;
  while (!factorable(ny_pppm)) ny_pppm++;
  while (!factorable(nz_pppm)) nz_pppm++;
  
  if (triclinic == 0) {
    h_x = xprd/nx_pppm;
    h_y = yprd/ny_pppm;
    h_z = zprd_slab/nz_pppm;
  } else {
    double tmp[3];
    tmp[0] = nx_pppm;
    tmp[1] = ny_pppm;
    tmp[2] = nz_pppm;
    x2lamdaT(&tmp[0],&tmp[0]);
    h_x = 1.0/tmp[0];
    h_y = 1.0/tmp[1];
    h_z = 1.0/tmp[2];
  }
  
  if (nx_pppm >= OFFSET || ny_pppm >= OFFSET || nz_pppm >= OFFSET)
    error->all(FLERR,"PPPM grid is too large");
}

/* ----------------------------------------------------------------------
   check if all factors of n are in list of factors
   return 1 if yes, 0 if no 
------------------------------------------------------------------------- */

int EVB_PPPM::factorable(int n)
{
  int i;

  while (n > 1) {
    for (i = 0; i < nfactors; i++) {
      if (n % factors[i] == 0) {
	n /= factors[i];
	break;
      }
    }
    if (i == nfactors) return 0;
  }

  return 1;
}

/* ----------------------------------------------------------------------
   compute estimated kspace force error
------------------------------------------------------------------------- */

double EVB_PPPM::compute_df_kspace()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  bigint natoms = atom->natoms;
  double df_kspace = 0.0;
  if (differentiation_flag == 1) {
    double qopt = compute_qopt();
    df_kspace = sqrt(qopt/natoms)*q2/(xprd*yprd*zprd_slab);
  } else {
    double lprx = estimate_ik_error(xprd/nx_pppm,xprd,natoms);
    double lpry = estimate_ik_error(yprd/ny_pppm,yprd,natoms);
    double lprz = estimate_ik_error(zprd_slab/nz_pppm,zprd_slab,natoms);
    df_kspace = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
  }
  return df_kspace;
}

/* ----------------------------------------------------------------------
   compute qopt
------------------------------------------------------------------------- */

double EVB_PPPM::compute_qopt()
{
  double qopt = 0.0;
  double *prd = (triclinic==0) ? domain->prd : domain->prd_lamda;
  
  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double u1, u2, sqk;
  double sum1,sum2,sum3,sum4,dot2;

  int k,l,m,nx,ny,nz;
  const int twoorder = 2*order;

  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    const int mper = m - nz_pppm*(2*m/nz_pppm);

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      const int lper = l - ny_pppm*(2*l/ny_pppm);

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        const int kper = k - nx_pppm*(2*k/nx_pppm);

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

        if (sqk != 0.0) {

          sum1 = 0.0;
          sum2 = 0.0;
          sum3 = 0.0;
          sum4 = 0.0;
          for (nx = -2; nx <= 2; nx++) {
            qx = unitkx*(kper+nx_pppm*nx);
            sx = exp(-0.25*square(qx/g_ewald));
            argx = 0.5*qx*xprd/nx_pppm;
            wx = powsinxx(argx,twoorder);
            qx *= qx;

            for (ny = -2; ny <= 2; ny++) {
              qy = unitky*(lper+ny_pppm*ny);
              sy = exp(-0.25*square(qy/g_ewald));
              argy = 0.5*qy*yprd/ny_pppm;
              wy = powsinxx(argy,twoorder);
              qy *= qy;

              for (nz = -2; nz <= 2; nz++) {
                qz = unitkz*(mper+nz_pppm*nz);
                sz = exp(-0.25*square(qz/g_ewald));
                argz = 0.5*qz*zprd_slab/nz_pppm;
                wz = powsinxx(argz,twoorder);
                qz *= qz;

                dot2 = qx+qy+qz;
                u1   = sx*sy*sz;
                u2   = wx*wy*wz;
                sum1 += u1*u1/dot2*MY_4PI*MY_4PI;
                sum2 += u1 * u2 * MY_4PI;
                sum3 += u2;
                sum4 += dot2*u2;
              }
            }
          }
          sum2 *= sum2;
          qopt += sum1 - sum2/(sum3*sum4);
        }
      }
    }
  }
  double qopt_all;
  MPI_Allreduce(&qopt,&qopt_all,1,MPI_DOUBLE,MPI_SUM,world);
  return qopt_all;
}

/* ----------------------------------------------------------------------
   estimate kspace force error for ik method
------------------------------------------------------------------------- */

double EVB_PPPM::estimate_ik_error(double h, double prd, bigint natoms)
{
  double sum = 0.0;
  for (int m = 0; m < order; m++)
    sum += acons[order][m] * pow(h*g_ewald,2.0*m);
  double value = q2 * pow(h*g_ewald,(double)order) *
    sqrt(g_ewald*prd*sqrt(MY_2PI)*sum/natoms) / (prd*prd);

  return value;
}

/* ----------------------------------------------------------------------
   adjust the g_ewald parameter to near its optimal value
   using a Newton-Raphson solver
------------------------------------------------------------------------- */

void EVB_PPPM::adjust_gewald()
{
  double dx;

  for (int i = 0; i < LARGE; i++) {
    dx = newton_raphson_f() / derivf();
    g_ewald -= dx;
    if (fabs(newton_raphson_f()) < SMALL) return;
  }

  char str[128];
  sprintf(str, "Could not compute g_ewald");
  error->all(FLERR, str);
}

/* ----------------------------------------------------------------------
 Calculate f(x) using Newton-Raphson solver
 ------------------------------------------------------------------------- */

double EVB_PPPM::newton_raphson_f()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  bigint natoms = atom->natoms;

  double df_rspace = 2.0*q2*exp(-g_ewald*g_ewald*cutoff*cutoff) /
       sqrt(natoms*cutoff*xprd*yprd*zprd);

  double df_kspace = compute_df_kspace();

  return df_rspace - df_kspace;
}

/* ----------------------------------------------------------------------
 Calculate numerical derivative f'(x) using forward difference
 [f(x + h) - f(x)] / h
 ------------------------------------------------------------------------- */

double EVB_PPPM::derivf()
{
  double h = 0.000001;  //Derivative step-size
  double df,f1,f2,g_ewald_old;

  f1 = newton_raphson_f();
  g_ewald_old = g_ewald;
  g_ewald += h;
  f2 = newton_raphson_f();
  g_ewald = g_ewald_old;
  df = (f2 - f1)/h;

  return df;
}

/* ----------------------------------------------------------------------
   Calculate the final estimate of the accuracy
------------------------------------------------------------------------- */

double EVB_PPPM::final_accuracy()
{
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;
  bigint natoms = atom->natoms;

  double df_kspace = compute_df_kspace();
  double q2_over_sqrt = q2 / sqrt(natoms*cutoff*xprd*yprd*zprd_slab);
  double df_rspace = 2.0 * q2_over_sqrt * exp(-g_ewald*g_ewald*cutoff*cutoff);
  double df_table = estimate_table_accuracy(q2_over_sqrt,df_rspace);
  double estimated_accuracy = sqrt(df_kspace*df_kspace + df_rspace*df_rspace +
   df_table*df_table);

  return estimated_accuracy;
}

/* ----------------------------------------------------------------------
   set local subset of PPPM/FFT grid that I own
   n xyz lo/hi in = 3d brick that I own (inclusive)
   n xyz lo/hi out = 3d brick + ghost cells in 6 directions (inclusive)
   n xyz lo/hi fft = FFT columns that I own (all of x dim, 2d decomp in yz)
------------------------------------------------------------------------- */

void EVB_PPPM::set_grid_local()
{
  // global indices of PPPM grid range from 0 to N-1
  // nlo_in,nhi_in = lower/upper limits of the 3d sub-brick of
  //   global PPPM grid that I own without ghost cells
  // for slab PPPM, assign z grid as if it were not extended

  nxlo_in = static_cast<int> (comm->xsplit[comm->myloc[0]] * nx_pppm);
  nxhi_in = static_cast<int> (comm->xsplit[comm->myloc[0]+1] * nx_pppm) - 1;

  nylo_in = static_cast<int> (comm->ysplit[comm->myloc[1]] * ny_pppm);
  nyhi_in = static_cast<int> (comm->ysplit[comm->myloc[1]+1] * ny_pppm) - 1;

  nzlo_in = static_cast<int>
      (comm->zsplit[comm->myloc[2]] * nz_pppm/slab_volfactor);
  nzhi_in = static_cast<int>
      (comm->zsplit[comm->myloc[2]+1] * nz_pppm/slab_volfactor) - 1;

  // nlower,nupper = stencil size for mapping particles to PPPM grid

  nlower = -(order-1)/2;
  nupper = order/2;

  // shift values for particle <-> grid mapping
  // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

  if (order % 2) shift = OFFSET + 0.5;
  else shift = OFFSET;
  if (order % 2) shiftone = 0.0;
  else shiftone = 0.5;

  // nlo_out,nhi_out = lower/upper limits of the 3d sub-brick of
  //   global PPPM grid that my particles can contribute charge to
  // effectively nlo_in,nhi_in + ghost cells
  // nlo,nhi = global coords of grid pt to "lower left" of smallest/largest
  //           position a particle in my box can be at
  // dist[3] = particle position bound = subbox + skin/2.0 + qdist
  //   qdist = offset due to TIP4P fictitious charge
  //   convert to triclinic if necessary
  // nlo_out,nhi_out = nlo,nhi + stencil size for particle mapping
  // for slab PPPM, assign z grid as if it were not extended

  double *prd,*sublo,*subhi;

  if (triclinic == 0) {
    prd = domain->prd;
    boxlo = domain->boxlo;
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    prd = domain->prd_lamda;
    boxlo = domain->boxlo_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double zprd_slab = zprd*slab_volfactor;

  double dist[3];
  double cuthalf = 0.5*neighbor->skin + qdist;
  if (triclinic == 0) dist[0] = dist[1] = dist[2] = cuthalf;
  else kspacebbox(cuthalf,&dist[0]);

  int nlo,nhi;

  nlo = static_cast<int> ((sublo[0]-dist[0]-boxlo[0]) *
                            nx_pppm/xprd + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[0]+dist[0]-boxlo[0]) *
                            nx_pppm/xprd + shift) - OFFSET;
  nxlo_out = nlo + nlower;
  nxhi_out = nhi + nupper;

  nlo = static_cast<int> ((sublo[1]-dist[1]-boxlo[1]) *
                            ny_pppm/yprd + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[1]+dist[1]-boxlo[1]) *
                            ny_pppm/yprd + shift) - OFFSET;
  nylo_out = nlo + nlower;
  nyhi_out = nhi + nupper;

  nlo = static_cast<int> ((sublo[2]-dist[2]-boxlo[2]) *
                            nz_pppm/zprd_slab + shift) - OFFSET;
  nhi = static_cast<int> ((subhi[2]+dist[2]-boxlo[2]) *
                            nz_pppm/zprd_slab + shift) - OFFSET;
  nzlo_out = nlo + nlower;
  nzhi_out = nhi + nupper;

  // for slab PPPM, change the grid boundary for processors at +z end
  //   to include the empty volume between periodically repeating slabs
  // for slab PPPM, want charge data communicated from -z proc to +z proc,
  //   but not vice versa, also want field data communicated from +z proc to
  //   -z proc, but not vice versa
  // this is accomplished by nzhi_in = nzhi_out on +z end (no ghost cells)
  // also insure no other procs use ghost cells beyond +z limit

  if (slabflag) {
    if (comm->myloc[2] == comm->procgrid[2]-1)
      nzhi_in = nzhi_out = nz_pppm - 1;
    nzhi_out = MIN(nzhi_out,nz_pppm-1);
  }
    
  // decomposition of FFT mesh
  // global indices range from 0 to N-1
  // proc owns entire x-dimension, clumps of columns in y,z dimensions
  // npey_fft,npez_fft = # of procs in y,z dims
  // if nprocs is small enough, proc can own 1 or more entire xy planes,
  //   else proc owns 2d sub-blocks of yz plane
  // me_y,me_z = which proc (0-npe_fft-1) I am in y,z dimensions
  // nlo_fft,nhi_fft = lower/upper limit of the section
  //   of the global FFT mesh that I own

  int npey_fft,npez_fft;
  if (nz_pppm >= nprocs) {
    npey_fft = 1;
    npez_fft = nprocs;
  } else procs2grid2d(nprocs,ny_pppm,nz_pppm,&npey_fft,&npez_fft);

  int me_y = me % npey_fft;
  int me_z = me / npey_fft;

  nxlo_fft = 0;
  nxhi_fft = nx_pppm - 1;
  nylo_fft = me_y*ny_pppm/npey_fft;
  nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
  nzlo_fft = me_z*nz_pppm/npez_fft;
  nzhi_fft = (me_z+1)*nz_pppm/npez_fft - 1;

  // PPPM grid pts owned by this proc, including ghosts

  ngrid = (nxhi_out-nxlo_out+1) * (nyhi_out-nylo_out+1) *
    (nzhi_out-nzlo_out+1);

  // FFT grids owned by this proc, without ghosts
  // nfft = FFT points in FFT decomposition on this proc
  // nfft_brick = FFT points in 3d brick-decomposition on this proc
  // nfft_both = greater of 2 values

  nfft = (nxhi_fft-nxlo_fft+1) * (nyhi_fft-nylo_fft+1) *
    (nzhi_fft-nzlo_fft+1);
  int nfft_brick = (nxhi_in-nxlo_in+1) * (nyhi_in-nylo_in+1) *
    (nzhi_in-nzlo_in+1);
  nfft_both = MAX(nfft,nfft_brick);
}

/* ----------------------------------------------------------------------
   pre-compute Green's function denominator expansion coeffs, Gamma(2n) 
------------------------------------------------------------------------- */

void EVB_PPPM::compute_gf_denom()
{
  int k,l,m;
  
  for (l = 1; l < order; l++) gf_b[l] = 0.0;
  gf_b[0] = 1.0;
  
  for (m = 1; m < order; m++) {
    for (l = m; l > 0; l--) 
      gf_b[l] = 4.0 * (gf_b[l]*(l-m)*(l-m-0.5)-gf_b[l-1]*(l-m-1)*(l-m-1));
    gf_b[0] = 4.0 * (gf_b[0]*(l-m)*(l-m-0.5));
  }

  int ifact = 1;
  for (k = 1; k < 2*order; k++) ifact *= k;
  double gaminv = 1.0/ifact;
  for (l = 0; l < order; l++) gf_b[l] *= gaminv;
}

/* ----------------------------------------------------------------------
   pre-compute modified (Hockney-Eastwood) Coulomb Green's function
------------------------------------------------------------------------- */

void EVB_PPPM::compute_gf_ik()
{
  const double * const prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double sum1,dot1,dot2;
  double numerator,denominator;
  double sqk;

  int k,l,m,n,nx,ny,nz,kper,lper,mper;

  const int nbx = static_cast<int> ((g_ewald*xprd/(MY_PI*nx_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nby = static_cast<int> ((g_ewald*yprd/(MY_PI*ny_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int nbz = static_cast<int> ((g_ewald*zprd_slab/(MY_PI*nz_pppm)) *
                                    pow(-log(EPS_HOC),0.25));
  const int twoorder = 2*order;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    snz = square(sin(0.5*unitkz*mper*zprd_slab/nz_pppm));

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      sny = square(sin(0.5*unitky*lper*yprd/ny_pppm));

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);
        snx = square(sin(0.5*unitkx*kper*xprd/nx_pppm));

        sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

        if (sqk != 0.0) {
          numerator = 12.5663706/sqk;
          denominator = gf_denom(snx,sny,snz);
          sum1 = 0.0;

          for (nx = -nbx; nx <= nbx; nx++) {
            qx = unitkx*(kper+nx_pppm*nx);
            sx = exp(-0.25*square(qx/g_ewald));
            argx = 0.5*qx*xprd/nx_pppm;
            wx = powsinxx(argx,twoorder);

            for (ny = -nby; ny <= nby; ny++) {
              qy = unitky*(lper+ny_pppm*ny);
              sy = exp(-0.25*square(qy/g_ewald));
              argy = 0.5*qy*yprd/ny_pppm;
              wy = powsinxx(argy,twoorder);

              for (nz = -nbz; nz <= nbz; nz++) {
                qz = unitkz*(mper+nz_pppm*nz);
                sz = exp(-0.25*square(qz/g_ewald));
                argz = 0.5*qz*zprd_slab/nz_pppm;
                wz = powsinxx(argz,twoorder);

                dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                dot2 = qx*qx+qy*qy+qz*qz;
                sum1 += (dot1/dot2) * sx*sy*sz * wx*wy*wz;
              }
            }
          }
          greensfn[n++] = numerator*sum1/denominator;
        } else greensfn[n++] = 0.0;
      }
    }
  }
}

void EVB_PPPM::compute_gf_ik_triclinic()
{
  double snx,sny,snz;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double sum1,dot1,dot2;
  double numerator,denominator;
  double sqk;

  int k,l,m,n,nx,ny,nz,kper,lper,mper;

  double tmp[3];
  tmp[0] = (g_ewald/(MY_PI*nx_pppm)) * pow(-log(EPS_HOC),0.25);
  tmp[1] = (g_ewald/(MY_PI*ny_pppm)) * pow(-log(EPS_HOC),0.25);
  tmp[2] = (g_ewald/(MY_PI*nz_pppm)) * pow(-log(EPS_HOC),0.25);
  lamda2xT(&tmp[0],&tmp[0]);
  const int nbx = static_cast<int> (tmp[0]);
  const int nby = static_cast<int> (tmp[1]);
  const int nbz = static_cast<int> (tmp[2]);

  const int twoorder = 2*order;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    snz = square(sin(MY_PI*mper/nz_pppm));

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      sny = square(sin(MY_PI*lper/ny_pppm));

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);
        snx = square(sin(MY_PI*kper/nx_pppm));

        double unitk_lamda[3];
        unitk_lamda[0] = 2.0*MY_PI*kper;
        unitk_lamda[1] = 2.0*MY_PI*lper;
        unitk_lamda[2] = 2.0*MY_PI*mper;
        x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);

        sqk = square(unitk_lamda[0]) + square(unitk_lamda[1]) + square(unitk_lamda[2]);

        if (sqk != 0.0) {
          numerator = 12.5663706/sqk;
          denominator = gf_denom(snx,sny,snz);
          sum1 = 0.0;

          for (nx = -nbx; nx <= nbx; nx++) {
            argx = MY_PI*kper/nx_pppm + MY_PI*nx;
            wx = powsinxx(argx,twoorder);

            for (ny = -nby; ny <= nby; ny++) {
              argy = MY_PI*lper/ny_pppm + MY_PI*ny;
              wy = powsinxx(argy,twoorder);

              for (nz = -nbz; nz <= nbz; nz++) {
                argz = MY_PI*mper/nz_pppm + MY_PI*nz;
                wz = powsinxx(argz,twoorder);

                double b[3];
                b[0] = 2.0*MY_PI*nx_pppm*nx;
                b[1] = 2.0*MY_PI*ny_pppm*ny;
                b[2] = 2.0*MY_PI*nz_pppm*nz;
                x2lamdaT(&b[0],&b[0]);

                qx = unitk_lamda[0]+b[0];
                sx = exp(-0.25*square(qx/g_ewald));

                qy = unitk_lamda[1]+b[1];
                sy = exp(-0.25*square(qy/g_ewald));

                qz = unitk_lamda[2]+b[2];
                sz = exp(-0.25*square(qz/g_ewald));

                dot1 = unitk_lamda[0]*qx + unitk_lamda[1]*qy + unitk_lamda[2]*qz;
                dot2 = qx*qx+qy*qy+qz*qz;
                sum1 += (dot1/dot2) * sx*sy*sz * wx*wy*wz;
              }
            }
          }
          greensfn[n++] = numerator*sum1/denominator;
        } else greensfn[n++] = 0.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute optimized Green's function for energy calculation
------------------------------------------------------------------------- */

void EVB_PPPM::compute_gf_ad()
{
  const double * const prd = (triclinic==0) ? domain->prd : domain->prd_lamda;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double zprd_slab = zprd*slab_volfactor;
  const double unitkx = (MY_2PI/xprd);
  const double unitky = (MY_2PI/yprd);
  const double unitkz = (MY_2PI/zprd_slab);

  double snx,sny,snz,sqk;
  double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
  double numerator,denominator;
  int k,l,m,n,kper,lper,mper;

  const int twoorder = 2*order;

  for (int i = 0; i < 6; i++) sf_coeff[i] = 0.0;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);
    qz = unitkz*mper;
    snz = square(sin(0.5*qz*zprd_slab/nz_pppm));
    sz = exp(-0.25*square(qz/g_ewald));
    argz = 0.5*qz*zprd_slab/nz_pppm;
    wz = powsinxx(argz,twoorder);

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);
      qy = unitky*lper;
      sny = square(sin(0.5*qy*yprd/ny_pppm));
      sy = exp(-0.25*square(qy/g_ewald));
      argy = 0.5*qy*yprd/ny_pppm;
      wy = powsinxx(argy,twoorder);

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);
        qx = unitkx*kper;
        snx = square(sin(0.5*qx*xprd/nx_pppm));
        sx = exp(-0.25*square(qx/g_ewald));
        argx = 0.5*qx*xprd/nx_pppm;
        wx = powsinxx(argx,twoorder);

        sqk = qx*qx + qy*qy + qz*qz;

        if (sqk != 0.0) {
          numerator = MY_4PI/sqk;
          denominator = gf_denom(snx,sny,snz);
          greensfn[n] = numerator*sx*sy*sz*wx*wy*wz/denominator;
          sf_coeff[0] += sf_precoeff1[n]*greensfn[n];
          sf_coeff[1] += sf_precoeff2[n]*greensfn[n];
          sf_coeff[2] += sf_precoeff3[n]*greensfn[n];
          sf_coeff[3] += sf_precoeff4[n]*greensfn[n];
          sf_coeff[4] += sf_precoeff5[n]*greensfn[n];
          sf_coeff[5] += sf_precoeff6[n]*greensfn[n];
          n++;
        } else {
          greensfn[n] = 0.0;
          sf_coeff[0] += sf_precoeff1[n]*greensfn[n];
          sf_coeff[1] += sf_precoeff2[n]*greensfn[n];
          sf_coeff[2] += sf_precoeff3[n]*greensfn[n];
          sf_coeff[3] += sf_precoeff4[n]*greensfn[n];
          sf_coeff[4] += sf_precoeff5[n]*greensfn[n];
          sf_coeff[5] += sf_precoeff6[n]*greensfn[n];
          n++;
        }
      }
    }
  }

  // compute the coefficients for the self-force correction

  double prex, prey, prez;
  prex = prey = prez = MY_PI/volume;
  prex *= nx_pppm/xprd;
  prey *= ny_pppm/yprd;
  prez *= nz_pppm/zprd_slab;
  sf_coeff[0] *= prex;
  sf_coeff[1] *= prex*2;
  sf_coeff[2] *= prey;
  sf_coeff[3] *= prey*2;
  sf_coeff[4] *= prez;
  sf_coeff[5] *= prez*2;

  // communicate values with other procs

  double tmp[6];
  MPI_Allreduce(sf_coeff,tmp,6,MPI_DOUBLE,MPI_SUM,world);
  for (n = 0; n < 6; n++) sf_coeff[n] = tmp[n];
}

/* ----------------------------------------------------------------------
   compute self force coefficients for ad-differentiation scheme
------------------------------------------------------------------------- */

void EVB_PPPM::compute_sf_precoeff()
{
  int i,k,l,m,n;
  int nx,ny,nz,kper,lper,mper;
  double wx0[5],wy0[5],wz0[5],wx1[5],wy1[5],wz1[5],wx2[5],wy2[5],wz2[5];
  double qx0,qy0,qz0,qx1,qy1,qz1,qx2,qy2,qz2;
  double u0,u1,u2,u3,u4,u5,u6;
  double sum1,sum2,sum3,sum4,sum5,sum6;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm*(2*m/nz_pppm);

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm*(2*l/ny_pppm);

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm*(2*k/nx_pppm);

        sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = 0.0;
        for (i = 0; i < 5; i++) {

          qx0 = MY_2PI*(kper+nx_pppm*(i-2));
          qx1 = MY_2PI*(kper+nx_pppm*(i-1));
          qx2 = MY_2PI*(kper+nx_pppm*(i  ));
          wx0[i] = powsinxx(0.5*qx0/nx_pppm,order);
          wx1[i] = powsinxx(0.5*qx1/nx_pppm,order);
          wx2[i] = powsinxx(0.5*qx2/nx_pppm,order);

          qy0 = MY_2PI*(lper+ny_pppm*(i-2));
          qy1 = MY_2PI*(lper+ny_pppm*(i-1));
          qy2 = MY_2PI*(lper+ny_pppm*(i  ));
          wy0[i] = powsinxx(0.5*qy0/ny_pppm,order);
          wy1[i] = powsinxx(0.5*qy1/ny_pppm,order);
          wy2[i] = powsinxx(0.5*qy2/ny_pppm,order);

          qz0 = MY_2PI*(mper+nz_pppm*(i-2));
          qz1 = MY_2PI*(mper+nz_pppm*(i-1));
          qz2 = MY_2PI*(mper+nz_pppm*(i  ));

          wz0[i] = powsinxx(0.5*qz0/nz_pppm,order);
          wz1[i] = powsinxx(0.5*qz1/nz_pppm,order);
          wz2[i] = powsinxx(0.5*qz2/nz_pppm,order);
        }

        for (nx = 0; nx < 5; nx++) {
          for (ny = 0; ny < 5; ny++) {
            for (nz = 0; nz < 5; nz++) {
              u0 = wx0[nx]*wy0[ny]*wz0[nz];
              u1 = wx1[nx]*wy0[ny]*wz0[nz];
              u2 = wx2[nx]*wy0[ny]*wz0[nz];
              u3 = wx0[nx]*wy1[ny]*wz0[nz];
              u4 = wx0[nx]*wy2[ny]*wz0[nz];
              u5 = wx0[nx]*wy0[ny]*wz1[nz];
              u6 = wx0[nx]*wy0[ny]*wz2[nz];

              sum1 += u0*u1;
              sum2 += u0*u2;
              sum3 += u0*u3;
              sum4 += u0*u4;
              sum5 += u0*u5;
              sum6 += u0*u6;
            }
          }
        }

        // store values

        sf_precoeff1[n] = sum1;
        sf_precoeff2[n] = sum2;
        sf_precoeff3[n] = sum3;
        sf_precoeff4[n] = sum4;
        sf_precoeff5[n] = sum5;
        sf_precoeff6[n++] = sum6;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   ghost-swap to accumulate full density in brick decomposition 
   remap density from 3d brick decomposition to FFT decomposition
------------------------------------------------------------------------- */

void EVB_PPPM::brick2fft()
{
  int i,n,ix,iy,iz;

  // copy grabs inner portion of density from 3d brick
  // remap could be done as pre-stage of FFT,
  //   but this works optimally on only double values, not complex values

  n = 0;
  for (iz = nzlo_in; iz <= nzhi_in; iz++)
    for (iy = nylo_in; iy <= nyhi_in; iy++)
      for (ix = nxlo_in; ix <= nxhi_in; ix++)
	density_fft[n++] = density_brick[iz][iy][ix];

  remap->perform(density_fft,density_fft,work1);
}

/* ----------------------------------------------------------------------
   find center grid pt for each of my particles
   check that full stencil for the particle will fit in my 3d brick
   store central grid pt indices in part2grid array 
------------------------------------------------------------------------- */

void EVB_PPPM::particle_map()
{
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];
  
  int nx,ny,nz;

  double **x = atom->x;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (int i = 0; i < nlocal; i++) {
    
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // current particle coord can be outside global and local box
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

    nx = static_cast<int> ((x[i][0] - boxlox) * delxinv+shift) - OFFSET;
    ny = static_cast<int> ((x[i][1] - boxloy) * delyinv+shift) - OFFSET;
    nz = static_cast<int> ((x[i][2] - boxloz) * delzinv+shift) - OFFSET;

    part2grid[i][0] = nx;
    part2grid[i][1] = ny;
    part2grid[i][2] = nz;

    // check that entire stencil around nx,ny,nz will fit in my 3d brick

    if (nx+nlower < nxlo_out || nx+nupper > nxhi_out ||
	ny+nlower < nylo_out || ny+nupper > nyhi_out ||
	nz+nlower < nzlo_out || nz+nupper > nzhi_out) flag = 1;
  }
  
  if (flag) error->all(FLERR,"Out of range atoms - cannot compute EVB_PPPM");
}

/* ----------------------------------------------------------------------
   create discretized "density" on section of global grid due to my particles
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including ghosts)
   in global grid 
------------------------------------------------------------------------- */

void EVB_PPPM::make_rho()
{
  const double * const q = atom->q;
  const double * const * const x = atom->x;
  const int nlocal = atom->nlocal;
  
  // set up clear 3d density array
  FFT_SCALAR * const * const * const db = &(density_brick[0]);
  memset(&(db[nzlo_out][nylo_out][nxlo_out]),0,ngrid*sizeof(FFT_SCALAR));

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  
  if (order == 5) {
    for (int i = 0; i < nlocal; i++) {
      const double ddx = (x[i][0]- boxlox) * delxinv;
      const double ddy = (x[i][1]- boxloy) * delyinv;
      const double ddz = (x[i][2]- boxloz) * delzinv;
      const int nx = static_cast<int> (ddx+shift) - OFFSET;
      const int ny = static_cast<int> (ddy+shift) - OFFSET;
      const int nz = static_cast<int> (ddz+shift) - OFFSET;
      part2grid[i][0] = nx;
      part2grid[i][1] = ny;
      part2grid[i][2] = nz;
      const FFT_SCALAR dx = nx+shiftone - ddx;
      const FFT_SCALAR dy = ny+shiftone - ddy;
      const FFT_SCALAR dz = nz+shiftone - ddz; 
      part2grid_dr[i][0] = dx;
      part2grid_dr[i][1] = dy;
      part2grid_dr[i][2] = dz; 

      // Code specific to order = 5
      //compute_rho1d_thr(r1d,dx,dy,dz);
      // completely unrolled loop
      const FFT_SCALAR dx2 = dx*dx;
      const FFT_SCALAR dx3 = dx2*dx;
      const FFT_SCALAR dx4 = dx2*dx2;
      const FFT_SCALAR dy2 = dy*dy;
      const FFT_SCALAR dy3 = dy2*dy;
      const FFT_SCALAR dy4 = dy2*dy2;
      const FFT_SCALAR dz2 = dz*dz;
      const FFT_SCALAR dz3 = dz2*dz;
      const FFT_SCALAR dz4 = dz2*dz2;
      int k = -2;
      rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
      rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
      rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
      k = -1;
      rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
      rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
      rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
      k = 0;
      rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
      rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
      rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
      k = 1;
      rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
      rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
      rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
      k = 2;
      rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
      rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
      rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
      
      const FFT_SCALAR z0 = delvolinv * q[i];
      for (int n = nlower; n <= nupper; n++) {
	const FFT_SCALAR y0 = z0*rho1d[2][n];
	for (int m = nlower; m <= nupper; m++) {
	  const FFT_SCALAR x0 = y0*rho1d[1][m];
	  for (int l = nlower; l <= nupper; l++) {
	    db[n+nz][m+ny][l+nx] += x0*rho1d[0][l];
	  }
	}
      }
    } // for(i<nlocal)
  } else {
    for (int i = 0; i < nlocal; i++) {
      const double ddx = (x[i][0]-boxlox)*delxinv;
      const double ddy = (x[i][1]-boxloy)*delyinv;
      const double ddz = (x[i][2]-boxloz)*delzinv;
      const int nx = static_cast<int> (ddx+shift) - OFFSET;
      const int ny = static_cast<int> (ddy+shift) - OFFSET;
      const int nz = static_cast<int> (ddz+shift) - OFFSET;
      part2grid[i][0] = nx;
      part2grid[i][1] = ny;
      part2grid[i][2] = nz;
      const FFT_SCALAR dx = nx+shiftone - ddx;
      const FFT_SCALAR dy = ny+shiftone - ddy;
      const FFT_SCALAR dz = nz+shiftone - ddz; 
      part2grid_dr[i][0] = dx;
      part2grid_dr[i][1] = dy;
      part2grid_dr[i][2] = dz;

      // General order code 
      compute_rho1d(dx,dy,dz);
      
      const FFT_SCALAR z0 = delvolinv * q[i];
      for (int n = nlower; n <= nupper; n++) {
	const FFT_SCALAR y0 = z0*rho1d[2][n];
	for (int m = nlower; m <= nupper; m++) {
	  const FFT_SCALAR x0 = y0*rho1d[1][m];
	  for (int l = nlower; l <= nupper; l++) {
	    db[n+nz][m+ny][l+nx] += x0*rho1d[0][l];
	  }
	}
      }

    } // for (i<nlocal)
  } // if (order == 5)
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver 
------------------------------------------------------------------------- */

void EVB_PPPM::poisson(int eflag, int vflag)
{
  if (differentiation_flag == 1) poisson_ad(vflag);
  else poisson_ik(vflag);
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik
------------------------------------------------------------------------- */

void EVB_PPPM::poisson_ik(int vflag)
{
  int i,j,k,n;
  double eng;

  // transform charge density (r -> k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,1);

  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;

  if (vflag) {
    n = 0;
    for (i = 0; i < nfft; i++) {
      eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
      for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
      energy += eng;
      n += 2;
    }
  } else {
    n = 0;
    for (i = 0; i < nfft; i++) {
      energy += greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
      n += 2;
    }
    energy *= s2;
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }
  
  if (triclinic) {
    poisson_ik_triclinic();
    return;
  }
  
  // compute gradients of V(r) in each of 3 dims by transformimg -ik*V(k)
  // FFT leaves data in 3d brick decomposition
  // copy it into inner portion of vdx,vdy,vdz arrays

  // x direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = -fkx[i]*work1[n+1];
        work2[n+1] = fkx[i]*work1[n];
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

  // y direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = -fky[j]*work1[n+1];
        work2[n+1] = fky[j]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdy_brick[k][j][i] = work2[n];
        n += 2;
      }

  // z direction gradient

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++)
    for (j = nylo_fft; j <= nyhi_fft; j++)
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        work2[n] = -fkz[k]*work1[n+1];
        work2[n+1] = fkz[k]*work1[n];
        n += 2;
      }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdz_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ik for a triclinic system
------------------------------------------------------------------------- */

void EVB_PPPM::poisson_ik_triclinic()
{
  int i,j,k,n;

  // compute gradients of V(r) in each of 3 dims by transforming ik*V(k)
  // FFT leaves data in 3d brick decomposition
  // copy it into inner portion of vdx,vdy,vdz arrays

  // x direction gradient

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = -fkx[i]*work1[n+1];
    work2[n+1] = fkx[i]*work1[n];
    n += 2;
  }

  fft2->compute(work2,work2,FFT3d::BACKWARD);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdx_brick[k][j][i] = work2[n];
        n += 2;
      }

  // y direction gradient

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = -fky[i]*work1[n+1];
    work2[n+1] = fky[i]*work1[n];
    n += 2;
  }

  fft2->compute(work2,work2,FFT3d::BACKWARD);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdy_brick[k][j][i] = work2[n];
        n += 2;
      }

  // z direction gradient

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = -fkz[i]*work1[n+1];
    work2[n+1] = fkz[i]*work1[n];
    n += 2;
  }

  fft2->compute(work2,work2,FFT3d::BACKWARD);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        vdz_brick[k][j][i] = work2[n];
        n += 2;
      }
}


/* ----------------------------------------------------------------------
   FFT-based Poisson solver for ad
------------------------------------------------------------------------- */

void EVB_PPPM::poisson_ad(int vflag)
{
  int i,j,k,n;
  double eng;

  // transform charge density (r -> k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] = density_fft[i];
    work1[n++] = ZEROF;
  }

  fft1->compute(work1,work1,1);

  // global energy and virial contribution

  double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
  double s2 = scaleinv*scaleinv;
  
  if (vflag) {
    n = 0;
    for (i = 0; i < nfft; i++) {
      eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
      for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
      energy += eng;
      n += 2;
    }
  } else {
    n = 0;
    for (i = 0; i < nfft; i++) {
      energy +=
	s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
      n += 2;
    }
  }

  // scale by 1/total-grid-pts to get rho(k)
  // multiply by Green's function to get V(k)

  n = 0;
  for (i = 0; i < nfft; i++) {
    work1[n++] *= scaleinv * greensfn[i];
    work1[n++] *= scaleinv * greensfn[i];
  }

  n = 0;
  for (i = 0; i < nfft; i++) {
    work2[n] = work1[n];
    work2[n+1] = work1[n+1];
    n += 2;
  }

  fft2->compute(work2,work2,-1);

  n = 0;
  for (k = nzlo_in; k <= nzhi_in; k++)
    for (j = nylo_in; j <= nyhi_in; j++)
      for (i = nxlo_in; i <= nxhi_in; i++) {
        u_brick[k][j][i] = work2[n];
        n += 2;
      }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles 
------------------------------------------------------------------------- */

void EVB_PPPM::fieldforce()
{
  if (differentiation_flag == 1) fieldforce_ad();
  else fieldforce_ik();
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ik
------------------------------------------------------------------------- */

void EVB_PPPM::fieldforce_ik()
{
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx,eky,ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;

  int nlocal = atom->nlocal;
  
  for (i = 0; i < nlocal; i++) {
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx+shiftone - (x[i][0] - boxlox) * delxinv;
    dy = ny+shiftone - (x[i][1] - boxloy) * delyinv;
    dz = nz+shiftone - (x[i][2] - boxloz) * delzinv;
    compute_rho1d(dx,dy,dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n+nz;
      z0 = rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m+ny;
        y0 = z0*rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l+nx;
          x0 = y0*rho1d[0][l];
          ekx -= x0*vdx_brick[mz][my][mx];
          eky -= x0*vdy_brick[mz][my][mx];
          ekz -= x0*vdz_brick[mz][my][mx];
        }
      }
    }
    
    // convert E-field to force

    const double qfactor = force->qqrd2e * scale * q[i];
    f[i][0] += qfactor*ekx;
    f[i][1] += qfactor*eky;
    if (slabflag != 2) f[i][2] += qfactor*ekz;
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for ad
------------------------------------------------------------------------- */

void EVB_PPPM::fieldforce_ad()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz;
  FFT_SCALAR ekx,eky,ekz;
  double s1,s2,s3;
  double sf = 0.0;
  double *prd;

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];

  double hx_inv = nx_pppm/xprd;
  double hy_inv = ny_pppm/yprd;
  double hz_inv = nz_pppm/zprd;

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;

  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx+shiftone - (x[i][0] - boxlox) * delxinv;
    dy = ny+shiftone - (x[i][1] - boxloy) * delyinv;
    dz = nz+shiftone - (x[i][2] - boxloz) * delzinv;

    compute_rho1d(dx,dy,dz);
    compute_drho1d(dx,dy,dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n+nz;
      for (m = nlower; m <= nupper; m++) {
        my = m+ny;
        for (l = nlower; l <= nupper; l++) {
          mx = l+nx;
          ekx += drho1d[0][l]*rho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
          eky += rho1d[0][l]*drho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
          ekz += rho1d[0][l]*rho1d[1][m]*drho1d[2][n]*u_brick[mz][my][mx];
        }
      }
    }
    ekx *= hx_inv;
    eky *= hy_inv;
    ekz *= hz_inv;

    // convert E-field to force and substract self forces

    const double qi = q[i];
    const double qfactor = force->qqrd2e * scale;

    s1 = x[i][0]*hx_inv;
    s2 = x[i][1]*hy_inv;
    s3 = x[i][2]*hz_inv;
    sf = sf_coeff[0]*sin(2*MY_PI*s1);
    sf += sf_coeff[1]*sin(4*MY_PI*s1);
    sf *= 2*qi*qi;
    f[i][0] += qfactor*(ekx*qi - sf);

    sf = sf_coeff[2]*sin(2*MY_PI*s2);
    sf += sf_coeff[3]*sin(4*MY_PI*s2);
    sf *= 2*qi*qi;
    f[i][1] += qfactor*(eky*qi - sf);


    sf = sf_coeff[4]*sin(2*MY_PI*s3);
    sf += sf_coeff[5]*sin(4*MY_PI*s3);
    sf *= 2*qi*qi;
    if (slabflag != 2) f[i][2] += qfactor*(ekz*qi - sf);
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my ENV particles 
------------------------------------------------------------------------- */

void EVB_PPPM::fieldforce_env()
{
  if (differentiation_flag == 1) fieldforce_env_ad();
  else fieldforce_env_ik();
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my ENV particles for ik
------------------------------------------------------------------------- */

void EVB_PPPM::fieldforce_env_ik()
{
  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz,x0,y0,z0;
  FFT_SCALAR ekx,eky,ekz;

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;

  int nlocal = atom->nlocal;

  int * is_cplx_atom = evb_engine->complex_atom;

  for (i = 0; i < nlocal; i++) {
    if(is_cplx_atom[i]) continue; // Skip complex atoms

    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx+shiftone - (x[i][0] - boxlox) * delxinv;
    dy = ny+shiftone - (x[i][1] - boxloy) * delyinv;
    dz = nz+shiftone - (x[i][2] - boxloz) * delzinv;

    compute_rho1d(dx,dy,dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n+nz;
      z0 = rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m+ny;
        y0 = z0*rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l+nx;
          x0 = y0*rho1d[0][l];
          ekx -= x0*vdx_brick[mz][my][mx];
          eky -= x0*vdy_brick[mz][my][mx];
          ekz -= x0*vdz_brick[mz][my][mx];
        }
      }
    }

    // convert E-field to force

    const double qfactor = force->qqrd2e * scale * q[i];
    f[i][0] += qfactor*ekx;
    f[i][1] += qfactor*eky;
    if (slabflag != 2) f[i][2] += qfactor*ekz;
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my ENV particles for ad
------------------------------------------------------------------------- */

void EVB_PPPM::fieldforce_env_ad()
{
  int i,l,m,n,nx,ny,nz,mx,my,mz;
  FFT_SCALAR dx,dy,dz;
  FFT_SCALAR ekx,eky,ekz;
  double s1,s2,s3;
  double sf = 0.0;
  double *prd;

  if (triclinic == 0) prd = domain->prd;
  else prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];

  double hx_inv = nx_pppm/xprd;
  double hy_inv = ny_pppm/yprd;
  double hz_inv = nz_pppm/zprd;

  const double boxlox = boxlo[0];
  const double boxloy = boxlo[1];
  const double boxloz = boxlo[2];

  // loop over my charges, interpolate electric field from nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  // ek = 3 components of E-field on particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;

  int nlocal = atom->nlocal;

  int * is_cplx_atom = evb_engine->complex_atom;

  for (i = 0; i < nlocal; i++) {
    if(is_cplx_atom[i]) continue; // Skip complex atoms

    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx+shiftone - (x[i][0] - boxlox) * delxinv;
    dy = ny+shiftone - (x[i][1] - boxloy) * delyinv;
    dz = nz+shiftone - (x[i][2] - boxloz) * delzinv;

    compute_rho1d(dx,dy,dz);
    compute_drho1d(dx,dy,dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n+nz;
      for (m = nlower; m <= nupper; m++) {
        my = m+ny;
        for (l = nlower; l <= nupper; l++) {
          mx = l+nx;
          ekx += drho1d[0][l]*rho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
          eky += rho1d[0][l]*drho1d[1][m]*rho1d[2][n]*u_brick[mz][my][mx];
          ekz += rho1d[0][l]*rho1d[1][m]*drho1d[2][n]*u_brick[mz][my][mx];
        }
      }
    }
    ekx *= hx_inv;
    eky *= hy_inv;
    ekz *= hz_inv;

    // convert E-field to force and substract self forces

    const double qi = q[i];
    const double qfactor = force->qqrd2e * scale;

    s1 = x[i][0]*hx_inv;
    s2 = x[i][1]*hy_inv;
    s3 = x[i][2]*hz_inv;
    sf = sf_coeff[0]*sin(2*MY_PI*s1);
    sf += sf_coeff[1]*sin(4*MY_PI*s1);
    sf *= 2*qi*qi;
    f[i][0] += qfactor*(ekx*qi - sf);

    sf = sf_coeff[2]*sin(2*MY_PI*s2);
    sf += sf_coeff[3]*sin(4*MY_PI*s2);
    sf *= 2*qi*qi;
    f[i][1] += qfactor*(eky*qi - sf);


    sf = sf_coeff[4]*sin(2*MY_PI*s3);
    sf += sf_coeff[5]*sin(4*MY_PI*s3);
    sf *= 2*qi*qi;
    if (slabflag != 2) f[i][2] += qfactor*(ekz*qi - sf);
  }
}

/* ----------------------------------------------------------------------
   pack own values to buf to send to another proc
------------------------------------------------------------------------- */

void EVB_PPPM::pack_forward_grid(int flag, void *vbuf, int nlist, int *list)
{
  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;
  
  int n = 0;

  if (flag == FORWARD_IK) {
    FFT_SCALAR *xsrc = &vdx_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *ysrc = &vdy_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *zsrc = &vdz_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[n++] = xsrc[list[i]];
      buf[n++] = ysrc[list[i]];
      buf[n++] = zsrc[list[i]];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *src = &u_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      buf[i] = src[list[i]];
  } else if (flag == FORWARD_IK_PERATOM) {
    FFT_SCALAR *esrc = &u_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      if (eflag_atom) buf[n++] = esrc[list[i]];
      if (vflag_atom) {
        buf[n++] = v0src[list[i]];
        buf[n++] = v1src[list[i]];
        buf[n++] = v2src[list[i]];
        buf[n++] = v3src[list[i]];
        buf[n++] = v4src[list[i]];
        buf[n++] = v5src[list[i]];
      }
    }
  } else if (flag == FORWARD_AD_PERATOM) {
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      buf[n++] = v0src[list[i]];
      buf[n++] = v1src[list[i]];
      buf[n++] = v2src[list[i]];
      buf[n++] = v3src[list[i]];
      buf[n++] = v4src[list[i]];
      buf[n++] = v5src[list[i]];
    }
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's own values from buf and set own ghost values
------------------------------------------------------------------------- */

void EVB_PPPM::unpack_forward_grid(int flag, void *vbuf, int nlist, int *list)
{
  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;
  
  int n = 0;

  if (flag == FORWARD_IK) {
    FFT_SCALAR *xdest = &vdx_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *ydest = &vdy_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *zdest = &vdz_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      xdest[list[i]] = buf[n++];
      ydest[list[i]] = buf[n++];
      zdest[list[i]] = buf[n++];
    }
  } else if (flag == FORWARD_AD) {
    FFT_SCALAR *dest = &u_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      dest[list[i]] = buf[i];
  } else if (flag == FORWARD_IK_PERATOM) {
    FFT_SCALAR *esrc = &u_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      if (eflag_atom) esrc[list[i]] = buf[n++];
      if (vflag_atom) {
        v0src[list[i]] = buf[n++];
        v1src[list[i]] = buf[n++];
        v2src[list[i]] = buf[n++];
        v3src[list[i]] = buf[n++];
        v4src[list[i]] = buf[n++];
        v5src[list[i]] = buf[n++];
      }
    }
  } else if (flag == FORWARD_AD_PERATOM) {
    FFT_SCALAR *v0src = &v0_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v1src = &v1_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v2src = &v2_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v3src = &v3_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v4src = &v4_brick[nzlo_out][nylo_out][nxlo_out];
    FFT_SCALAR *v5src = &v5_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++) {
      v0src[list[i]] = buf[n++];
      v1src[list[i]] = buf[n++];
      v2src[list[i]] = buf[n++];
      v3src[list[i]] = buf[n++];
      v4src[list[i]] = buf[n++];
      v5src[list[i]] = buf[n++];
    }
  }
}

/* ----------------------------------------------------------------------
   pack ghost values into buf to send to another proc
------------------------------------------------------------------------- */

void EVB_PPPM::pack_reverse_grid(int flag, void *vbuf, int nlist, int *list)
{
  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;

  if (flag == REVERSE_RHO) {
    FFT_SCALAR *src = &density_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      buf[i] = src[list[i]];
  }
}

/* ----------------------------------------------------------------------
   unpack another proc's ghost values from buf and add to own values
------------------------------------------------------------------------- */

void EVB_PPPM::unpack_reverse_grid(int flag, void *vbuf, int nlist, int *list)
{
  FFT_SCALAR *buf = (FFT_SCALAR *) vbuf;

  if (flag == REVERSE_RHO) {
    FFT_SCALAR *dest = &density_brick[nzlo_out][nylo_out][nxlo_out];
    for (int i = 0; i < nlist; i++)
      dest[list[i]] += buf[i];
  }
}

/* ----------------------------------------------------------------------
   map nprocs to NX by NY grid as PX by PY procs - return optimal px,py 
------------------------------------------------------------------------- */

void EVB_PPPM::procs2grid2d(int nprocs, int nx, int ny, int *px, int *py)
{
  // loop thru all possible factorizations of nprocs
  // surf = surface area of largest proc sub-domain
  // innermost if test minimizes surface area and surface/volume ratio

  int bestsurf = 2 * (nx + ny);
  int bestboxx = 0;
  int bestboxy = 0;

  int boxx,boxy,surf,ipx,ipy;

  ipx = 1;
  while (ipx <= nprocs) {
    if (nprocs % ipx == 0) {
      ipy = nprocs/ipx;
      boxx = nx/ipx;
      if (nx % ipx) boxx++;
      boxy = ny/ipy;
      if (ny % ipy) boxy++;
      surf = boxx + boxy;
      if (surf < bestsurf || 
	  (surf == bestsurf && boxx*boxy > bestboxx*bestboxy)) {
	bestsurf = surf;
	bestboxx = boxx;
	bestboxy = boxy;
	*px = ipx;
	*py = ipy;
      }
    }
    ipx++;
  }
}

/* ----------------------------------------------------------------------
   charge assignment into rho1d
   dx,dy,dz = distance of particle from "lower left" grid point 
------------------------------------------------------------------------- */

void EVB_PPPM::compute_rho1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy,
			     const FFT_SCALAR &dz)
{
  //if (order == 5) {
  if(false) { // disabled from LAMMPS Nov/21
    // order = 5 case, completely unrolled loops
    const double dx2 = dx*dx;
    const double dx3 = dx2*dx;
    const double dx4 = dx2*dx2;
    const double dy2 = dy*dy;
    const double dy3 = dy2*dy;
    const double dy4 = dy2*dy2;
    const double dz2 = dz*dz;
    const double dz3 = dz2*dz;
    const double dz4 = dz2*dz2;
    int k = -2;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
    k = -1;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
    k = 0;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
    k = 1;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
    k = 2;
    rho1d[0][k] = rho_coeff[0][k] + rho_coeff[1][k]*dx + rho_coeff[2][k]*dx2 + rho_coeff[3][k]*dx3 + rho_coeff[4][k]*dx4;
    rho1d[1][k] = rho_coeff[0][k] + rho_coeff[1][k]*dy + rho_coeff[2][k]*dy2 + rho_coeff[3][k]*dy3 + rho_coeff[4][k]*dy4;
    rho1d[2][k] = rho_coeff[0][k] + rho_coeff[1][k]*dz + rho_coeff[2][k]*dz2 + rho_coeff[3][k]*dz3 + rho_coeff[4][k]*dz4;
  } else {
    // general case
    FFT_SCALAR r1, r2, r3;
    for (int k = (1-order)/2; k <= order/2; ++k) {
      r1 = r2 = r3 = ZEROF;
      for (int l = order-1; l >= 0; --l) {
        r1 = rho_coeff[l][k] + r1 * dx;
        r2 = rho_coeff[l][k] + r2 * dy;
        r3 = rho_coeff[l][k] + r3 * dz;
      }
      rho1d[0][k] = r1;
      rho1d[1][k] = r2;
      rho1d[2][k] = r3;
    }
  }
}

/* ----------------------------------------------------------------------
   charge assignment into drho1d
   dx,dy,dz = distance of particle from "lower left" grid point
------------------------------------------------------------------------- */

void EVB_PPPM::compute_drho1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy,
                          const FFT_SCALAR &dz)
{
  if (order == 5) {
    // order = 5 case, completely unrolled loops
    const double dx2 = dx*dx;
    const double dx3 = dx2*dx;
    const double dy2 = dy*dy;
    const double dy3 = dy2*dy;
    const double dz2 = dz*dz;
    const double dz3 = dz2*dz;

    int k = -2;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
    k = -1;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
    k = 0;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
    k = 1;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
    k = 2;
    drho1d[0][k] = drho_coeff[0][k] + drho_coeff[1][k]*dx + drho_coeff[2][k]*dx2 + drho_coeff[3][k]*dx3;
    drho1d[1][k] = drho_coeff[0][k] + drho_coeff[1][k]*dy + drho_coeff[2][k]*dy2 + drho_coeff[3][k]*dy3;
    drho1d[2][k] = drho_coeff[0][k] + drho_coeff[1][k]*dz + drho_coeff[2][k]*dz2 + drho_coeff[3][k]*dz3;
  } else {
    FFT_SCALAR r1,r2,r3;
    for (int k = (1-order)/2; k <= order/2; k++) {
      r1 = r2 = r3 = ZEROF;
      
      for (int l = order-2; l >= 0; l--) {
	r1 = drho_coeff[l][k] + r1*dx;
	r2 = drho_coeff[l][k] + r2*dy;
	r3 = drho_coeff[l][k] + r3*dz;
      }
      drho1d[0][k] = r1;
      drho1d[1][k] = r2;
      drho1d[2][k] = r3;
    }
  }
}

/* ----------------------------------------------------------------------
   generate coeffients for the weight function of order n

              (n-1)
  Wn(x) =     Sum    wn(k,x) , Sum is over every other integer
           k=-(n-1)
  For k=-(n-1),-(n-1)+2, ....., (n-1)-2,n-1
      k is odd integers if n is even and even integers if n is odd
              ---
             | n-1
             | Sum a(l,j)*(x-k/2)**l   if abs(x-k/2) < 1/2
  wn(k,x) = <  l=0
             |
             |  0                       otherwise
              ---
  a coeffients are packed into the array rho_coeff to eliminate zeros
  rho_coeff(l,((k+mod(n+1,2))/2) = a(l,k) 
------------------------------------------------------------------------- */

void EVB_PPPM::compute_rho_coeff()
{
  int j,k,l,m;
  FFT_SCALAR s;

  FFT_SCALAR **a;
  memory->create2d_offset(a,order,-order,order,"EVB_PPPM:a");

  for (k = -order; k <= order; k++) 
    for (l = 0; l < order; l++)
      a[l][k] = 0.0;
        
  a[0][0] = 1.0;
  for (j = 1; j < order; j++) {
    for (k = -j; k <= j; k += 2) {
      s = 0.0;
      for (l = 0; l < j; l++) {
	a[l+1][k] = (a[l][k+1]-a[l][k-1]) / (l+1);
#ifdef FFT_SINGLE
	s += powf(0.5,(float) l+1) *
	  (a[l][k-1] + powf(-1.0,(float) l) * a[l][k+1]) / (l+1);
#else
	s += pow(0.5,(double) l+1) * 
	  (a[l][k-1] + pow(-1.0,(double) l) * a[l][k+1]) / (l+1);
#endif
      }
      a[0][k] = s;
    }
  }

  m = (1-order)/2;
  for (k = -(order-1); k < order; k += 2) {
    for (l = 0; l < order; l++)
      rho_coeff[l][m] = a[l][k];
    for (l = 1; l < order; l++)
      drho_coeff[l-1][m] = l*a[l][k];
    m++;
  }

  memory->destroy2d_offset(a,-order);
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if 
   adequate empty space is left between repeating slabs (J. Chem. Phys. 
   111, 3155).  Slabs defined here to be parallel to the xy plane. 
------------------------------------------------------------------------- */

void EVB_PPPM::slabcorr_cplx()
{
  // compute local cplx contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

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
  
  energy += qqrd2e * e_slabcorr / comm->nprocs;

  // add on force corrections

  double ffact = -4.0 * MY_PI * qqrd2e / volume;
  double **f = atom->f;

  for(int i=0; i<nlocal; i++) f[i][2] += ffact * q[i] * (dipole_all - qsum*x[i][2]);
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

void EVB_PPPM::slabcorr_exch()
{
  // compute local cplx contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  int nlocal_cplx = evb_engine->evb_complex->nlocal_cplx;
  int* cplx_list = evb_engine->evb_complex->cplx_list;

  double dipole_cplx    = 0.0;
  double dipole_r2_cplx = 0.0;
  double dipole_exch    = 0.0;
  double dipole_r2_exch = 0.0;
  double qsum_exch      = 0.0;
  for(int i=0; i<nlocal_cplx; i++) {
    int iatm = cplx_list[i];
    double qx = q[iatm] * x[iatm][2];

    dipole_cplx    += qx;
    dipole_r2_cplx += qx * x[iatm][2];

    if(is_exch_chg[iatm]) {
      dipole_exch    += qx;
      dipole_r2_exch += qx * x[iatm][2];
      qsum_exch      += q[iatm];
    }
  }
  
  // sum local contributions to get global dipole moments
  
  double tmp = 0.0;
  MPI_Allreduce(&dipole_cplx, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  double dipole_all = dipole_env + tmp;

  tmp = 0.0;
  MPI_Allreduce(&dipole_r2_cplx, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  double dipole_r2_all = dipole_r2_env + tmp;
  
  tmp = dipole_exch;
  dipole_exch = 0.0;
  MPI_Allreduce(&tmp, &dipole_exch, 1, MPI_DOUBLE, MPI_SUM, world);

  tmp = dipole_r2_exch;
  dipole_r2_exch = 0.0;
  MPI_Allreduce(&tmp, &dipole_r2_exch, 1, MPI_DOUBLE, MPI_SUM, world);

  tmp = qsum_exch;
  qsum_exch = 0.0;
  MPI_Allreduce(&tmp, &qsum_exch, 1, MPI_DOUBLE, MPI_SUM, world);

  // compute corrections
  double dipole_mexch    = dipole_all - dipole_exch;
  double dipole_r2_mexch = dipole_r2_all - dipole_r2_exch;
  double qsum_mexch      = qsum - qsum_exch;

  double dip_int    = dipole_all * dipole_all - dipole_mexch * dipole_mexch - dipole_exch * dipole_exch;
  double dip_r2_int = qsum * dipole_r2_all - (qsum_mexch * dipole_r2_mexch) - (qsum_exch * dipole_r2_exch);
  double qsum_int   = qsum * qsum - qsum_mexch * qsum_mexch - qsum_exch * qsum_exch;

  const double e_slabcorr = MY_2PI * (dip_int - dip_r2_int - qsum_int * zprd * zprd / 12.0) / volume;

  off_diag_energy += qqrd2e * e_slabcorr / comm->nprocs;

  // add on force corrections

  // dip_int  = dipole_all - dipole_mexch - dipole_exch; // dip_int is identically zero
  // qsum_int = qsum - qsum_mexch - qsum_exch; // qsum_int is identically zero

  // double ffact = -4.0 * MY_PI * qqrd2e / volume;
  // double **f = atom->f;

  // for(int i=0; i<nlocal; i++) f[i][2] += ffact * q[i] * (dip_int - qsum_int * x[i][2]);
}

/* ----------------------------------------------------------------------
   perform and time the 1d FFTs required for N timesteps
------------------------------------------------------------------------- */

int EVB_PPPM::timing_1d(int n, double &time1d)
{
  double time1,time2;

  for (int i = 0; i < 2*nfft_both; i++) work1[i] = ZEROF;

  MPI_Barrier(world);
  time1 = MPI_Wtime();

  for (int i = 0; i < n; i++) {
    fft1->timing1d(work1,nfft_both,1);
    fft2->timing1d(work1,nfft_both,-1);
    if (differentiation_flag != 1) {
      fft2->timing1d(work1,nfft_both,-1);
      fft2->timing1d(work1,nfft_both,-1);
    }
  }

  MPI_Barrier(world);
  time2 = MPI_Wtime();
  time1d = time2 - time1;

  if (differentiation_flag) return 2;
  return 4;
}

/* ----------------------------------------------------------------------
   perform and time the 3d FFTs required for N timesteps
------------------------------------------------------------------------- */

int EVB_PPPM::timing_3d(int n, double &time3d)
{
  double time1,time2;

  for (int i = 0; i < 2*nfft_both; i++) work1[i] = ZEROF;

  MPI_Barrier(world);
  time1 = MPI_Wtime();

  for (int i = 0; i < n; i++) {
    fft1->compute(work1,work1,1);
    fft2->compute(work1,work1,-1);
    if (differentiation_flag != 1) {
      fft2->compute(work1,work1,-1);
      fft2->compute(work1,work1,-1);
    }
  }

  MPI_Barrier(world);
  time2 = MPI_Wtime();
  time3d = time2 - time1;

  if (differentiation_flag) return 2;
  return 4;
}

/* ----------------------------------------------------------------------
   memory usage of local arrays 
------------------------------------------------------------------------- */

double EVB_PPPM::memory_usage()
{
  double bytes = nmax*3 * sizeof(double);
  int nbrick = (nxhi_out-nxlo_out+1) * (nyhi_out-nylo_out+1) * 
    (nzhi_out-nzlo_out+1);
  if (differentiation_flag == 1) {
    bytes += 2 * nbrick * sizeof(FFT_SCALAR);
  } else {
    bytes += 4 * nbrick * sizeof(FFT_SCALAR);
  }
  bytes += 6 * nfft_both * sizeof(double);
  bytes += nfft_both * sizeof(double);
  bytes += nfft_both*5 * sizeof(FFT_SCALAR);
  
  bytes += (double)(ngc_buf1 + ngc_buf2) * npergrid * sizeof(FFT_SCALAR);  
  
  return bytes;
}
