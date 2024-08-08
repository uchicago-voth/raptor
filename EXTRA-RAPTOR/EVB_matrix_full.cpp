/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Wim R. Cardoen and Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "memory.h"
#include "error.h"
#include "update.h"
#include "atom.h"
#include "atom_vec.h"
#include "neigh_list.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "comm.h"
#include "universe.h"

#include "EVB_engine.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_chain.h"
#include "EVB_complex.h"
#include "EVB_reaction.h"
#include "EVB_list.h"
#include "EVB_matrix_full.h"
#include "EVB_repul.h"
#include "EVB_offdiag.h"
#include "EVB_kspace.h"
#include "mp_verlet.h"

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);	\
  a[k][l]=h+s*(g-h*tau);

// ** AWGL ** //
#if defined (_OPENMP)
#include <omp.h>
#endif


/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

EVB_MatrixFull::EVB_MatrixFull(LAMMPS *lmp, EVB_Engine *engine) : EVB_Matrix(lmp,engine)
{

}

/* ---------------------------------------------------------------------- */

EVB_MatrixFull::~EVB_MatrixFull()
{

}

/* ---------------------------------------------------------------------- */

void EVB_MatrixFull::setup()
{
  int max1 = 0, max2 = 0;
    
  for(int i=0; i<evb_engine->ncomplex; i++) {
    if(evb_engine->all_complex[i]->nstate>max1) max1 = evb_engine->all_complex[i]->nstate;
    if(evb_engine->all_complex[i]->nextra_coupling>max2) max2 = evb_engine->all_complex[i]->nextra_coupling;
  }
  
  natom = evb_engine->natom;
  
  bool inc_atom = false;
  bool inc_state = false;
  bool inc_extra = false;
  
  while (natom>max_atom) { max_atom  += 100;  inc_atom  = true; }
  while (max1>max_state) { max_state += 20;   inc_state = true; }
  while (max2>max_extra) { max_extra += 20;   inc_extra = true; }
  
  int nt = 1;
#if defined(_OPENMP)
#define _MAX_ATOM max_atom*comm->nthreads
  nt = comm->nthreads;
#else
#define _MAX_ATOM max_atom
#endif
  
#ifdef BGQ
  double scale = sizeof(double) / 1024.0 / 1024.0;
  if(universe->me==0) {
    fprintf(stdout,"max_atom= %i  max_state= %i  max_extra= %i  nt= %i  _MAX_ATOM= %i\n",max_atom,max_state,max_extra,nt,_MAX_ATOM);
    fprintf(stdout,"f_extra_coupling: %i x %i x %i = %f MB.\n",max_extra,_MAX_ATOM,3,max_extra*_MAX_ATOM*3 * scale);
    fprintf(stdout,"f_env:                 %i x %i = %f MB.\n",_MAX_ATOM,3,_MAX_ATOM*3 * scale);
    fprintf(stdout,"f_diagonal:       %i x %i x %i = %f MB.\n",max_state,_MAX_ATOM,3,max_state*_MAX_ATOM*3 * scale);
    fprintf(stdout,"f_off_diagonal:   %i x %i x %i = %f MB.\n",max_state-1,_MAX_ATOM,3,(max_state-1)*_MAX_ATOM*3 * scale);
    fprintf(stdout,"Total= %f\n",(max_extra + 2*max_state)*(_MAX_ATOM*3) * scale);
  }
  
  if(universe->me==0) {
    fprintf(stdout,"\nget_memory() in full_matrix->setup just before force allocation.\n");
    evb_engine->get_memory();
  }
#endif  
  
  // Reallocate memory space
  
  if(inc_extra || inc_atom) memory->grow(f_extra_coupling,max_extra,_MAX_ATOM, 3,"EVB_MatrixFull:f_extra_coupling");
  
  if(inc_atom) memory->grow(f_env,_MAX_ATOM,3,"EVB_MatrixFull:f_env");
  
  if(inc_state || inc_atom) {
    memory->grow(f_diagonal,max_state,_MAX_ATOM,3,"EVB_MatrixFull:f_diagonal");
    memory->grow(f_off_diagonal,max_state-1,_MAX_ATOM,3,"EVB_MatrixFull:f_off_diagonal");
  }
  
#ifdef BGQ
  if(universe->me==0) {
    fprintf(stdout,"\nget_memory() in full_matrix->setup just after force allocation.\n");
    evb_engine->get_memory();
  }
#endif  
  
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixFull::clear(bool eflag, bool vflag, bool fflag)
{
  nstate = evb_complex->nstate;
  nextra = evb_complex->nextra_coupling;
  
  size_e = EDIAG_NITEM;
  for(int i=0; i<nstate; i++) { e_diagonal[i]=energy+size_e; size_e+=EDIAG_NITEM; }
  e_repulsive = energy+size_e;
  size_e+=nstate;
  size_ediag = size_e - EDIAG_NITEM;
  
  if(eflag) {
    memset(energy,0,sizeof(double)*size_e);
    if(nstate>1) memset(e_offdiag,0,sizeof(double)*(nstate-1)*EOFF_NITEM);
    if(nextra>0) memset(e_extra,0,sizeof(double)*nextra*EOFF_NITEM);
  }
  
  if(vflag) {
    if(nstate>0) memset(v_diagonal,0,sizeof(double)*nstate*6); 
    if(nstate>1) memset(v_offdiag,0,sizeof(double)*(nstate-1)*6);
    if(nextra>0) memset(v_extra,0,sizeof(double)*nextra*6);
  }
  
  if(fflag) {
    if(natom>0) memset(&(f_env[0][0]),0,sizeof(double)*3*natom);
    if(natom>0) for(int i=0; i<nstate; i++) {
	memset(&(f_diagonal[i][0][0]),0,sizeof(double)*3*natom);
	if(i<nstate-1) memset(&(f_off_diagonal[i][0][0]),0,sizeof(double)*3*natom);
      }
    
    if(nextra>0 && natom>0) for(int i=0; i<nextra; i++)
			      memset(&(f_extra_coupling[i][0][0]),0,sizeof(double)*3*natom);
    
#if defined (_OPENMP)
    evb_engine->Force_Clear(2);
#endif
  }
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixFull::clear(bool eflag, bool vflag, bool fflag, bool envfflag)
{
  nstate = evb_complex->nstate;
  nextra = evb_complex->nextra_coupling;
  
  size_e = EDIAG_NITEM;
  for(int i=0; i<nstate; i++) { e_diagonal[i]=energy+size_e; size_e+=EDIAG_NITEM; }
  e_repulsive = energy+size_e;
  size_e+=nstate;
  size_ediag = size_e - EDIAG_NITEM;
  
  if(eflag) {
    memset(energy,0,sizeof(double)*size_e);
    if(nstate>1) memset(e_offdiag,0,sizeof(double)*(nstate-1)*EOFF_NITEM);
    if(nextra>0) memset(e_extra,0,sizeof(double)*nextra*EOFF_NITEM);
  }
  
  if(vflag) {
    if(nstate>0) memset(v_diagonal,0,sizeof(double)*nstate*6); 
    if(nstate>1) memset(v_offdiag,0,sizeof(double)*(nstate-1)*6);
    if(nextra>0) memset(v_extra,0,sizeof(double)*nextra*6);
  }
  
  if(fflag) {
    if(natom>0 && envfflag) memset(&(f_env[0][0]),0,sizeof(double)*3*natom);
    if(natom>0) for(int i=0; i<nstate; i++) {
	memset(&(f_diagonal[i][0][0]),0,sizeof(double)*3*natom);
	if(i<nstate-1) memset(&(f_off_diagonal[i][0][0]),0,sizeof(double)*3*natom);
      }
    
    if(nextra>0 && natom>0) for(int i=0; i<nextra; i++)
			      memset(&(f_extra_coupling[i][0][0]),0,sizeof(double)*3*natom);
    
#if defined (_OPENMP)
    evb_engine->Force_Clear(2);
#endif
  }
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixFull::compute_hellmann_feynman(int vflag)
{
#if defined(_OPENMP)
  // ** AWGL ** //
  compute_hellmann_feynman_omp(vflag);
  return;
#endif

  double **f_des = atom->f;
  double *v_des = evb_engine->virial;
  double *Cs = evb_complex->Cs;
  double *Cs2= evb_complex->Cs2;
  int natom_cplx = evb_complex->natom_cplx;
  int *cplx_list = evb_complex->cplx_list;

  GET_OFFDIAG_EXCH(evb_complex);

  if(natom==0) return;

  for(int i=0; i<natom; i++) {
    f_des[i][0]=f_env[i][0];
    f_des[i][1]=f_env[i][1];
    f_des[i][2]=f_env[i][2];
  }
  
  if(vflag) for(int i = 0; i < 6; i++) v_des[i] = v_env[i];
  
  /*** Diagonal elements ***/

  for (int i=0;i<nstate;i++) {
    double **f_src = f_diagonal[i];
    
    for(int j=0; j<natom; j++) {
      f_des[j][0]+=f_src[j][0]*Cs2[i];
      f_des[j][1]+=f_src[j][1]*Cs2[i];
      f_des[j][2]+=f_src[j][2]*Cs2[i];
    }
    
    if(vflag) for(int j=0; j<6; j++) v_des[j] += v_diagonal[i][j] * Cs2[i];
  }

  /*** Off-Diagonal elements ***/
  
  int *parent = evb_complex->parent_id;
  
  for (int i=0; i<nstate-1; i++) {
    double **f_src = f_off_diagonal[i];
    double C = 2*Cs[i+1]*Cs[parent[i+1]];
    
    if(evb_engine->mp_verlet && evb_engine->mp_verlet->is_master==0) C*=e_offdiag[i][EOFF_ARQ];
    
    for(int j=0; j<natom; j++) {
      f_des[j][0]+=f_src[j][0]*C;
      f_des[j][1]+=f_src[j][1]*C;
      f_des[j][2]+=f_src[j][2]*C;
    }
    
    if(vflag) for(int j = 0; j < 6; j++) v_des[j] += v_offdiag[i][j] * C;
  }
  
  /*** Extra couplings ***/
  
  for (int i=0; i<nextra; i++) {
    double **f_src = f_extra_coupling[i];
    double C = 2*Cs[extra_j[i]]*Cs[extra_i[i]];
    
    if(evb_engine->mp_verlet && evb_engine->mp_verlet->is_master==0) C*=e_extra[i][EOFF_ARQ];
    
    for(int j=0; j<natom; j++) {
      f_des[j][0]+=f_src[j][0]*C;
      f_des[j][1]+=f_src[j][1]*C;
      f_des[j][2]+=f_src[j][2]*C;
    }
    
    if(vflag) for(int j = 0; j < 6; j++) v_des[j] += v_extra[i][j] * C;
  }
}


/* ---------------------------------------------------------------------- */

void EVB_MatrixFull::compute_hellmann_feynman_omp(int vflag)
{
  // ** AWGL : OpenMP version of compute_hellman_feynman ** //

  double **f_des = atom->f;
  double *v_des = evb_engine->virial;
  double *Cs = evb_complex->Cs;
  double *Cs2= evb_complex->Cs2;
  int natom_cplx = evb_complex->natom_cplx;
  int *cplx_list = evb_complex->cplx_list;

  GET_OFFDIAG_EXCH(evb_complex);

  if(natom==0) return;

  double *fd = &(f_des[0][0]);
  double *fe = &(f_env[0][0]);
  memcpy(fd, fe, sizeof(double)*3*natom);

  /*** Diagonal elements ***/

#if defined(_OPENMP)
#pragma omp parallel default(none)\
 shared(Cs, Cs2, f_des, vflag, extra_i, extra_j)
#endif
  {
#if defined(_OPENMP)
    // each thread works on a fixed chunk of atoms.
    const int nthreads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int jnum = natom;
    const int jdelta = 1 + jnum/nthreads;
    const int jfrom = tid*jdelta;
    const int jto = ((jfrom + jdelta) > jnum) ? jnum : jfrom + jdelta;
#else
    const int jfrom = 0;
    const int jto = natom;
#endif

    // swap order, better parallelism
    for(int j=jfrom; j<jto; j++) {
      double f_dj0 = 0.0;
      double f_dj1 = 0.0;
      double f_dj2 = 0.0;
      
      for (int i=0; i<nstate; i++) {
        double * const * const f_src = f_diagonal[i];
        f_dj0 += f_src[j][0]*Cs2[i];
        f_dj1 += f_src[j][1]*Cs2[i];
        f_dj2 += f_src[j][2]*Cs2[i];
      }
      
      f_des[j][0] += f_dj0;
      f_des[j][1] += f_dj1;
      f_des[j][2] += f_dj2;
      
      /*** Off-Diagonal elements ***/
      
      const bool special_case = evb_engine->mp_verlet && evb_engine->mp_verlet->is_master==0;
      int * const parent = evb_complex->parent_id;
      
      f_dj0 = 0.0;
      f_dj1 = 0.0;
      f_dj2 = 0.0;
      
      if(special_case) {
	for (int i=0; i<nstate-1; ++i) {
	  double * const * const f_src = f_off_diagonal[i];
	  const double C = Cs[i+1]*Cs[parent[i+1]] * e_offdiag[i][EOFF_ARQ];
	  f_dj0 += f_src[j][0]*C;
	  f_dj1 += f_src[j][1]*C;
	  f_dj2 += f_src[j][2]*C;
	}
      } else {
	for (int i=0; i<nstate-1; ++i) {
	  double * const * const f_src = f_off_diagonal[i];
	  const double C = Cs[i+1]*Cs[parent[i+1]];
	  f_dj0 += f_src[j][0]*C;
	  f_dj1 += f_src[j][1]*C;
	  f_dj2 += f_src[j][2]*C;
	}
      }
      
      f_des[j][0] += 2.0 * f_dj0;
      f_des[j][1] += 2.0 * f_dj1;
      f_des[j][2] += 2.0 * f_dj2;
      
      /*** Extra couplings ***/
      
      f_dj0 = 0.0;
      f_dj1 = 0.0;
      f_dj2 = 0.0;
      
      if(special_case) {
	for (int i=0; i<nextra; i++) {
	  double * const * const f_src = f_extra_coupling[i];
	  const double C = Cs[extra_j[i]]*Cs[extra_i[i]] * e_extra[i][EOFF_ARQ];
	  f_dj0 += f_src[j][0]*C;
	  f_dj1 += f_src[j][1]*C;
	  f_dj2 += f_src[j][2]*C;
	}
      } else {
	for (int i=0; i<nextra; i++) {
	  double * const * const f_src = f_extra_coupling[i];
	  const double C = Cs[extra_j[i]]*Cs[extra_i[i]];
	  f_dj0 += f_src[j][0]*C;
	  f_dj1 += f_src[j][1]*C;
	  f_dj2 += f_src[j][2]*C;
	}
      }
      
      f_des[j][0] += 2.0 * f_dj0;
      f_des[j][1] += 2.0 * f_dj1;
      f_des[j][2] += 2.0 * f_dj2;
      
    }
    
  } // close OpenMP bracket
  
  if(vflag) {
    // ENV
    for(int i = 0; i < 6; i++) v_des[i] = v_env[i];

    // Diagonals
    for(int i=0; i<nstate; i++) for(int k=0; k<6; k++) v_des[k] += v_diagonal[i][k] * Cs2[i];
    
    // Off-diagonals
    int * const parent = evb_complex->parent_id;
    for(int i=0; i<nstate-1; i++) {
      double C = 2.0 * Cs[i+1] * Cs[parent[i+1]];
      if(evb_engine->mp_verlet && evb_engine->mp_verlet->is_master==0) C*=e_offdiag[i][EOFF_ARQ];

      for(int k = 0; k < 6; k++) v_des[k] += v_offdiag[i][k] * C;
    }

    // Extra
    for(int i=0; i<nextra; i++) {
      double C = 2.0 * Cs[extra_j[i]] * Cs[extra_i[i]];
      if(evb_engine->mp_verlet && evb_engine->mp_verlet->is_master==0) C*=e_extra[i][EOFF_ARQ];
      for(int k = 0; k < 6; k++) v_des[k] += v_extra[i][k] * C;
    }
  }
}
