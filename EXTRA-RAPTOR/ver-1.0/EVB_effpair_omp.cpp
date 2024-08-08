/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "memory.h"
#include "comm.h"
#include "error.h"
#include "update.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "kspace.h"

#include "EVB_engine.h"
#include "EVB_complex.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_chain.h"
#include "EVB_effpair.h"
#include "EVB_list.h"
#include "EVB_offdiag.h"
#include "EVB_matrix.h"
#include "EVB_timer.h"

#define _CRACKER_PAIR_LJ_CHARMM_COUL_LONG
#define _CRACKER_PAIR_LJ_CUT_COUL_LONG
#if defined (_OPENMP)
#include <omp.h>
#define _CRACKER_PAIR_LJ_CUT_COUL_LONG_OMP
#endif
  #include "EVB_cracker.h"
#undef _CRACKER_PAIR_LJ_CHARMM_COUL_LONG
#undef _CRACKER_PAIR_LJ_CUT_COUL_LONG
#if defined (_OPENMP)
#undef _CRACKER_PAIR_LJ_CUT_COUL_LONG_OMP
#endif
//#include "pair_table_lj_cut_coul_long.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

void EVB_EffPair::pre_compute_omp()
{
  // realloc memory for cpl_pair_list labeling
  if(evb_list->npair_cpl > max_pair) {
    max_pair = evb_list->npair_cpl;
    
    r2inv = (double*) memory->srealloc(r2inv,sizeof(double)*max_pair,"EVB_EffPair:r2inv");
    r6inv = (double*) memory->srealloc(r6inv,sizeof(double)*max_pair,"EVB_EffPair:r6inv");
    pre_ecoul = (double*) memory->srealloc(pre_ecoul,sizeof(double)*max_pair,"EVB_EffPair:pre_ecoul");
    pre_fcoul = (double*) memory->srealloc(pre_fcoul,sizeof(double)*max_pair,"EVB_EffPair:pre_fcoul");
    
    pre_ecoul_exch = (double*) memory->srealloc(pre_ecoul_exch,sizeof(double)*max_pair,"EVB_EffPair:pre_ecoul_exch");
    pre_fcoul_exch = (double*) memory->srealloc(pre_fcoul_exch,sizeof(double)*max_pair,"EVB_EffPair:pre_fcoul_exch");
    
    cut_coul = (bool*) memory->srealloc(cut_coul,sizeof(bool)*max_pair,"EVB_EffPair:cut_coul");
    cut_lj = (bool*) memory->srealloc(cut_lj,sizeof(bool)*max_pair,"EVB_EffPair:cut_lj");
  }
   
  const double g_ewald = force->kspace->g_ewald;
  const double qqrd2e = force->qqrd2e;
  const double * const * const x = atom->x;
  
  NeighList *list = evb_engine->get_pair_list();
  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  
  int i,j,id;

  // The value is taken from the first off-diagonal type and assumed to be the same for all.
  // This needs to be generalized for multiple types of off-diagonals
  const int is_Vij_ex = evb_engine->all_offdiag[0]->is_Vij_ex;
  const double cut = sqrt(cut_coulsq);
  const double cut_cgis = 0.81 * cut;
  const double cut_cgis2 = cut_cgis * cut_cgis;
  const double rcutinv = 1.0 / cut;

  const double onehalf  = 1.0 / 2.0;
  const double onethird = 1.0 / 3.0;

  const double A = -1.0 / (cut * cut * cut_cgis * (cut - cut_cgis));
  const double B = -1.0 / (cut * cut * cut_cgis);
  const double C =  1.0 / (cut * cut);

  const double A3 = A * onethird;
  const double B2 = B * onehalf;

  const double dr_cgis = cut_cgis - cut;
  const double Bdr2_cgis = B * dr_cgis * dr_cgis * onehalf;
  const double Adr3_cgis = A * dr_cgis * dr_cgis * dr_cgis * onethird;
  const double cgis_const = Adr3_cgis + Bdr2_cgis + C * dr_cgis - rcutinv;

#if defined (_OPENMP)
#pragma omp parallel default(none)\
  shared(ilist, firstneigh, numneigh, stdout) \
  private(i,j,id)
#endif
  {
    
#if defined (_OPENMP)
    const int tid = omp_get_thread_num();
    const int dn = inum / comm->nthreads + 1;
    const int istart = tid * dn;
    int ii = istart + dn;
    if(ii > inum) ii = inum;
    const int iend = ii;
#else
    const int tid = 0;
    const int istart = 0;
    const int iend = inum;
#endif

    id = 0;
    for(i=0; i<iend; i++) {
      const int atomi = ilist[i];
      const int jnum = numneigh[atomi];

      if(i < istart) {
	id+= jnum;
	continue;
      }

      const int *jlist = firstneigh[atomi];
      
      const double xi = x[atomi][0];
      const double yi = x[atomi][1];
      const double zi = x[atomi][2];

      for(j=0; j<jnum; j++) {
	const int atomj = jlist[j] & NEIGHMASK;
	
	const double dx = xi - x[atomj][0];
	const double dy = yi - x[atomj][1];
	const double dz = zi - x[atomj][2];
      	const double r2 = dx*dx + dy*dy + dz*dz;

	r2inv[id] = 1.0 / r2;
	
	if(r2>cut_coulsq) cut_coul[id]=true;
	else {
	  cut_coul[id]=false;
	  
	  const double r = sqrt(r2);

	  const double EWALD_F = 1.12837917;
	  const double EWALD_P = 0.3275911;
	  const double EA1 =      0.254829592;
	  const double EA2 =     -0.284496736;
	  const double EA3 =      1.421413741;
	  const double EA4 =     -1.453152027;
	  const double EA5 =      1.061405429;

	  const double grij = g_ewald * r;
	  const double expm2 = exp(-grij*grij);
	  const double t = 1.0 / (1.0 + EWALD_P*grij);
	  const double erfc = t * (EA1+t*(EA2+t*(EA3+t*(EA4+t*EA5)))) * expm2;
	  const double prefactor = qqrd2e / r;
	  pre_ecoul[id] = prefactor * erfc;
	  pre_fcoul[id] = prefactor * (erfc+EWALD_F*grij*expm2);
	  
	  if(is_Vij_ex==1) {
	    pre_ecoul_exch[id] = pre_ecoul[id];
	    pre_fcoul_exch[id] = pre_fcoul[id];
	    
	  } else if(is_Vij_ex==4 || is_Vij_ex==5) {
	    const double rinv = 1.0 / r;
	    
	    pre_ecoul_exch[id] = rinv;
	    pre_fcoul_exch[id] = r2inv[id];
	    
	    if(r < cut_cgis) { 
	      pre_ecoul_exch[id] += cgis_const - B2 * (r2 - cut_cgis2);
	      pre_fcoul_exch[id] += B * r;
	    } else { 
	      const double dr = r - cut;
	      const double dr2 = dr * dr;
	      pre_ecoul_exch[id] += -rcutinv + A3 * dr2 * dr + B2 * dr2 + C * dr;
	      pre_fcoul_exch[id] += -A * dr2 - B * dr - C;
	    }
	    
	    pre_ecoul_exch[id] *= qqrd2e;
	    pre_fcoul_exch[id] *= qqrd2e * r;
	    
	  } else if(is_Vij_ex == 0) {
	    pre_ecoul_exch[id] = 0.0;
	    pre_fcoul_exch[id] = 0.0;
	  } else error->one(FLERR,"r-space approx. for off-diagonal not recognized.");
	}
	
	if(r2>cut_ljsq) cut_lj[id]=true;
	else {
	  cut_lj[id]=false;
	  r6inv[id] = r2inv[id]*r2inv[id]*r2inv[id];
	}
	
	id++;
      } // for(j<jnum)
    } // for(i<inum)
  } // openmp for

}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_vdw_eff_omp()
{
  const double *Cs2 = evb_complex->Cs2;
  const int *cplx_list = evb_complex->cplx_list;
  
  const int natp = evb_type->natp;
  const int *atp_list = evb_type->atp_list;

  int i, j, t;

#if defined (_OPENMP)
#pragma omp parallel default(none)\
  shared(Cs2, cplx_list, atp_list)\
  private(i, j, t)
#endif
  {
#if defined (_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for(i=0; i<evb_complex->natom_cplx; i++) {
      const int id = cplx_list[i];
      
      memset(lj1[id],0,sizeof(double)*natp);
      memset(lj2[id],0,sizeof(double)*natp);
      
      for(j=0; j<evb_complex->nstate; j++) {
	const int* type = evb_complex->status[j].type;
	const int type_I = type[i];
	const double c = Cs2[j];
	
	for(t=0; t<natp; t++) { 
	  const int indx = atp_list[t];
	  lj1[id][t] += c * ptrLJ1[type_I][indx];
	  lj2[id][t] += c * ptrLJ2[type_I][indx];
	}
      }
    } // for(i<natom_cplx)
  } // omp parallel

}
