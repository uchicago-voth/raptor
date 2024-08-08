/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Adrian Lange
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "EVB_complex.h"
#include "EVB_engine.h"
#include "EVB_type.h"
#include "EVB_offdiag_pt.h"
#include "EVB_kspace.h"
#include "EVB_source.h"

#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "universe.h"
#include "update.h"
#include "mp_verlet.h"
#include "math_const.h"

#define _CRACKER_PAIR_LJ_CUT_COUL_CUT
#define _CRACKER_PAIR_LJ_CHARMM_COUL_CHARMM
  #include "EVB_cracker.h"
#undef _CRACKER_PAIR_LJ_CUT_COUL_CUT
#undef _CRACKER_PAIR_LJ_CHARMM_COUL_CUT

#define SMALL 0.000001

// ** AWGL ** // 
#if defined(_OPENMP)
#include <omp.h>
#endif

#define DATOM 0
#define AATOM 1
#define HATOM 2

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */
/*
   AWGL: OpenMP threaded routines
*/
/* ---------------------------------------------------------------------- */
double EVB_OffDiag::exch_chg_long_omp(int vflag)
{
  // If my proc doesn't have an exch chg, exit b/c there's nothing to do here
  if (!(evb_engine->has_exch_chg)) return 0.0;

  int newton_pair = force->newton_pair;
  if(newton_pair){
    if(vflag){ return exch_chg_long_omp_eval<1,1>();}
    else     { return exch_chg_long_omp_eval<0,1>();}
  } else {
    if(vflag){ return exch_chg_long_omp_eval<1,0>();}
    else     { return exch_chg_long_omp_eval<0,0>();}
  }
}

template <int VFLAG, int NEWTON_PAIR>
double EVB_OffDiag::exch_chg_long_omp_eval()
{
    const double g_ewald = force->kspace->g_ewald;
    const double qqrd2e = force->qqrd2e;
    const int nlocal = atom->nlocal;
    
    double ene = 0.0;
    int itmp;
    double *p_cutoff = (double *) force->pair->extract((char*)("cut_coul"),itmp);
    cut = (*p_cutoff);
    cut_sq = cut*cut;

    const double * const q = atom->q; 
    const double * const * const x = atom->x;
    
    NeighList *list = evb_engine->get_pair_list();
    const int inum = list->inum;
    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    int i, j; 

    // NOTE: This is the *original* force array that has extra allocated space for threads. 
    //       It should be empty! We don't empty it in this routine b/c Force_Reduce will pick it up. 
    //       Force_Reduce will also sum it into where atom->f currently points
    double **lf = evb_engine->lmp_f; 

    
    // NOTE: The criticals for VFLAG are me being lazy. But, I don't think we usually need the virial,
    //       so it shouldn't matter too much, at least for now.

    // If full neighbor list, scale energies/forces by 0.5.
    double full_neigh_scale = 1.0;
    if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

#if defined(_OPENMP)
#pragma omp parallel default(none)				\
  shared(ilist,firstneigh,numneigh,lf,full_neigh_scale)		\
  private(i,j)\
  reduction(+:ene)
#endif
  {

    #if defined (_OPENMP)
    const int tid = omp_get_thread_num();
    const int nall = nlocal + atom->nghost;
    const int off = nall * tid;
    #else
    const int tid = 0; // default
    const int off = 0; // default
    #endif

    // Let the compiler decide the break up of this loop
    // Scheduling to help load balance here since this loop is weird
    #if defined (_OPENMP)
    #pragma omp for schedule(guided,1) 
    #endif
    for(i=0; i<inum; ++i)
    {
       const int atomi = ilist[i];
       const int jnum = numneigh[atomi];
       const int * jlist = firstneigh[atomi];
	
       double fix = 0.0;
       double fiy = 0.0;
       double fiz = 0.0;

       if(is_exch_chg[atomi]){
        // iflag = true here
        const double xxi = x[atomi][0];
        const double xyi = x[atomi][1];
        const double xzi = x[atomi][2];

	for(j=0; j<jnum; ++j)
	{
            int atomj = jlist[j];
            atomj &=NEIGHMASK;	  
            
            if( !(is_exch_chg[atomj]) ) 
            {
		const double qiqj = q[atomi] * q[atomj];
                if(fabs(qiqj) > SMALL) 
		{
                    const double dx = xxi - x[atomj][0];
                    const double dy = xyi - x[atomj][1];
                    const double dz = xzi - x[atomj][2];
                    const double r2 = dx*dx + dy*dy + dz*dz;
                
                    if (r2 < cut_sq) 
                    {
                        const double r = sqrt(r2);

                        const double A1 =  0.254829592;
                        const double A2 = -0.284496736;
                        const double A3 =  1.421413741;
                        const double A4 = -1.453152027;
                        const double A5 =  1.061405429;
                        const double EWALD_F = 1.12837917;
                        const double INV_EWALD_P = 1.0/0.3275911;

                        const double grij = g_ewald * r;
                        const double expm2 = exp(-grij*grij); 
                        const double t = INV_EWALD_P / (INV_EWALD_P + grij);
                        const double erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                        const double prefactor = qqrd2e * qiqj / r;
                        double epair = prefactor * erfc;
                        double fpair = epair + prefactor*EWALD_F*grij*expm2;
                        
                        fpair *= A_Rq / r2 * full_neigh_scale;
                        ene += epair;
                
                        const double ftmpx = fpair * dx;
                        const double ftmpy = fpair * dy;
                        const double ftmpz = fpair * dz;
                        fix += ftmpx;
                        fiy += ftmpy;
                        fiz += ftmpz; 
                        
                        if (NEWTON_PAIR || j < nlocal) 
                        {
                            lf[atomj + off][0] -= ftmpx;
                            lf[atomj + off][1] -= ftmpy;
                            lf[atomj + off][2] -= ftmpz; 
                        }
                        
                        if (VFLAG) 
                        {
                          #if defined(_OPENMP)
                          #pragma omp critical
                          #endif
                          {
                            virial[0] += ftmpx * dx; 
                            virial[1] += ftmpy * dy;
                            virial[2] += ftmpz * dz;
                            virial[3] += ftmpx * dy;
                            virial[4] += ftmpx * dz;
                            virial[5] += ftmpy * dz;
                          }
                        }
                    }
                }  
                
            } // End of calculation
	} // End of loop atom j 1

       } else {
        // iflag = false here

	for(j=0; j<jnum; ++j)
	{
            int atomj = jlist[j];
            atomj &=NEIGHMASK;	  
            
            if( is_exch_chg[atomj] )
            {
		const double qiqj = q[atomi] * q[atomj];
                if(fabs(qiqj) > SMALL) 
		{
                    // not worth saving xxi, xyi, xzi, since won't need them as much here
                    const double dx = x[atomi][0] - x[atomj][0];
                    const double dy = x[atomi][1] - x[atomj][1];
                    const double dz = x[atomi][2] - x[atomj][2];
                    const double r2 = dx*dx + dy*dy + dz*dz;
                
                    if (r2 < cut_sq) 
                    {
                        const double r = sqrt(r2);

                        const double A1 =  0.254829592;
                        const double A2 = -0.284496736;
                        const double A3 =  1.421413741;
                        const double A4 = -1.453152027;
                        const double A5 =  1.061405429;
                        const double EWALD_F = 1.12837917;
                        const double INV_EWALD_P = 1.0/0.3275911;

                        const double grij = g_ewald * r;
                        const double expm2 = exp(-grij*grij);
                        const double t = INV_EWALD_P / (INV_EWALD_P + grij);
                        const double erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                        const double prefactor = qqrd2e * qiqj / r;
                        double epair = prefactor * erfc;
                        double fpair = epair + prefactor*EWALD_F*grij*expm2;
                        
                        fpair *= A_Rq / r2 * full_neigh_scale; 
                        ene += epair;
                
                        const double ftmpx = fpair * dx;
                        const double ftmpy = fpair * dy;
                        const double ftmpz = fpair * dz;
                        fix += ftmpx;
                        fiy += ftmpy;
                        fiz += ftmpz; 
                        
                        if (NEWTON_PAIR || j < nlocal) 
                        {
                            lf[atomj + off][0] -= ftmpx;
                            lf[atomj + off][1] -= ftmpy;
                            lf[atomj + off][2] -= ftmpz; 
                        }
                        
                        if (VFLAG) 
                        {
                          #if defined(_OPENMP)
                          #pragma omp critical
                          #endif
                          {
                            virial[0] += ftmpx * dx; 
                            virial[1] += ftmpy * dy;
                            virial[2] += ftmpz * dz;
                            virial[3] += ftmpx * dy;
                            virial[4] += ftmpx * dz;
                            virial[5] += ftmpy * dz;
                          }
                        }
                    }
                }  
                
            } // End of calculation
	} // End of loop atom j 2
       } // End of else

       lf[atomi + off][0] += fix;
       lf[atomi + off][1] += fiy;
       lf[atomi + off][2] += fiz; 

    } // End of loop atom i

  } // End of OpenMP section
  ene *= full_neigh_scale;
  return ene;
}

/* ---------------------------------------------------------------------- */

double EVB_OffDiag::exch_chg_cut_omp(int vflag)
{
  // If my proc doesn't have an exch chg, exit b/c there's nothing to do here
  if (!(evb_engine->has_exch_chg)) return 0.0;

  int newton_pair = force->newton_pair;
  if(newton_pair){
    if(vflag){ return exch_chg_cut_omp_eval<1,1>();}
    else     { return exch_chg_cut_omp_eval<0,1>();}
  } else {
    if(vflag){ return exch_chg_cut_omp_eval<1,0>();}
    else     { return exch_chg_cut_omp_eval<0,0>();}
  }
}

template <int VFLAG, int NEWTON_PAIR>
double EVB_OffDiag::exch_chg_cut_omp_eval()
{
  const int newton_pair = force->newton_pair;
  const double qqrd2e = force->qqrd2e;
  const int nlocal = atom->nlocal;
    
  double ene = 0.0;

  if(strcmp(force->pair_style,"lj/cut/coul/cut")==0) {
    PairLJCutCoulCut* ptr = (PairLJCutCoulCut*)(force->pair);
    cut = ptr->cut_coul_global;
    cut_sq = cut*cut;
  } else if(strcmp(force->pair_style,"lj/charmm/coul/charmm")==0) {
    PairLJCharmmCoulCharmm* ptr = (PairLJCharmmCoulCharmm*)(force->pair);
    cut = ptr->cut_coul;
    cut_sq = cut*cut;
  } else {
    int itmp;
    double *p_cutoff = (double *) force->pair->extract((char*)("cut_coul"),itmp);
    cut = (*p_cutoff);
    cut_sq = cut*cut;
  }

  const double * const q = atom->q; 
  const double * const * const x = atom->x;
 
  NeighList *list = evb_engine->get_pair_list();   
  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  
  int i, j; 
  
  // NOTE: This is the *original* force array that has extra allocated space for threads. 
  //       It should be empty! We don't empty it in this routine b/c Force_Reduce will pick it up. 
  //       Force_Reduce will also sum it into where atom->f currently points
  double **lf = evb_engine->lmp_f; 
  
  
  // NOTE: The criticals for VFLAG are me being lazy. But, I don't think we usually need the virial,
  //       so it shouldn't matter too much, at least for now.
  
  // If full neighbor list, scale energies/forces by 0.5.
  double full_neigh_scale = 1.0;
  if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

#if defined(_OPENMP)
#pragma omp parallel default(none)				\
  shared(ilist,firstneigh,numneigh,lf,full_neigh_scale)		\
  private(i,j)\
  reduction(+:ene)
#endif
  {
    
#if defined (_OPENMP)
    const int tid = omp_get_thread_num();
    const int nall = nlocal + atom->nghost;
    const int off = nall * tid;
#else
    const int tid = 0; // default
    const int off = 0; // default
#endif
    
    // Let the compiler decide the break up of this loop
    // Scheduling to help load balance here since this loop is weird
    #if defined (_OPENMP)
    #pragma omp for schedule(guided,1) 
    #endif
    for(i=0; i<inum; ++i) {
      const int atomi = ilist[i];
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
      
      double fix = 0.0;
      double fiy = 0.0;
      double fiz = 0.0;
      
      if(is_exch_chg[atomi]) {
	// iflag = true here
	const double xxi = x[atomi][0];
	const double xyi = x[atomi][1];
	const double xzi = x[atomi][2];
	
	for(j=0; j<jnum; ++j) {
	  int atomj = jlist[j];
	  atomj &=NEIGHMASK;	  
	  
	  if( !(is_exch_chg[atomj]) ) {
	    const double qiqj = q[atomi] * q[atomj];
	    if(fabs(qiqj) > SMALL) {
	      const double dx = xxi - x[atomj][0];
	      const double dy = xyi - x[atomj][1];
	      const double dz = xzi - x[atomj][2];
	      const double r2 = dx*dx + dy*dy + dz*dz;
	      
	      if (r2 < cut_sq) {
		const double r2inv = 1.0 / r2;
		const double ecoul = qqrd2e * qiqj * sqrt(r2inv);
		const double fpair = ecoul * r2inv * A_Rq * full_neigh_scale;
		ene += ecoul;
		
		const double ftmpx = fpair * dx;
		const double ftmpy = fpair * dy;
		const double ftmpz = fpair * dz;
		fix += ftmpx;
		fiy += ftmpy;
		fiz += ftmpz; 
		
		if (NEWTON_PAIR || j < nlocal) {
		  lf[atomj + off][0] -= ftmpx;
		  lf[atomj + off][1] -= ftmpy;
		  lf[atomj + off][2] -= ftmpz; 
		}
                
		if (VFLAG) {
#if defined(_OPENMP)
#pragma omp critical
#endif
		  {
		    virial[0] += ftmpx * dx; 
		    virial[1] += ftmpy * dy;
		    virial[2] += ftmpz * dz;
		    virial[3] += ftmpx * dy;
		    virial[4] += ftmpx * dz;
		    virial[5] += ftmpy * dz;
		  }
		}
	      }
	    }  
	    
	  } // End of calculation
	} // End of loop atom j 1
	
      } else {
        // iflag = false here
	
	for(j=0; j<jnum; ++j) {
	  int atomj = jlist[j];
	  atomj &=NEIGHMASK;	  
	  
	  if( is_exch_chg[atomj] ) {
	    const double qiqj = q[atomi] * q[atomj];
	    if(fabs(qiqj) > SMALL) {
	      // not worth saving xxi, xyi, xzi, since won't need them as much here
	      const double dx = x[atomi][0] - x[atomj][0];
	      const double dy = x[atomi][1] - x[atomj][1];
	      const double dz = x[atomi][2] - x[atomj][2];
	      const double r2 = dx*dx + dy*dy + dz*dz;
              
	      if (r2 < cut_sq) {
		const double r2inv = 1.0 / r2;
		const double ecoul = qqrd2e * qiqj * sqrt(r2inv);
		const double fpair = ecoul * r2inv * A_Rq * full_neigh_scale;
		ene += ecoul;
		
		const double ftmpx = fpair * dx;
		const double ftmpy = fpair * dy;
		const double ftmpz = fpair * dz;
		fix += ftmpx;
		fiy += ftmpy;
		fiz += ftmpz; 
                
		if (NEWTON_PAIR || j < nlocal) {
		  lf[atomj + off][0] -= ftmpx;
		  lf[atomj + off][1] -= ftmpy;
		  lf[atomj + off][2] -= ftmpz; 
		}
		
		if (VFLAG) {
#if defined(_OPENMP)
#pragma omp critical
#endif
		  {
		    virial[0] += ftmpx * dx; 
		    virial[1] += ftmpy * dy;
		    virial[2] += ftmpz * dz;
		    virial[3] += ftmpx * dy;
		    virial[4] += ftmpx * dz;
		    virial[5] += ftmpy * dz;
		  }
		}
	      }
	    }  
            
	  } // End of calculation
	} // End of loop atom j 2
      } // End of else
      
      lf[atomi + off][0] += fix;
      lf[atomi + off][1] += fiy;
      lf[atomi + off][2] += fiz; 
      
    } // End of loop atom i
    
  } // End of OpenMP section
  
  ene *= full_neigh_scale;
  return ene;
}

/* ---------------------------------------------------------------------- */

double EVB_OffDiag::exch_chg_debye_omp(int vflag)
{
  // If my proc doesn't have an exch chg, exit b/c there's nothing to do here
  if (!(evb_engine->has_exch_chg)) return 0.0;

  int newton_pair = force->newton_pair;
  if(newton_pair){
    if(vflag){ return exch_chg_debye_omp_eval<1,1>();}
    else     { return exch_chg_debye_omp_eval<0,1>();}
  } else {
    if(vflag){ return exch_chg_debye_omp_eval<1,0>();}
    else     { return exch_chg_debye_omp_eval<0,0>();}
  }
}

template <int VFLAG, int NEWTON_PAIR>
double EVB_OffDiag::exch_chg_debye_omp_eval()
{
  const int newton_pair = force->newton_pair;
  const double qqrd2e = force->qqrd2e;
  const int nlocal = atom->nlocal;
    
  double ene = 0.0;

  if(strcmp(force->pair_style,"lj/cut/coul/cut")==0) {
    PairLJCutCoulCut* ptr = (PairLJCutCoulCut*)(force->pair);
    cut = ptr->cut_coul_global;
    cut_sq = cut*cut;
  } else if(strcmp(force->pair_style,"lj/charmm/coul/charmm")==0) {
    PairLJCharmmCoulCharmm* ptr = (PairLJCharmmCoulCharmm*)(force->pair);
    cut = ptr->cut_coul;
    cut_sq = cut*cut;
  } else {
    int itmp;
    double *p_cutoff = (double *) force->pair->extract((char*)("cut_coul"),itmp);
    cut = (*p_cutoff);
    cut_sq = cut*cut;
  }
  
  const double * const q = atom->q; 
  const double * const * const x = atom->x;
    
  NeighList *list = evb_engine->get_pair_list();
  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  
  int i, j; 
  
  // NOTE: This is the *original* force array that has extra allocated space for threads. 
  //       It should be empty! We don't empty it in this routine b/c Force_Reduce will pick it up. 
  //       Force_Reduce will also sum it into where atom->f currently points
  double **lf = evb_engine->lmp_f; 
  
  
  // NOTE: The criticals for VFLAG are me being lazy. But, I don't think we usually need the virial,
  //       so it shouldn't matter too much, at least for now.
  
  // If full neighbor list, scale energies/forces by 0.5.
  double full_neigh_scale = 1.0;
  if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

#if defined(_OPENMP)
#pragma omp parallel default(none)				\
  shared(ilist,firstneigh,numneigh,lf,full_neigh_scale)		\
  private(i,j)\
  reduction(+:ene)
#endif
  {
    
#if defined (_OPENMP)
    const int tid = omp_get_thread_num();
    const int nall = nlocal + atom->nghost;
    const int off = nall * tid;
#else
    const int tid = 0; // default
    const int off = 0; // default
#endif
    
    // Let the compiler decide the break up of this loop
    // Scheduling to help load balance here since this loop is weird
    #if defined (_OPENMP)
    #pragma omp for schedule(guided,1) 
    #endif
    for(i=0; i<inum; ++i) {
      const int atomi = ilist[i];
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
      
      double fix = 0.0;
      double fiy = 0.0;
      double fiz = 0.0;
      
      if(is_exch_chg[atomi]) {
	// iflag = true here
	const double xxi = x[atomi][0];
	const double xyi = x[atomi][1];
	const double xzi = x[atomi][2];
	
	for(j=0; j<jnum; ++j) {
	  int atomj = jlist[j];
	  atomj &=NEIGHMASK;	  
	  
	  if( !(is_exch_chg[atomj]) ) {
	    const double qiqj = q[atomi] * q[atomj];
	    if(fabs(qiqj) > SMALL) {
	      const double dx = xxi - x[atomj][0];
	      const double dy = xyi - x[atomj][1];
	      const double dz = xzi - x[atomj][2];
	      const double r2 = dx*dx + dy*dy + dz*dz;
	      
	      if (r2 < cut_sq) {
		const double r = sqrt(r2);
		const double r2inv = 1.0 / r2;
		const double rinv = sqrt(r2inv);
		const double screened = qqrd2e * qiqj * exp(-kappa*r) * full_neigh_scale;
		const double ecoul = screened * rinv;
		const double fpair = screened * (kappa + rinv) * ecoul * r2inv * A_Rq;
		ene += ecoul;
		
		const double ftmpx = fpair * dx;
		const double ftmpy = fpair * dy;
		const double ftmpz = fpair * dz;
		fix += ftmpx;
		fiy += ftmpy;
		fiz += ftmpz; 
		
		if (NEWTON_PAIR || j < nlocal) {
		  lf[atomj + off][0] -= ftmpx;
		  lf[atomj + off][1] -= ftmpy;
		  lf[atomj + off][2] -= ftmpz; 
		}
                
		if (VFLAG) {
#if defined(_OPENMP)
#pragma omp critical
#endif
		  {
		    virial[0] += ftmpx * dx; 
		    virial[1] += ftmpy * dy;
		    virial[2] += ftmpz * dz;
		    virial[3] += ftmpx * dy;
		    virial[4] += ftmpx * dz;
		    virial[5] += ftmpy * dz;
		  }
		}
	      }
	    }  
	    
	  } // End of calculation
	} // End of loop atom j 1
	
      } else {
        // iflag = false here
	
	for(j=0; j<jnum; ++j) {
	  int atomj = jlist[j];
	  atomj &=NEIGHMASK;	  
	  
	  if( is_exch_chg[atomj] ) {
	    const double qiqj = q[atomi] * q[atomj];
	    if(fabs(qiqj) > SMALL) {
	      // not worth saving xxi, xyi, xzi, since won't need them as much here
	      const double dx = x[atomi][0] - x[atomj][0];
	      const double dy = x[atomi][1] - x[atomj][1];
	      const double dz = x[atomi][2] - x[atomj][2];
	      const double r2 = dx*dx + dy*dy + dz*dz;
              
	      if (r2 < cut_sq) {
		const double r = sqrt(r2);
		const double r2inv = 1.0 / r2;
		const double rinv = sqrt(r2inv);
		const double screened = qqrd2e * qiqj * exp(-kappa*r) * full_neigh_scale;
		const double ecoul = screened * rinv;
		const double fpair = screened * (kappa + rinv) * ecoul * r2inv * A_Rq;
		ene += ecoul;

		const double ftmpx = fpair * dx;
		const double ftmpy = fpair * dy;
		const double ftmpz = fpair * dz;
		fix += ftmpx;
		fiy += ftmpy;
		fiz += ftmpz; 
                
		if (NEWTON_PAIR || j < nlocal) {
		  lf[atomj + off][0] -= ftmpx;
		  lf[atomj + off][1] -= ftmpy;
		  lf[atomj + off][2] -= ftmpz; 
		}
		
		if (VFLAG) {
#if defined(_OPENMP)
#pragma omp critical
#endif
		  {
		    virial[0] += ftmpx * dx; 
		    virial[1] += ftmpy * dy;
		    virial[2] += ftmpz * dz;
		    virial[3] += ftmpx * dy;
		    virial[4] += ftmpx * dz;
		    virial[5] += ftmpy * dz;
		  }
		}
	      }
	    }  
            
	  } // End of calculation
	} // End of loop atom j 2
      } // End of else
      
      lf[atomi + off][0] += fix;
      lf[atomi + off][1] += fiy;
      lf[atomi + off][2] += fiz; 
      
    } // End of loop atom i
    
  } // End of OpenMP section
  
  return ene;
}

/* ---------------------------------------------------------------------- */

double EVB_OffDiag::exch_chg_wolf_omp(int vflag)
{
  // If my proc doesn't have an exch chg, exit b/c there's nothing to do here
  if (!(evb_engine->has_exch_chg)) return 0.0;

  int newton_pair = force->newton_pair;
  if(newton_pair){
    if(vflag){ return exch_chg_wolf_omp_eval<1,1>();}
    else     { return exch_chg_wolf_omp_eval<0,1>();}
  } else {
    if(vflag){ return exch_chg_wolf_omp_eval<1,0>();}
    else     { return exch_chg_wolf_omp_eval<0,0>();}
  }
}

template <int VFLAG, int NEWTON_PAIR>
double EVB_OffDiag::exch_chg_wolf_omp_eval()
{
  const int newton_pair = force->newton_pair;
  const double qqrd2e = force->qqrd2e;
  const int nlocal = atom->nlocal;
    
  double ene = 0.0;

  if(strcmp(force->pair_style,"lj/cut/coul/cut")==0) {
    PairLJCutCoulCut* ptr = (PairLJCutCoulCut*)(force->pair);
    cut = ptr->cut_coul_global;
    cut_sq = cut*cut;
  } else if(strcmp(force->pair_style,"lj/charmm/coul/charmm")==0) {
    PairLJCharmmCoulCharmm* ptr = (PairLJCharmmCoulCharmm*)(force->pair);
    cut = ptr->cut_coul;
    cut_sq = cut*cut;
  } else {
    int itmp;
    double *p_cutoff = (double *) force->pair->extract((char*)("cut_coul"),itmp);
    cut = (*p_cutoff);
    cut_sq = cut*cut;
  }
  
  const double * const q = atom->q; 
  const double * const * const x = atom->x;
    
  NeighList *list = evb_engine->get_pair_list();
  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  int i, j; 
  
  // NOTE: This is the *original* force array that has extra allocated space for threads. 
  //       It should be empty! We don't empty it in this routine b/c Force_Reduce will pick it up. 
  //       Force_Reduce will also sum it into where atom->f currently points
  double **lf = evb_engine->lmp_f; 
  
  
  // NOTE: The criticals for VFLAG are me being lazy. But, I don't think we usually need the virial,
  //       so it shouldn't matter too much, at least for now.
  
  // If full neighbor list, scale energies/forces by 0.5.
  double full_neigh_scale = 1.0;
  if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

#if defined(_OPENMP)
#pragma omp parallel default(none)				\
  shared(ilist,firstneigh,numneigh,lf,full_neigh_scale)		\
  private(i,j)\
  reduction(+:ene)
#endif
  {
    
#if defined (_OPENMP)
    const int tid = omp_get_thread_num();
    const int nall = nlocal + atom->nghost;
    const int off = nall * tid;
#else
    const int tid = 0; // default
    const int off = 0; // default
#endif

    double scale = 2.0 * kappa / MY_PIS;  // 2 \alpha / sqrt(PI)
    
    // Let the compiler decide the break up of this loop
    // Scheduling to help load balance here since this loop is weird
    #if defined (_OPENMP)
    #pragma omp for schedule(guided,1) 
    #endif
    for(i=0; i<inum; ++i) {
      const int atomi = ilist[i];
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
      
      double fix = 0.0;
      double fiy = 0.0;
      double fiz = 0.0;
      
      if(is_exch_chg[atomi]) {
	// iflag = true here
	const double xxi = x[atomi][0];
	const double xyi = x[atomi][1];
	const double xzi = x[atomi][2];
	
	for(j=0; j<jnum; ++j) {
	  int atomj = jlist[j];
	  atomj &=NEIGHMASK;	  
	  
	  if( !(is_exch_chg[atomj]) ) {
	    const double qiqj = q[atomi] * q[atomj];
	    if(fabs(qiqj) > SMALL) {
	      const double dx = xxi - x[atomj][0];
	      const double dy = xyi - x[atomj][1];
	      const double dz = xzi - x[atomj][2];
	      const double r2 = dx*dx + dy*dy + dz*dz;
	      
	      if (r2 < cut_sq) {
		double r = sqrt(r2);
		double r2inv = 1.0 / r2;
		double rinv = 1.0 / r;
		double rcutinv = 1.0 / cut;

		const double A1 =  0.254829592;
		const double A2 = -0.284496736;
		const double A3 =  1.421413741;
		const double A4 = -1.453152027;
		const double A5 =  1.061405429;
		
		double kr = kappa * r;
		double krcut = kappa * cut;
		// erfc(\alpha * r) / r
		double t = 1.0 / (1.0 + kr);
		double A = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * rinv;
		// erfc(\alpha * rcut) / rcut
		t = 1.0 / (1.0 + krcut);
		double B = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * rcutinv;
		
		double C = scale * rcutinv * exp(-krcut * krcut);
		double D = scale * rinv    * exp(-kr * kr);
		
		double ecoul = qqrd2e * qiqj * (A - B + (B * rcutinv + C) * (r - cut));
		double fpair = qqrd2e * qiqj * (( A * rinv + D - (B * rcutinv + C) ) * rinv) * full_neigh_scale;
		ene += ecoul;

		const double ftmpx = fpair * dx;
		const double ftmpy = fpair * dy;
		const double ftmpz = fpair * dz;
		fix += ftmpx;
		fiy += ftmpy;
		fiz += ftmpz; 
		
		if (NEWTON_PAIR || j < nlocal) {
		  lf[atomj + off][0] -= ftmpx;
		  lf[atomj + off][1] -= ftmpy;
		  lf[atomj + off][2] -= ftmpz; 
		}
                
		if (VFLAG) {
#if defined(_OPENMP)
#pragma omp critical
#endif
		  {
		    virial[0] += ftmpx * dx; 
		    virial[1] += ftmpy * dy;
		    virial[2] += ftmpz * dz;
		    virial[3] += ftmpx * dy;
		    virial[4] += ftmpx * dz;
		    virial[5] += ftmpy * dz;
		  }
		}
	      }
	    }  
	    
	  } // End of calculation
	} // End of loop atom j 1
	
      } else {
        // iflag = false here
	
	for(j=0; j<jnum; ++j) {
	  int atomj = jlist[j];
	  atomj &=NEIGHMASK;	  
	  
	  if( is_exch_chg[atomj] ) {
	    const double qiqj = q[atomi] * q[atomj];
	    if(fabs(qiqj) > SMALL) {
	      // not worth saving xxi, xyi, xzi, since won't need them as much here
	      const double dx = x[atomi][0] - x[atomj][0];
	      const double dy = x[atomi][1] - x[atomj][1];
	      const double dz = x[atomi][2] - x[atomj][2];
	      const double r2 = dx*dx + dy*dy + dz*dz;
              
	      if (r2 < cut_sq) {
		double r = sqrt(r2);
		double r2inv = 1.0 / r2;
		double rinv = 1.0 / r;
		double rcutinv = 1.0 / cut;

		const double A1 =  0.254829592;
		const double A2 = -0.284496736;
		const double A3 =  1.421413741;
		const double A4 = -1.453152027;
		const double A5 =  1.061405429;
		
		double kr = kappa * r;
		double krcut = kappa * cut;
		// erfc(\alpha * r) / r
		double t = 1.0 / (1.0 + kr);
		double A = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * rinv;
		// erfc(\alpha * rcut) / rcut
		t = 1.0 / (1.0 + krcut);
		double B = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * rcutinv;
		
		double C = scale * rcutinv * exp(-krcut * krcut);
		double D = scale * rinv    * exp(-kr * kr);
		
		double ecoul = qqrd2e * qiqj * (A - B + (B * rcutinv + C) * (r - cut));
		double fpair = qqrd2e * qiqj * (( A * rinv + D - (B * rcutinv + C) ) * rinv) * full_neigh_scale;

		ene += ecoul;

		const double ftmpx = fpair * dx;
		const double ftmpy = fpair * dy;
		const double ftmpz = fpair * dz;
		fix += ftmpx;
		fiy += ftmpy;
		fiz += ftmpz; 
                
		if (NEWTON_PAIR || j < nlocal) {
		  lf[atomj + off][0] -= ftmpx;
		  lf[atomj + off][1] -= ftmpy;
		  lf[atomj + off][2] -= ftmpz; 
		}
		
		if (VFLAG) {
#if defined(_OPENMP)
#pragma omp critical
#endif
		  {
		    virial[0] += ftmpx * dx; 
		    virial[1] += ftmpy * dy;
		    virial[2] += ftmpz * dz;
		    virial[3] += ftmpx * dy;
		    virial[4] += ftmpx * dz;
		    virial[5] += ftmpy * dz;
		  }
		}
	      }
	    }  
            
	  } // End of calculation
	} // End of loop atom j 2
      } // End of else
      
      lf[atomi + off][0] += fix;
      lf[atomi + off][1] += fiy;
      lf[atomi + off][2] += fiz; 
      
    } // End of loop atom i
    
  } // End of OpenMP section
  ene *= full_neigh_scale;
  return ene;
}

/* ---------------------------------------------------------------------- */

double EVB_OffDiag::exch_chg_cgis_omp(int vflag)
{
  // If my proc doesn't have an exch chg, exit b/c there's nothing to do here
  if (!(evb_engine->has_exch_chg)) return 0.0;

  int newton_pair = force->newton_pair;
  if(newton_pair){
    if(vflag){ return exch_chg_cgis_omp_eval<1,1>();}
    else     { return exch_chg_cgis_omp_eval<0,1>();}
  } else {
    if(vflag){ return exch_chg_cgis_omp_eval<1,0>();}
    else     { return exch_chg_cgis_omp_eval<0,0>();}
  }
}

template <int VFLAG, int NEWTON_PAIR>
double EVB_OffDiag::exch_chg_cgis_omp_eval()
{
  const int newton_pair = force->newton_pair;
  const double qqrd2e = force->qqrd2e;
  const int nlocal = atom->nlocal;
    
  double ene = 0.0;

  if(strcmp(force->pair_style,"lj/cut/coul/cut")==0) {
    PairLJCutCoulCut* ptr = (PairLJCutCoulCut*)(force->pair);
    cut = ptr->cut_coul_global;
    cut_sq = cut*cut;
  } else if(strcmp(force->pair_style,"lj/charmm/coul/charmm")==0) {
    PairLJCharmmCoulCharmm* ptr = (PairLJCharmmCoulCharmm*)(force->pair);
    cut = ptr->cut_coul;
    cut_sq = cut*cut;
  } else {
    int itmp;
    double *p_cutoff = (double *) force->pair->extract((char*)("cut_coul"),itmp);
    cut = (*p_cutoff);
    cut_sq = cut*cut;
  }

  const double cut_cgis = 0.81 * cut; // The 0.81 could be an input parameter
  const double cut_cgis2 = cut_cgis * cut_cgis;
  const double rcutinv = 1.0 / cut;

  const double onehalf = 1.0 / 2.0;
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
  
  const double * const q = atom->q; 
  const double * const * const x = atom->x;
  

  NeighList *list = evb_engine->get_pair_list();
  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  
  double scale = 2.0 * kappa / MY_PIS;  // 2 \alpha / sqrt(PI)

  int i, j; 
  
  // NOTE: This is the *original* force array that has extra allocated space for threads. 
  //       It should be empty! We don't empty it in this routine b/c Force_Reduce will pick it up. 
  //       Force_Reduce will also sum it into where atom->f currently points
  double **lf = evb_engine->lmp_f; 
  
  
  // NOTE: The criticals for VFLAG are me being lazy. But, I don't think we usually need the virial,
  //       so it shouldn't matter too much, at least for now.

  // If full neighbor list, scale energies/forces by 0.5.
  double full_neigh_scale = 1.0;
  if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

#if defined(_OPENMP)
#pragma omp parallel default(none)				\
  shared(ilist,firstneigh,numneigh,lf,full_neigh_scale)		\
  private(i,j)\
  reduction(+:ene)
#endif
  {
    
#if defined (_OPENMP)
    const int tid = omp_get_thread_num();
    const int nall = nlocal + atom->nghost;
    const int off = nall * tid;
#else
    const int tid = 0; // default
    const int off = 0; // default
#endif
    
    // Let the compiler decide the break up of this loop
    // Scheduling to help load balance here since this loop is weird
    #if defined (_OPENMP)
    #pragma omp for schedule(guided,1) 
    #endif
    for(i=0; i<inum; ++i) {
      const int atomi = ilist[i];
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
      
      double fix = 0.0;
      double fiy = 0.0;
      double fiz = 0.0;
      
      if(is_exch_chg[atomi]) {
	// iflag = true here
	const double xxi = x[atomi][0];
	const double xyi = x[atomi][1];
	const double xzi = x[atomi][2];
	
	for(j=0; j<jnum; ++j) {
	  int atomj = jlist[j];
	  atomj &=NEIGHMASK;	  
	  
	  if( !(is_exch_chg[atomj]) ) {
	    const double qiqj = q[atomi] * q[atomj];
	    if(fabs(qiqj) > SMALL) {
	      const double dx = xxi - x[atomj][0];
	      const double dy = xyi - x[atomj][1];
	      const double dz = xzi - x[atomj][2];
	      const double r2 = dx*dx + dy*dy + dz*dz;
	      
	      if (r2 < cut_sq) {
		double r = sqrt(r2);
		double rinv = 1.0 / r;
		double r2inv = rinv * rinv;
		
		double ecoul = rinv;
		double fpair = r2inv;
		
		if(r < cut_cgis) {
		  ecoul += cgis_const - B2 * (r2 - cut_cgis2);
		  fpair += B * r;
		} else {
		  const double dr = r - cut;
		  const double dr2 = dr * dr;
		  ecoul += -rcutinv + A3 * dr2 * dr + B2 * dr2 + C * dr;
		  fpair += -A * dr2 - B * dr - C;
		}
		
		ecoul*= qqrd2e * qiqj;
		fpair*= qqrd2e * qiqj * A_Rq * rinv * full_neigh_scale;
		
		ene += ecoul;

		const double ftmpx = fpair * dx;
		const double ftmpy = fpair * dy;
		const double ftmpz = fpair * dz;
		fix += ftmpx;
		fiy += ftmpy;
		fiz += ftmpz; 
		
		if (NEWTON_PAIR || j < nlocal) {
		  lf[atomj + off][0] -= ftmpx;
		  lf[atomj + off][1] -= ftmpy;
		  lf[atomj + off][2] -= ftmpz; 
		}
                
		if (VFLAG) {
#if defined(_OPENMP)
#pragma omp critical
#endif
		  {
		    virial[0] += ftmpx * dx; 
		    virial[1] += ftmpy * dy;
		    virial[2] += ftmpz * dz;
		    virial[3] += ftmpx * dy;
		    virial[4] += ftmpx * dz;
		    virial[5] += ftmpy * dz;
		  }
		}
	      }
	    }  
	    
	  } // End of calculation
	} // End of loop atom j 1
	
      } else {
        // iflag = false here
	
	for(j=0; j<jnum; ++j) {
	  int atomj = jlist[j];
	  atomj &=NEIGHMASK;	  
	  
	  if( is_exch_chg[atomj] ) {
	    const double qiqj = q[atomi] * q[atomj];
	    if(fabs(qiqj) > SMALL) {
	      // not worth saving xxi, xyi, xzi, since won't need them as much here
	      const double dx = x[atomi][0] - x[atomj][0];
	      const double dy = x[atomi][1] - x[atomj][1];
	      const double dz = x[atomi][2] - x[atomj][2];
	      const double r2 = dx*dx + dy*dy + dz*dz;
              
	      if (r2 < cut_sq) {
		double r = sqrt(r2);
		double rinv = 1.0 / r;
		double r2inv = rinv * rinv;
		
		double ecoul = rinv;
		double fpair = r2inv;
		
		if(r < cut_cgis) {
		  ecoul += cgis_const - B2 * (r2 - cut_cgis2);
		  fpair += B * r;
		} else {
		  const double dr = r - cut;
		  const double dr2 = dr * dr;
		  ecoul += -rcutinv + A3 * dr2 * dr + B2 * dr2 + C * dr;
		  fpair += -A * dr2 - B * dr - C;
		}
		
		ecoul*= qqrd2e * qiqj;
		fpair*= qqrd2e * qiqj * A_Rq * rinv * full_neigh_scale;
		
		ene += ecoul;

		const double ftmpx = fpair * dx;
		const double ftmpy = fpair * dy;
		const double ftmpz = fpair * dz;
		fix += ftmpx;
		fiy += ftmpy;
		fiz += ftmpz; 
                
		if (NEWTON_PAIR || j < nlocal) {
		  lf[atomj + off][0] -= ftmpx;
		  lf[atomj + off][1] -= ftmpy;
		  lf[atomj + off][2] -= ftmpz; 
		}
		
		if (VFLAG) {
#if defined(_OPENMP)
#pragma omp critical
#endif
		  {
		    virial[0] += ftmpx * dx; 
		    virial[1] += ftmpy * dy;
		    virial[2] += ftmpz * dz;
		    virial[3] += ftmpx * dy;
		    virial[4] += ftmpx * dz;
		    virial[5] += ftmpy * dz;
		  }
		}
	      }
	    }  
            
	  } // End of calculation
	} // End of loop atom j 2
      } // End of else
      
      lf[atomi + off][0] += fix;
      lf[atomi + off][1] += fiy;
      lf[atomi + off][2] += fiz; 
      
    } // End of loop atom i
    
  } // End of OpenMP section
  ene *= full_neigh_scale;
  return ene;
}
