/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "EVB_offdiag.h"
#include "EVB_kspace.h"

#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "math_const.h"

#define _CRACKER_PAIR_LJ_CUT_COUL_CUT
#define _CRACKER_PAIR_LJ_CHARMM_COUL_CHARMM
  #include "EVB_cracker.h"
#undef _CRACKER_PAIR_LJ_CUT_COUL_CUT
#undef _CRACKER_PAIR_LJ_CHARMM_COUL_CHARMM

#define SMALL 0.000001

#define EWALD_F   1.128379170
#define EWALD_P   0.327591100
#define EA1       0.254829592
#define EA2      -0.284496736
#define EA3       1.421413741
#define EA4      -1.453152027
#define EA5       1.061405429

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

int EVB_OffDiag::max_nexch = 7;

/* ---------------------------------------------------------------------- */

EVB_OffDiag::EVB_OffDiag(LAMMPS *lmp, EVB_Engine *engine) : Pointers(lmp), EVB_Pointers(engine)
{

}

/* ---------------------------------------------------------------------- */

EVB_OffDiag::~EVB_OffDiag()
{

}

/* ----------------------------------------------------------------------*/

double EVB_OffDiag::exch_chg_cut(int vflag)
{
#if defined(_OPENMP)
  return exch_chg_cut_omp(vflag);
#endif

  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  int nlocal = atom->nlocal;
  
  double ene = 0.0;
  
  if(strcmp(force->pair_style,"lj/cut/coul/cut")==0)
  {
    PairLJCutCoulCut* ptr = (PairLJCutCoulCut*)(force->pair);
    cut = ptr->cut_coul_global;
    cut_sq = cut*cut;
  }
  else if(strcmp(force->pair_style,"lj/charmm/coul/charmm")==0)
  {
    PairLJCharmmCoulCharmm* ptr = (PairLJCharmmCoulCharmm*)(force->pair);
    cut = ptr->cut_coul;
    cut_sq = cut*cut;
  }
  else {
    int itmp;
    double *p_cutoff = (double *) force->pair->extract((char*)("cut_coul"),itmp);
    cut = (*p_cutoff);
    cut_sq = cut*cut;
  }
 
  double *q = atom->q; 
  double **f = atom->f;
  double **x = atom->x;

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
    
  // If full neighbor list, scale energies/forces by 0.5.
  double full_neigh_scale = 1.0;
  if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

  for(int i=0; i<inum; i++) {
    bool iflag = false;
    
    int atomi = ilist[i];
    if(is_exch_chg[atomi]) iflag = true;
    
    int jnum = numneigh[atomi];
    int *jlist = firstneigh[atomi];
    
    for(int j=0; j<jnum; j++) {
      int atomj = jlist[j];
      atomj &=NEIGHMASK;
      
      bool jflag = false;
      if (is_exch_chg[atomj]) jflag = true;
      
      if( (iflag && (!jflag)) || (jflag && (!iflag)) ) {   
	/*************************************************************/
	/*************************************************************/
	double qiqj = q[atomi] * q[atomj];
	if (fabs(qiqj) > SMALL) {
	  double dr[3],r2;
	  VECTOR_SUB(dr,x[atomi],x[atomj]);
	  VECTOR_R2(r2,dr);
	  
	  if (r2 < cut_sq) {
	    double r2inv = 1.0 / r2;
	    double ecoul = qqrd2e * qiqj * sqrt(r2inv);
	    double fpair = ecoul  * r2inv * A_Rq * full_neigh_scale;
	    
	    double ftmpx = fpair * dr[0];
	    double ftmpy = fpair * dr[1];
	    double ftmpz = fpair * dr[2];
	    
	    if(atomi < nlocal) ene+= ecoul;
	    f[atomi][0] += ftmpx;
	    f[atomi][1] += ftmpy;
	    f[atomi][2] += ftmpz;
            
	    if (newton_pair || atomj < nlocal) {
	      f[atomj][0] -= ftmpx;
	      f[atomj][1] -= ftmpy;
	      f[atomj][2] -= ftmpz;
	    }
            
	    if (vflag) {
	      virial[0] += ftmpx * dr[0];
	      virial[1] += ftmpy * dr[1];
	      virial[2] += ftmpz * dr[2];
	      virial[3] += ftmpx * dr[1];
	      virial[4] += ftmpx * dr[2];
	      virial[5] += ftmpy * dr[2];
	    }
            
	  }
	}  
	/*************************************************************/
	/*************************************************************/	  
        
      } // End of calculation
    } // End of loop atom j
  } // End of loop atom i

  ene *= full_neigh_scale;
  return ene;
}

/* ---------------------------------------------------------------------- */

double EVB_OffDiag::exch_chg_debye(int vflag)
{
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  int nlocal = atom->nlocal;
  
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
  
  double *q = atom->q; 
  double **f = atom->f;
  double **x = atom->x;

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // If full neighbor list, scale energies/forces by 0.5.
  double full_neigh_scale = 1.0;
  if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

  for(int i=0; i<inum; i++) {
    bool iflag = false;
    
    int atomi = ilist[i];
    if(is_exch_chg[atomi]) iflag = true;
    
    int jnum = numneigh[atomi];
    int *jlist = firstneigh[atomi];
    
    for(int j=0; j<jnum; j++) {
      int atomj = jlist[j];
      atomj &=NEIGHMASK;	  
      
      bool jflag = false;
      if (is_exch_chg[atomj]) jflag = true;
      
      if( (iflag && (!jflag)) || (jflag && (!iflag)) ) {   
	/*************************************************************/
	/*************************************************************/
	double qiqj = q[atomi] * q[atomj];
	if (fabs(qiqj) > SMALL) {
	  double dr[3],r2;
	  VECTOR_SUB(dr,x[atomi],x[atomj]);
	  VECTOR_R2(r2,dr);
	  
	  if (r2 < cut_sq) {
	    double r = sqrt(r2);
	    double r2inv = 1.0 / r2;
	    double rinv = 1.0/ r;
	    double screened = qqrd2e * qiqj * exp(-kappa*r) * full_neigh_scale;
	    double ecoul = screened * rinv;
	    double fpair = screened * (kappa+rinv) * r2inv * A_Rq;
            
	    double ftmpx = fpair * dr[0];
	    double ftmpy = fpair * dr[1];
	    double ftmpz = fpair * dr[2];
            
	    if(atomi < nlocal) ene+= ecoul;
	    f[atomi][0] += ftmpx;
	    f[atomi][1] += ftmpy;
	    f[atomi][2] += ftmpz;
            
	    if (newton_pair || atomj < nlocal) {
	      f[atomj][0] -= ftmpx;
	      f[atomj][1] -= ftmpy;
	      f[atomj][2] -= ftmpz;
	    }
	    
	    if (vflag) {
	      virial[0] += ftmpx * dr[0];
	      virial[1] += ftmpy * dr[1];
	      virial[2] += ftmpz * dr[2];
	      virial[3] += ftmpx * dr[1];
	      virial[4] += ftmpx * dr[2];
	      virial[5] += ftmpy * dr[2];
	    }
	  }
	}  
	/*************************************************************/
	/*************************************************************/	  
        
      } // End of calculation
    } // End of loop atom j
  } // End of loop atom i

  return ene;
}

/* ---------------------------------------------------------------------- */
/* Eqs. 18 & 19 of 
   C. J. Fennell and J. D. Gezelter, JCP, 124(23), 234104 (2006) -------- */

double EVB_OffDiag::exch_chg_wolf(int vflag)
{
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  int nlocal = atom->nlocal;
  
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
  
  double *q = atom->q; 
  double **f = atom->f;
  double **x = atom->x;

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  // If full neighbor list, scale energies/forces by 0.5.
  double full_neigh_scale = 1.0;
  if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

  double scale = 2.0 * kappa / MY_PIS;  // 2 \alpha / sqrt(PI)
    
  for(int i=0; i<inum; i++) {
    bool iflag = false;
    
    int atomi = ilist[i];
    if(is_exch_chg[atomi]) iflag = true;
    
    int jnum = numneigh[atomi];
    int *jlist = firstneigh[atomi];
    
    for(int j=0; j<jnum; j++) {
      int atomj = jlist[j];
      atomj &= NEIGHMASK;
      
      bool jflag = false;
      if (is_exch_chg[atomj]) jflag = true;
      
      if( (iflag && (!jflag)) || (jflag && (!iflag)) ) {   
	/*************************************************************/
	/*************************************************************/
	double qiqj = q[atomi] * q[atomj];
	if (fabs(qiqj) > SMALL) {
	  double dr[3],r2;
	  VECTOR_SUB(dr,x[atomi],x[atomj]);
	  VECTOR_R2(r2,dr);
	  
	  if (r2 < cut_sq) {
	    double r = sqrt(r2);
	    double r2inv = 1.0 / r2;
	    double rinv = 1.0/ r;
	    double rcutinv = 1.0 / cut;
	    
	    double kr = kappa * r;
	    double krcut = kappa * cut;
	    // erfc(\alpha * r) / r
	    double t = 1.0 / (1.0 + kr);
	    double A = t * (EA1+t*(EA2+t*(EA3+t*(EA4+t*EA5)))) * rinv;
	    // erfc(\alpha * rcut) / rcut
	    t = 1.0 / (1.0 + krcut);
	    double B = t * (EA1+t*(EA2+t*(EA3+t*(EA4+t*EA5)))) * rcutinv;
	    
	    double C = scale * rcutinv * exp(-krcut * krcut);
	    double D = scale * rinv    * exp(-kr * kr);
	    
	    double ecoul = qqrd2e * qiqj * (A - B + (B * rcutinv + C) * (r - cut));
	    double fpair = qqrd2e * qiqj * (( A * rinv + D - (B * rcutinv + C) ) * rinv) * full_neigh_scale;
	    
	    double ftmpx = fpair * dr[0];
	    double ftmpy = fpair * dr[1];
	    double ftmpz = fpair * dr[2];
            
	    if(atomi < nlocal) ene+= ecoul;
	    f[atomi][0] += ftmpx;
	    f[atomi][1] += ftmpy;
	    f[atomi][2] += ftmpz;
            
	    if (newton_pair || atomj < nlocal) {
	      f[atomj][0] -= ftmpx;
	      f[atomj][1] -= ftmpy;
	      f[atomj][2] -= ftmpz;
	    }
	    
	    if (vflag) {
	      virial[0] += ftmpx * dr[0];
	      virial[1] += ftmpy * dr[1];
	      virial[2] += ftmpz * dr[2];
	      virial[3] += ftmpx * dr[1];
	      virial[4] += ftmpx * dr[2];
	      virial[5] += ftmpy * dr[2];
	    }
	  }
	}  
	/*************************************************************/
	/*************************************************************/	  
        
      } // End of calculation
    } // End of loop atom j
  } // End of loop atom i
  
  ene *= full_neigh_scale;
  return ene;
}

/* ---------------------------------------------------------------------- */
/* Eqs. 21-23 of 
   Q. Shi, P. Liu, and G. A. Voth, JPCB, 112(50), 16230-16237 (2008) ---- */

double EVB_OffDiag::exch_chg_cgis(int vflag)
{
#if defined(_OPENMP)
  return exch_chg_cgis_omp(vflag);
#endif

  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;
  int nlocal = atom->nlocal;

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

  double *q = atom->q; 
  double **f = atom->f;
  double **x = atom->x;

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

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

  // If full neighbor list, scale energies/forces by 0.5.
  double full_neigh_scale = 1.0;
  if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

  for(int i=0; i<inum; i++) {
    bool iflag = false;
 
    int atomi = ilist[i];
    if(is_exch_chg[atomi]) iflag = true;
 
    int jnum = numneigh[atomi];
    int *jlist = firstneigh[atomi];
 
    for(int j=0; j<jnum; j++) {
      int atomj = jlist[j];
      atomj &= NEIGHMASK;
   
      bool jflag = false;
      if (is_exch_chg[atomj]) jflag = true;
   
      if( (iflag && (!jflag)) || (jflag && (!iflag)) ) {   
	/*************************************************************/
	/*************************************************************/
	double qiqj = q[atomi] * q[atomj];
	if (fabs(qiqj) > SMALL) {
	  double dr[3],r2;
	  VECTOR_SUB(dr,x[atomi],x[atomj]);
	  VECTOR_R2(r2,dr);
	  
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
	      double dr = r - cut;
	      double dr2 = dr * dr;
	      ecoul += -rcutinv + A3 * dr2 * dr + B2 * dr2 + C * dr;
	      fpair += -A * dr2 - B * dr - C;
	    }
	    
	    ecoul*= qqrd2e * qiqj;
	    fpair*= qqrd2e * qiqj * A_Rq * rinv * full_neigh_scale;

	    double ftmpx = fpair * dr[0];
	    double ftmpy = fpair * dr[1];
	    double ftmpz = fpair * dr[2];
         
	    if(atomi < nlocal) ene+= ecoul;
	    f[atomi][0] += ftmpx;
	    f[atomi][1] += ftmpy;
	    f[atomi][2] += ftmpz;
         
	    if (newton_pair || atomj < nlocal) {
	      f[atomj][0] -= ftmpx;
	      f[atomj][1] -= ftmpy;
	      f[atomj][2] -= ftmpz;
	    }
	    
	    if (vflag) {
	      virial[0] += ftmpx * dr[0];
	      virial[1] += ftmpy * dr[1];
	      virial[2] += ftmpz * dr[2];
	      virial[3] += ftmpx * dr[1];
	      virial[4] += ftmpx * dr[2];
	      virial[5] += ftmpy * dr[2];
	    }
         
	  }
	}  
	/*************************************************************/
	/*************************************************************/	  
     
      } // End of calculation
    } // End of loop atom j
  } // End of loop atom i

  ene *= full_neigh_scale;
  return ene;
}

/* ---------------------------------------------------------------------- */

double EVB_OffDiag::exch_chg_long(int vflag)
{
#if defined(_OPENMP)
  // ** AWGL ** //
  return exch_chg_long_omp(vflag);
#endif

    double g_ewald = force->kspace->g_ewald;
    int newton_pair = force->newton_pair;
    double qqrd2e = force->qqrd2e;
    int nlocal = atom->nlocal;
    
    double ene = 0.0;
    int itmp;
    double *p_cutoff = (double *) force->pair->extract((char*)("cut_coul"),itmp);
    cut = (*p_cutoff);
    cut_sq = cut*cut;

    double *q = atom->q; 
    double **f = atom->f;
    double **x = atom->x;
    
    NeighList *list = evb_engine->get_pair_list();
    int inum = list->inum;
    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    // If full neighbor list, scale energies/forces by 0.5.
    double full_neigh_scale = 1.0;
    if(evb_engine->evb_full_neigh) full_neigh_scale = 0.5;

    for(int i=0; i<inum; i++) {
      bool iflag = false;
      
      int atomi = ilist[i];
      if(is_exch_chg[atomi]) iflag = true;
      
      int jnum = numneigh[atomi];
      int *jlist = firstneigh[atomi];
      
      for(int j=0; j<jnum; j++) {
	int atomj = jlist[j];
	atomj &=NEIGHMASK;	  
	
	bool jflag = false;
	if(is_exch_chg[atomj]) jflag = true;
        
	if( (iflag && (!jflag)) || (jflag && (!iflag)) ) {
	  /*************************************************************/
	  /*************************************************************/
	  double qiqj = q[atomi] * q[atomj];
	  if(fabs(qiqj) > SMALL) {
	    double dr[3],r2;
	    VECTOR_SUB(dr,x[atomi],x[atomj]);
	    VECTOR_R2(r2,dr);
            
	    if (r2 < cut_sq) {
	      double r = sqrt(r2);
	      
	      double grij = g_ewald * r;
	      double expm2 = exp(-grij*grij);
	      double t = 1.0 / (1.0 + EWALD_P*grij);
	      double erfc = t * (EA1+t*(EA2+t*(EA3+t*(EA4+t*EA5)))) * expm2;
	      double prefactor = qqrd2e * qiqj / r;
	      double epair = prefactor * erfc;
	      double fpair = epair + prefactor*EWALD_F*grij*expm2;

	      fpair *= A_Rq / r2 * full_neigh_scale;

	      double ftmpx = fpair * dr[0];
	      double ftmpy = fpair * dr[1];
	      double ftmpz = fpair * dr[2];

	      if(atomi < nlocal) ene += epair;
	      f[atomi][0] += ftmpx;
	      f[atomi][1] += ftmpy;
	      f[atomi][2] += ftmpz;
              
	      if (newton_pair || atomj < nlocal) {
		f[atomj][0] -= ftmpx;
		f[atomj][1] -= ftmpy;
		f[atomj][2] -= ftmpz;
	      }
              
	      if (vflag) {
		virial[0] += ftmpx * dr[0];
		virial[1] += ftmpy * dr[1];
		virial[2] += ftmpz * dr[2];
		virial[3] += ftmpx * dr[1];
		virial[4] += ftmpx * dr[2];
		virial[5] += ftmpy * dr[2];
	      }
	    }
	  }  
	  /*************************************************************/
	  /*************************************************************/	  
          
	} // End of calculation
      } // End of loop atom j
    } // End of loop atom i
    
    ene *= full_neigh_scale;
    return ene;
}
