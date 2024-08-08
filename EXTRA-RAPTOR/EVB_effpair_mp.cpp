/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight
------------------------------------------------------------------------- */

#ifdef RAPTOR_MPVERLET

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
#include "neigh_list.h"
#include "kspace.h"
#include "universe.h"

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

#define _CRACKER_PAIR
#define _CRACKER_PAIR_LJ_CHARMM_COUL_LONG
#define _CRACKER_PAIR_LJ_CUT_COUL_LONG
#if defined (_OPENMP)
#define _CRACKER_PAIR_LJ_CUT_COUL_LONG_OMP
#endif
  #include "EVB_cracker.h"
#undef _CRACKER_PAIR
#undef _CRACKER_PAIR_LJ_CHARMM_COUL_LONG
#undef _CRACKER_PAIR_LJ_CUT_COUL_LONG
#if defined (_OPENMP)
#undef _CRACKER_PAIR_LJ_CUT_COUL_LONG_OMP
#endif

//#include "pair_table_lj_cut_coul_long.h"
//#include "pair_gulp_coul_long.h"

//#include "pair_electrode.h"
//#include "pair_electrode_omp.h"

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   Setup some pair_styles during the first phase of compute_sci_mp() when 
   worker partitions do not call compute() to evaluate ENV interactions.
   ---------------------------------------------------------------------- */

void EVB_EffPair::setup_pair_mp()
{
  if(strcmp(force->pair_style,"electrode") == 0) {
    ptrPair_5 = (PairElectrode*)(force->pair);
    ptrPair_5->et_setup();

#if defined (_OPENMP)
  } else {
    if(strcmp(force->pair_style,"electrode/omp") == 0) {
      ptrPair_6 = (PairElectrodeOMP*)(force->pair);
      ptrPair_6->et_setup();
    }
#endif

  }

  
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_finter_mp(int vflag)
{
  TIMER_STAMP(EVB_EffPair,compute_finter);

  int *pos = evb_engine->complex_pos;
  int *cplx_index = evb_engine->complex_atom;
  int nall = atom->nlocal+atom->nghost;
 
  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  double **x = atom->x;
  double **f = atom->f;
  int id = 0;
  
  double xtmp, ytmp, ztmp, fx, fy, fz, fpair;
  double vdw1, vdw2;
  
  // Distribute work across partitions
  int nw = universe->nworlds;
  int istart, iend;
  if(nw > 1) {
    if(universe->iworld == 0) return;
    int nwm1 = nw - 1;
    int dn = inum / nwm1 + 1;
    istart = (universe->iworld - 1) * dn;
    iend   = istart + dn;
    if(universe->iworld == nwm1) iend = inum; // Last partition plays cleanup
  } else {
    istart = 0;
    iend = inum;
  }
  if(istart > inum-1) return; // No work for this slave partition
  if(iend > inum) iend = inum; // Make sure the last slave partition has the correct upper limit

  
  // If using real-space method for off-diagonals, then only include diagonal contributions in q_eff.
  if(evb_engine->flag_DIAG_QEFF) {

    // When short-range potential used for exch charges

    // lj/cut/coul/long forces for diagonal-diagonal (d-d) charge interactions
    // real-space approximation forces for d-o, o-d, and o-o charge interactions
    // array q has diagonal contribution to effective charges
    // array q_offdiag has off-diagonal contribution to effective charges
    
    for(int i=0; i<iend; i++) {
      const int atomi = ilist[i];
      const int jnum = numneigh[atomi];
      
      if(i<istart) {
	id+= jnum;
	continue;
      }
      
      const int *jlist = firstneigh[atomi];
      
      const double xtmp = x[atomi][0];
      const double ytmp = x[atomi][1];
      const double ztmp = x[atomi][2];
      
      for(int j=0; j<jnum; j++) {
	const int atomj = jlist[j]  & NEIGHMASK;
	
	const EVB_Complex * cplx_A = evb_engine->all_complex[cplx_index[atomi]-1];
	const EVB_Complex * cplx_B = evb_engine->all_complex[cplx_index[atomj]-1];
	
	fpair = 0.0;
	
	if(!cut_lj[id]) {
	  const double * C2A = cplx_A->Cs2;
	  const double * C2B = cplx_B->Cs2;
	  const double nA = cplx_A->nstate;
	  const double nB = cplx_B->nstate;
	  vdw1 = 0.0;
	  vdw2 = 0.0;
	  
	  for(int i=0; i<nA; i++) for(int j=0; j<nB; j++) {
	      const int itype = cplx_A->status[i].type[pos[atomi]];
	      const int jtype = cplx_B->status[j].type[pos[atomj]];
	      const double factor = C2A[i]*C2B[j];
	      vdw1 += factor*ptrA[itype][jtype];
	      vdw2 += factor*ptrB[itype][jtype];
	    }
	  
	  fpair += r6inv[id] * (vdw1 * r6inv[id] - vdw2);
	}
	
	if(!cut_coul[id]) {
	  fpair += pre_fcoul[id] * q[atomi] * q[atomj];
	  fpair += pre_fcoul_exch[id] * (q_offdiag[atomi] * (q[atomj] + q_offdiag[atomj]) + q[atomi] * q_offdiag[atomj]);
	}
	
	if( !cut_lj[id] || !cut_coul[id]) {
	  fpair *= r2inv[id];
	  fx = (xtmp - x[atomj][0]) * fpair;
	  fy = (ytmp - x[atomj][1]) * fpair;
	  fz = (ztmp - x[atomj][2]) * fpair;
	  
	  f[atomi][0] += fx; f[atomj][0] -= fx;
	  f[atomi][1] += fy; f[atomj][1] -= fy;
	  f[atomi][2] += fz; f[atomj][2] -= fz;
	} 
	
	id++;
      }
    }
    
  } else {
    
    for(int i=0; i<iend; i++) {
      const int atomi = ilist[i];
      const int jnum = numneigh[atomi];
      
      if(i<istart) {
	id+= jnum;
	continue;
      }
      
      const int *jlist = firstneigh[atomi];
      
      const double xtmp = x[atomi][0];
      const double ytmp = x[atomi][1];
      const double ztmp = x[atomi][2];
      
      for(int j=0; j<jnum; j++) {
	const int atomj = jlist[j]  & NEIGHMASK;
	
	const EVB_Complex * cplx_A = evb_engine->all_complex[cplx_index[atomi]-1];
	const EVB_Complex * cplx_B = evb_engine->all_complex[cplx_index[atomj]-1];
	
	fpair = 0.0;
	
	if(!cut_lj[id]) {
	  const double * C2A = cplx_A->Cs2;
	  const double * C2B = cplx_B->Cs2;
	  const double nA = cplx_A->nstate;
	  const double nB = cplx_B->nstate;
	  vdw1 = 0.0;
	  vdw2 = 0.0;
	  
	  for(int i=0; i<nA; i++) for(int j=0; j<nB; j++) {
	      const int itype = cplx_A->status[i].type[pos[atomi]];
	      const int jtype = cplx_B->status[j].type[pos[atomj]];
	      const double factor = C2A[i]*C2B[j];
	      vdw1 += factor*ptrA[itype][jtype];
	      vdw2 += factor*ptrB[itype][jtype];
	    }
	  
	  fpair += r6inv[id] * (vdw1 * r6inv[id] - vdw2);
	}
	
	if(!cut_coul[id]) fpair += pre_fcoul[id] * q[atomi] * q[atomj];
	
	if( !cut_lj[id] || !cut_coul[id]) {
	  fpair*= r2inv[id];
	  fx = (xtmp - x[atomj][0]) * fpair;
	  fy = (ytmp - x[atomj][1]) * fpair;
	  fz = (ztmp - x[atomj][2]) * fpair;
	  
	  f[atomi][0] += fx; f[atomj][0] -= fx;
	  f[atomi][1] += fy; f[atomj][1] -= fy;
	  f[atomi][2] += fz; f[atomj][2] -= fz;
	} 
	
	id++;
      }
    }
    
  } // if(flag_DIAG_QEFF)
  
  TIMER_CLICK(EVB_EffPair,compute_finter);
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_finter_supp_mp(int vflag)
{  
  int ptrPair_indx = 0;
  if(strcmp(force->pair_style,"table/lj/cut/coul/long") == 0) {
    ptrPair_1 = (PairTableLJCutCoulLong*)(force->pair);
    ptrPair_indx = 1;
  } else if(strcmp(force->pair_style,"gulp/coul/long") == 0) {
    ptrPair_4 = (PairGulpCoulLong*)(force->pair);
    ptrPair_indx = 4;
  } else if(strcmp(force->pair_style,"electrode") == 0) {
    ptrPair_5 = (PairElectrode*)(force->pair);
    ptrPair_indx = 5;

#if defined (_OPENMP)
  } else if(strcmp(force->pair_style,"electrode/omp") == 0) {
    ptrPair_6 = (PairElectrodeOMP*)(force->pair);
    ptrPair_indx = 6;
#endif

  } else error->all(FLERR,"Unsupported pair_style in EVB_EffPair::compute_finter_supp_mp().");

  int *pos = evb_engine->complex_pos;
  int *cplx_index = evb_engine->complex_atom;
  int nall = atom->nlocal + atom->nghost;

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int *type = atom->type;
  double *q_src = atom->q;
  int *atp_index = evb_type->atp_index;
  
  double **x = atom->x;
  double **f = atom->f;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int i,itype,j,jnum,jtype;
  int *jlist;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,factor_lj,factor_coul;
  double fp,fpair;
  
  int icplx,jcplx;
  
  // Distribute work across partitions
  int nw = universe->nworlds;
  int istart, iend;
  if(nw > 1) {
    if(universe->iworld == 0) return;
    int nwm1 = nw - 1;
    int dn = inum / nwm1 + 1;
    istart = (universe->iworld - 1) * dn;
    iend   = istart + dn;
    if(universe->iworld == nwm1) iend = inum; // Last partition plays cleanup
  } else {
    istart = 0;
    iend = inum;
  }
  if(istart > inum-1) return; // No work for this slave partition
  if(iend > inum) iend = inum; // Make sure the last slave partition has the correct upper limit

  for(int ii=istart; ii<iend; ii++) {
    i = ilist[ii];
    icplx = cplx_index[i] - 1;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jnum = numneigh[i];
    jlist = firstneigh[i];
    
    EVB_Complex *cplx_A = evb_engine->all_complex[icplx];
    
    // Loop over neighbors to atom i
    for(int jj=0; jj<jnum; jj++) {
      j = jlist[jj];
      
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;
      jcplx = cplx_index[j] - 1;
      
      EVB_Complex *cplx_B = evb_engine->all_complex[jcplx];
      
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      
      // Loop over states of each complex and compute energy

      if(ptrPair_indx == 1 || ptrPair_indx == 4) {

	if(rsq < cut_ljsq) {
	  fpair = 0.0;
	  for(int istate=0; istate<cplx_A->nstate; istate++) {
	    itype = cplx_A->status[istate].type[pos[i]];
	    
	    for(int jstate=0; jstate<cplx_B->nstate; jstate++) {
	      jtype = cplx_B->status[jstate].type[pos[j]];
	      
	      if(ptrPair_indx == 1)      fp = ptrPair_1->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	      else if(ptrPair_indx == 4) fp = ptrPair_4->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);

	      fpair+= fp * cplx_A->Cs2[istate] * cplx_B->Cs2[jstate];
	    } 
	  }

	  f[i][0] += delx*fpair;
	  f[i][1] += dely*fpair;
	  f[i][2] += delz*fpair;
	
	  f[j][0] -= delx*fpair;
	  f[j][1] -= dely*fpair;
	  f[j][2] -= delz*fpair;
	}

      } else if (ptrPair_indx == 5 || ptrPair_indx == 6) {
	double fi[3], fj[3];
	double ffi[3], ffj[3];
	ffi[0] = ffi[1] = ffi[2] = 0.0;
	ffj[0] = ffj[1] = ffj[2] = 0.0;

	// Loop over states of each complex and compute energy
	if(rsq < cut_coulsq) {
	  fpair = 0.0;
	  for(int istate=0; istate<cplx_A->nstate; istate++) {
	    itype = cplx_A->status[istate].type[pos[i]];
	    
	    for(int jstate=0; jstate<cplx_B->nstate; jstate++) {
	      jtype = cplx_B->status[jstate].type[pos[j]];
	      const double scale = cplx_A->Cs2[istate] * cplx_B->Cs2[jstate];
	      
	      if(ptrPair_indx == 5)      ptrPair_5->single_fpair_noljcoul(i,j,&(fi[0]),&(fj[0]));
#if defined (_OPENMP)
	      else if(ptrPair_indx == 6) ptrPair_6->single_fpair_noljcoul(i,j,&(fi[0]),&(fj[0]));
#endif
	      
	      ffi[0] += fi[0] * scale;
	      ffi[1] += fi[1] * scale;
	      ffi[2] += fi[2] * scale;

	      ffj[0] += fj[0] * scale;
	      ffj[1] += fj[1] * scale;
	      ffj[2] += fj[2] * scale;
	    } 
	  }
	  
	  f[i][0] += ffi[0];
	  f[i][1] += ffi[1];
	  f[i][2] += ffi[2];
	  
	  f[j][0] += ffj[0];
	  f[j][1] += ffj[1];
	  f[j][2] += ffj[2];
	}

      }

    }
  }

}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_fenv_mp(int vflag)
{
  int *pos = evb_engine->complex_pos;
  int *cplx_index = evb_engine->complex_atom;
  int nall = atom->nlocal+atom->nghost;
  int *type = atom->type;

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  double **x = atom->x;
  double **f = atom->f;
  double g_ewald = force->kspace->g_ewald;
  double qqrd2e = force->qqrd2e;
  double dx,dy,dz,r2,r,t,erfc,grij,expm2,r2i,r6i,fpair;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;

  // Distribute work across partitions
  int nw = universe->nworlds;
  int istart, iend;
  if(nw > 1) {
    if(universe->iworld == 0) return;
    int nwm1 = nw - 1;
    int dn = inum / nwm1 + 1;
    istart = (universe->iworld - 1) * dn;
    iend   = istart + dn;
    if(universe->iworld == nwm1) iend = inum; // Last partition plays cleanup
  } else {
    istart = 0;
    iend = inum;
  }
  if(istart > inum-1) return; // No work for this slave partition
  if(iend > inum) iend = inum; // Make sure the last slave partition has the correct upper limit

  // lj/cut/coul/long forces
  for(int i=istart; i<iend; i++) {
    const int atomi = ilist[i];
    
    const int jnum = numneigh[atomi];
    const int * jlist = firstneigh[atomi];
    
    const double xi = x[atomi][0];
    const double yi = x[atomi][1];
    const double zi = x[atomi][2];
    
    for(int j=0; j<jnum; j++) {
      int atomj = jlist[j];
      const double factor_coul = special_coul[sbmask(atomj)];
      const double factor_lj = special_lj[sbmask(atomj)];
      atomj &= NEIGHMASK;
      
      double vdw1,vdw2;
      int ia, ib, ienv;   
      EVB_Complex* cplx;
      
      if(cplx_index[atomi] && !cplx_index[atomj]) {
	ia = atomi; ib = atomj; ienv=0;
	cplx = evb_engine->all_complex[cplx_index[atomi]-1];
      } else if (!cplx_index[atomi] && cplx_index[atomj]) {
	ia = atomj; ib = atomi; ienv=1;
	cplx = evb_engine->all_complex[cplx_index[atomj]-1];
      }
      else continue;
      
      bool flag = false;
      fpair = 0.0;
      dx = xi-x[atomj][0];
      dy = yi-x[atomj][1];
      dz = zi-x[atomj][2];
      r2 = dx*dx + dy*dy +dz*dz;
      r2i = 1.0/r2;
      
      if(r2<cut_coulsq) {
	flag = true;
	r = sqrt(r2);
	grij = g_ewald * r;
	expm2 = exp(-grij*grij);
	t = 1.0 / (1.0+EWALD_P*grij);
	erfc = t*(A1+t*(A2+t*(A3+t*(A4+t*A5))))*expm2;
	fpair += qqrd2e*q[ia]*q[ib]/r*(erfc+EWALD_F*grij*expm2)*factor_coul;
      }
      
      if(r2<cut_ljsq) {
	flag = true;
	vdw1 = vdw2 = 0.0;
	const int btype = type[ib];
        
	const int n = cplx->nstate;
	const double * C2 = cplx->Cs2;
        
	for(int i=0; i<n; i++) {
	  const int atype = cplx->status[i].type[pos[ia]]; 
	  vdw1 += C2[i]*ptrA[atype][btype];
	  vdw2 += C2[i]*ptrB[atype][btype];
	}
	
	r6i = r2i*r2i*r2i;
	fpair += r6i*(vdw1*r6i-vdw2)*factor_lj;
      }
      
      if(flag) {
	fpair *= r2i;
	
	if(ienv) {
	  f[atomi][0] += dx*fpair;
	  f[atomi][1] += dy*fpair;
	  f[atomi][2] += dz*fpair;
	} else {
	  f[atomj][0] -= dx*fpair;
	  f[atomj][1] -= dy*fpair;
	  f[atomj][2] -= dz*fpair;
	}
      }     
    }
  }

  // If real-space approximation used for off-diagonal electrostatics, then include off-diagonal contribution to q_eff.
  // CGIS is only method currently supported
  if(evb_engine->flag_DIAG_QEFF) {
    
    const double cut = sqrt(cut_coulsq);
    const double cut_cgis = 0.81 * cut; // The 0.81 could be an input parameter

    const double A = -1.0 / (cut * cut * cut_cgis * (cut - cut_cgis));
    const double B = -1.0 / (cut * cut * cut_cgis);
    const double C =  1.0 / (cut * cut);
    const double rcutinv = 1.0 / cut;
  
    for(int i=istart; i<iend; i++) {
      const int atomi = ilist[i];
      
      const int jnum = numneigh[atomi];
      const int *jlist = firstneigh[atomi];
      
      const double xi = x[atomi][0];
      const double yi = x[atomi][1];
      const double zi = x[atomi][2];
      
      for(int j=0; j<jnum; j++) {
	int atomj = jlist[j];
	const double factor_coul = special_coul[sbmask(atomj)];
	const double factor_lj = special_lj[sbmask(atomj)];
	atomj &= NEIGHMASK;
	
	double vdw1,vdw2;
	int ia, ib, ienv;   
	EVB_Complex* cplx;
	
	if(cplx_index[atomi] && !cplx_index[atomj]) {
	  ia = atomi; ib = atomj; ienv=0;
	  cplx = evb_engine->all_complex[cplx_index[atomi]-1];
	}
	else if (!cplx_index[atomi] && cplx_index[atomj]) {
	  ia = atomj; ib = atomi; ienv=1;
	  cplx = evb_engine->all_complex[cplx_index[atomj]-1];
	}
	else continue;
	
	bool flag = false;
	fpair = 0.0;
	dx = xi-x[atomj][0];
	dy = yi-x[atomj][1];
	dz = zi-x[atomj][2];
	r2 = dx*dx + dy*dy +dz*dz;
	
	if(r2<cut_coulsq) {
	  flag = true;
	  r = sqrt(r2);
	  const double rinv = 1.0 / r;
	  const double r2inv = rinv * rinv;

	  fpair = r2inv;

	  if(r < cut_cgis) fpair += B * r;
	  else {
	    double dr = r - cut;
	    fpair += -A * dr * dr - B * dr - C;
	  }
	  
	  fpair*= qqrd2e * q_offdiag[ia] * q_offdiag[ib] * rinv;
	}
      
	if(flag) {
	  if(ienv) {
	    f[atomi][0] += dx*fpair;
	    f[atomi][1] += dy*fpair;
	    f[atomi][2] += dz*fpair;
	  } else {
	    f[atomj][0] -= dx*fpair;
	    f[atomj][1] -= dy*fpair;
	    f[atomj][2] -= dz*fpair;
	  }
	}
	
      }
    }
    
  } // if(flag_DIAG_QEFF)
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_fenv_supp_mp(int vflag)
{
  int ptrPair_indx = 0;
  if(strcmp(force->pair_style,"table/lj/cut/coul/long") == 0) {
    ptrPair_1 = (PairTableLJCutCoulLong*)(force->pair);
    ptrPair_indx = 1;
  } else if(strcmp(force->pair_style,"gulp/coul/long") == 0) {
    ptrPair_4 = (PairGulpCoulLong*)(force->pair);
    ptrPair_indx = 4;
  } else if(strcmp(force->pair_style,"electrode") == 0) {
    ptrPair_5 = (PairElectrode*)(force->pair);
    ptrPair_indx = 5;

#if defined (_OPENMP)
  } else if(strcmp(force->pair_style,"electrode/omp") == 0) {
    ptrPair_6 = (PairElectrodeOMP*)(force->pair);
    ptrPair_indx = 6;
#endif

  } else error->all(FLERR,"Unsupported pair_style in EVB_EffPair::compute_fenv_supp().");
  
  int *pos = evb_engine->complex_pos;
  int *cplx_index = evb_engine->complex_atom;
  int nall = atom->nlocal + atom->nghost;

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int *type = atom->type;
  double *q_src = atom->q;
  int *atp_index = evb_type->atp_index;
  
  double **x = atom->x;
  double **f = atom->f;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int i,itype,j,jnum,jtype;
  int *jlist;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,factor_lj,factor_coul;
  double fp,fpair;
  
  int icplx,jcplx;

  // Distribute work across partitions
  int nw = universe->nworlds;
  int istart, iend;
  if(nw > 1) {
    if(universe->iworld == 0) return;
    int nwm1 = nw - 1;
    int dn = inum / nwm1 + 1;
    istart = (universe->iworld - 1) * dn;
    iend   = istart + dn;
    if(universe->iworld == nwm1) iend = inum; // Last partition plays cleanup
  } else {
    istart = 0;
    iend = inum;
  }
  if(istart > inum-1) return; // No work for this slave partition
  if(iend > inum) iend = inum; // Make sure the last slave partition has the correct upper limit

  for(int ii=istart; ii<iend; ii++) {
    i = ilist[ii];
    icplx = cplx_index[i] - 1;
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jnum = numneigh[i];
    jlist = firstneigh[i];
    
    // Loop over neighbors to atom i
    for(int jj=0; jj<jnum; jj++) {
      j = jlist[jj];
      
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;
      jcplx = cplx_index[j] - 1;
      
      EVB_Complex *cplx;
      int ia,ib,ienv;

      if(cplx_index[i] && !cplx_index[j]) {
	ia = i; ib = j; ienv=0;
	cplx = evb_engine->all_complex[icplx];
      } else if (!cplx_index[i] && cplx_index[j]) {
	ia = j; ib = i; ienv=1;
	cplx = evb_engine->all_complex[jcplx];
      }
      else continue;
      
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      
      // Loop over states of each complex and compute energy

      if(ptrPair_indx == 1 || ptrPair_indx == 4) {

	if(rsq < cut_ljsq) {
	  fpair = 0.0;
	  jtype = type[ib];
	  for(int istate=0; istate<cplx->nstate; istate++) {
	    itype = cplx->status[istate].type[pos[ia]];
	    
	    if(ptrPair_indx == 1) fp = ptrPair_1->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	    else if(ptrPair_indx == 4) fp = ptrPair_4->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	    
	    fpair+= fp * cplx->Cs2[istate];
	  }

	  if(ienv) {
	    f[i][0] += delx*fpair;
	    f[i][1] += dely*fpair;
	    f[i][2] += delz*fpair;
	  } else {
	    f[j][0] -= delx*fpair;
	    f[j][1] -= dely*fpair;
	    f[j][2] -= delz*fpair;
	  }
	} 

      } else if(ptrPair_indx == 5 || ptrPair_indx == 6) {

	if(rsq < cut_coulsq) {
	  double ffi[3], ffj[3];
	  ffi[0] = ffi[1] = ffi[2] = 0.0;
	  ffj[0] = ffj[1] = ffj[2] = 0.0;

	  if(ptrPair_indx == 5)      ptrPair_5->single_fpair_noljcoul(i,j,&(ffi[0]),&(ffj[0]));
#if defined (_OPENMP)
	  else if(ptrPair_indx == 6) ptrPair_6->single_fpair_noljcoul(i,j,&(ffi[0]),&(ffj[0]));
#endif
	  
	  if(ienv) {
	    f[i][0] += ffi[0];
	    f[i][1] += ffi[1];
	    f[i][2] += ffi[2];
	  } else {
	    f[j][0] += ffj[0];
	    f[j][1] += ffj[1];
	    f[j][2] += ffj[2];
	  }
	}

      }

    }
  }

  // Calculate particle-electrode forces
  if(universe->iworld == 0) {
    if(ptrPair_indx == 5) ptrPair_5->et_compute_pln_frc();
#if defined (_OPENMP)
    if(ptrPair_indx == 6) ptrPair_6->et_compute_pln_frc();
#endif
  }

}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_pair_mp(int vflag)
{
  //  TIMER_STAMP(EVB_EffPair,compute_pair_mp);

  if(p_style == 1)      compute_pair_mp_1(vflag);
  else if(p_style == 2) compute_pair_mp_2(vflag);
  else if(p_style == 3) compute_pair_mp_3(vflag);
  else if(p_style == 4) compute_pair_mp_4(vflag);
  else if(p_style == 5) compute_pair_mp_5(vflag);
  else if(p_style == 6) compute_pair_mp_6(vflag);

  //  TIMER_CLICK(EVB_EffPair,compute_pair_mp);
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_pair_mp_1(int vflag)
{
  //  TIMER_STAMP(EVB_EffPair,compute_pair_mp);
  PairLJCutCoulLong* ptrPair = (PairLJCutCoulLong*)(force->pair);

  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  evdwl = ecoul = 0.0;
  int eflag = 1;
  ptrPair->ev_setup(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  double **cutsq = ptrPair->cutsq;
  double **cut_ljsq = ptrPair->cut_ljsq;
  double **lj1_orig = ptrPair->lj1;
  double **lj2_orig = ptrPair->lj2;
  double **lj3_orig = ptrPair->lj3;
  double **lj4_orig = ptrPair->lj4;
  double **offset = ptrPair->offset;
  double g_ewald = ptrPair->g_ewald;

  inum = ptrPair->list->inum;
  ilist = ptrPair->list->ilist;
  numneigh = ptrPair->list->numneigh;
  firstneigh = ptrPair->list->firstneigh;

  // Distribute work across partitions
  int nw = universe->nworlds;
  int istart, iend;
  if(nw > 1) {
    int nwm1 = nw - 1;
    int dn = inum / nw + 1;
    istart = universe->iworld * dn;
    iend   = istart + dn;
    if(universe->iworld == nwm1) iend = inum; // Last partition plays cleanup
  } else {
    istart = 0;
    iend = inum;
  }
  if(istart > inum-1) return; // No work for this slave partition
  if(iend > inum) iend = inum; // Make sure the last slave partition has the correct upper limit

  // loop over neighbors of my atoms
  for (ii = istart; ii < iend; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;

        if (rsq < cut_coulsq) {
	  r = sqrt(rsq);
	  grij = g_ewald * r;
	  expm2 = exp(-grij*grij);
	  t = 1.0 / (1.0 + EWALD_P*grij);
	  erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	  prefactor = qqrd2e * qtmp*q[j]/r;
	  forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	  if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else forcecoul = 0.0;
	
        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1_orig[itype][jtype]*r6inv - lj2_orig[itype][jtype]);
        } else forcelj = 0.0;

        fpair = (forcecoul + factor_lj*forcelj) * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

	if (rsq < cut_coulsq) {
	  ecoul = prefactor*erfc;
	  if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
	} else ecoul = 0.0;
	
	if (rsq < cut_ljsq[itype][jtype]) {
	  evdwl = r6inv*(lj3_orig[itype][jtype]*r6inv-lj4_orig[itype][jtype]) - offset[itype][jtype];
	  evdwl *= factor_lj;
	} else evdwl = 0.0;
	
        ptrPair->ev_tally(i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  if (ptrPair->vflag_fdotr) ptrPair->virial_fdotr_compute();
  //  TIMER_CLICK(EVB_EffPair,compute_pair_mp);
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_pair_mp_2(int vflag)
{
  //  TIMER_STAMP(EVB_EffPair,compute_pair_mp);

  error->universe_all(FLERR,"Not yet coded");

  //  TIMER_CLICK(EVB_EffPair,compute_pair_mp);
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_pair_mp_3(int vflag)
{
  //  TIMER_STAMP(EVB_EffPair,compute_pair_mp);
  PairTableLJCutCoulLong* ptrPair = (PairTableLJCutCoulLong*)(force->pair);

  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  evdwl = ecoul = 0.0;
  int eflag = 1;
  ptrPair->ev_setup(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  double **cutsq = ptrPair->cutsq;
  double **cut_ljsq = ptrPair->cut_ljsq;
  double **lj1_orig = ptrPair->lj1;
  double **lj2_orig = ptrPair->lj2;
  double **lj3_orig = ptrPair->lj3;
  double **lj4_orig = ptrPair->lj4;
  double **offset = ptrPair->offset;
  double g_ewald = ptrPair->g_ewald;

  inum = ptrPair->list->inum;
  ilist = ptrPair->list->ilist;
  numneigh = ptrPair->list->numneigh;
  firstneigh = ptrPair->list->firstneigh;

  // Distribute work across partitions
  int nw = universe->nworlds;
  int istart, iend;
  if(nw > 1) {
    int nwm1 = nw - 1;
    int dn = inum / nw + 1;
    istart = universe->iworld * dn;
    iend   = istart + dn;
    if(universe->iworld == nwm1) iend = inum; // Last partition plays cleanup
  } else {
    istart = 0;
    iend = inum;
  }
  if(istart > inum-1) return; // No work for this slave partition
  if(iend > inum) iend = inum; // Make sure the last slave partition has the correct upper limit

  // loop over neighbors of my atoms
  for (ii = istart; ii < iend; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;

        if (rsq < cut_coulsq) {
	  r = sqrt(rsq);
	  grij = g_ewald * r;
	  expm2 = exp(-grij*grij);
	  t = 1.0 / (1.0 + EWALD_P*grij);
	  erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	  prefactor = qqrd2e * qtmp*q[j]/r;
	  forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	  if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else forcecoul = 0.0;
	
        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1_orig[itype][jtype]*r6inv - lj2_orig[itype][jtype]);
        } else forcelj = 0.0;

        fpair = (forcecoul + factor_lj*forcelj) * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

	if (rsq < cut_coulsq) {
	  ecoul = prefactor*erfc;
	  if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
	} else ecoul = 0.0;
	
	if (rsq < cut_ljsq[itype][jtype]) {
	  evdwl = r6inv*(lj3_orig[itype][jtype]*r6inv-lj4_orig[itype][jtype]) - offset[itype][jtype];
	  evdwl *= factor_lj;
	} else evdwl = 0.0;

	if(rsq < cut_ljsq[itype][jtype]) {
	  evdwl+= ptrPair->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	  fpair = ptrPair->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	  
	  f[i][0] += delx*fpair;
	  f[i][1] += dely*fpair;
	  f[i][2] += delz*fpair;
	  if (newton_pair || j < nlocal) {
	    f[j][0] -= delx*fpair;
	    f[j][1] -= dely*fpair;
	    f[j][2] -= delz*fpair;
	  }
	}
	
        ptrPair->ev_tally(i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  if (ptrPair->vflag_fdotr) ptrPair->virial_fdotr_compute();
  //  TIMER_CLICK(EVB_EffPair,compute_pair_mp);
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_pair_mp_4(int vflag)
{
  //  TIMER_STAMP(EVB_EffPair,compute_pair_mp);

  PairGulpCoulLong *ptrPair = (PairGulpCoulLong*)(force->pair);

  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  evdwl = ecoul = 0.0;
  int eflag = 1;
  ptrPair->ev_setup(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  double **cutsq = ptrPair->cutsq;
  double **cut_ljsq = ptrPair->cutsq_ljcut;
  double **lj1_orig = ptrPair->lj1_ljcut;
  double **lj2_orig = ptrPair->lj2_ljcut;
  double **lj3_orig = ptrPair->lj3_ljcut;
  double **lj4_orig = ptrPair->lj4_ljcut;
  double **offset = ptrPair->offset_ljcut;
  double g_ewald = ptrPair->g_ewald;

  inum = ptrPair->list->inum;
  ilist = ptrPair->list->ilist;
  numneigh = ptrPair->list->numneigh;
  firstneigh = ptrPair->list->firstneigh;

  // Distribute work across partitions
  int nw = universe->nworlds;
  int istart, iend;
  if(nw > 1) {
    int nwm1 = nw - 1;
    int dn = inum / nw + 1;
    istart = universe->iworld * dn;
    iend   = istart + dn;
    if(universe->iworld == nwm1) iend = inum; // Last partition plays cleanup
  } else {
    istart = 0;
    iend = inum;
  }
  if(istart > inum-1) return; // No work for this slave partition
  if(iend > inum) iend = inum; // Make sure the last slave partition has the correct upper limit

  // loop over neighbors of my atoms
  for (ii = istart; ii < iend; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;

        if (rsq < cut_coulsq) {
	  r = sqrt(rsq);
	  grij = g_ewald * r;
	  expm2 = exp(-grij*grij);
	  t = 1.0 / (1.0 + EWALD_P*grij);
	  erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	  prefactor = qqrd2e * qtmp*q[j]/r;
	  forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	  if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else forcecoul = 0.0;
	
        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1_orig[itype][jtype]*r6inv - lj2_orig[itype][jtype]);
        } else forcelj = 0.0;

        fpair = (forcecoul + factor_lj*forcelj) * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

	if (rsq < cut_coulsq) {
	  ecoul = prefactor*erfc;
	  if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
	} else ecoul = 0.0;
	
	if (rsq < cut_ljsq[itype][jtype]) {
	  evdwl = r6inv*(lj3_orig[itype][jtype]*r6inv-lj4_orig[itype][jtype]) - offset[itype][jtype];
	  evdwl *= factor_lj;
	} else evdwl = 0.0;

	evdwl+= ptrPair->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	fpair = ptrPair->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	
	f[i][0] += delx*fpair;
	f[i][1] += dely*fpair;
	f[i][2] += delz*fpair;
	if (newton_pair || j < nlocal) {
	  f[j][0] -= delx*fpair;
	  f[j][1] -= dely*fpair;
	  f[j][2] -= delz*fpair;
	}
      
        ptrPair->ev_tally(i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  if (ptrPair->vflag_fdotr) ptrPair->virial_fdotr_compute();

  //  TIMER_CLICK(EVB_EffPair,compute_pair_mp);
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_pair_mp_5(int vflag)
{
  //  TIMER_STAMP(EVB_EffPair,compute_pair_mp);

  PairElectrode *ptrPair = (PairElectrode*)(force->pair);

  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  evdwl = ecoul = 0.0;
  double eimage = 0.0;
  int eflag = 1;
  ptrPair->ev_setup(eflag,vflag);

  // Electrode part for ENV: based on PairElectrode::et_compute_setup()
  ptrPair->complex_atom = evb_engine->complex_atom;
  ptrPair->eimage = 0.0;
  ptrPair->et_setup(); // Is this necessary at this stage?
  if(universe->iworld == 0) {
    ptrPair->et_compute(); // Only master partition evaluates plane interactions (for now)
    eimage += ptrPair->eimage;
  }

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  double **cutsq = ptrPair->cutsq;
  double **cut_ljsq = ptrPair->cut_ljsq;
  double **lj1_orig = ptrPair->lj1;
  double **lj2_orig = ptrPair->lj2;
  double **lj3_orig = ptrPair->lj3;
  double **lj4_orig = ptrPair->lj4;
  double **offset = ptrPair->offset;
  double g_ewald = ptrPair->g_ewald;

  const double cut_coul = sqrt(cut_coulsq);
  const double pos_lo = ptrPair->pos_lo;
  const double pos_hi = ptrPair->pos_hi;
  double ** x_image = ptrPair->x_image;
  double ** x_image2 = ptrPair->x_image2;

  inum = ptrPair->list->inum;
  ilist = ptrPair->list->ilist;
  numneigh = ptrPair->list->numneigh;
  firstneigh = ptrPair->list->firstneigh;

  // Distribute work across partitions
  int nw = universe->nworlds;
  int istart, iend;
  if(nw > 1) {
    int nwm1 = nw - 1;
    int dn = inum / nw + 1;
    istart = universe->iworld * dn;
    iend   = istart + dn;
    if(universe->iworld == nwm1) iend = inum; // Last partition plays cleanup
  } else {
    istart = 0;
    iend = inum;
  }
  if(istart > inum-1) return; // No work for this slave partition
  if(iend > inum) iend = inum; // Make sure the last slave partition has the correct upper limit

  // loop over neighbors of my atoms
  for (ii = istart; ii < iend; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    const double dr_i2l = x[i][2] - pos_lo;
    const double dr_i2h = pos_hi - x[i][2];
    const double ddr_i2l = cut_coul - dr_i2l;
    const double ddr_i2h = cut_coul - dr_i2h;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;

        if (rsq < cut_coulsq) {
	  r = sqrt(rsq);
	  grij = g_ewald * r;
	  expm2 = exp(-grij*grij);
	  t = 1.0 / (1.0 + EWALD_P*grij);
	  erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	  prefactor = qqrd2e * qtmp*q[j]/r;
	  forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	  if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else forcecoul = 0.0;
	
        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1_orig[itype][jtype]*r6inv - lj2_orig[itype][jtype]);
        } else forcelj = 0.0;

        fpair = (forcecoul + factor_lj*forcelj) * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
	
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

	if (rsq < cut_coulsq) {
	  ecoul = prefactor*erfc;
	  if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
	} else ecoul = 0.0;
	
	if (rsq < cut_ljsq[itype][jtype]) {
	  evdwl = r6inv*(lj3_orig[itype][jtype]*r6inv-lj4_orig[itype][jtype]) - offset[itype][jtype];
	  evdwl *= factor_lj;
	} else evdwl = 0.0;
	
        ptrPair->ev_tally(i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
      }

      // Image charge interactions
      const double dr_j2l = x[j][2] - pos_lo;
      const double dr_j2h = pos_hi - x[j][2];
      const double ddr_j2l = cut_coul - dr_j2l;
      const double ddr_j2h = cut_coul - dr_j2h;

      if(ddr_i2l > 0.0 && dr_j2l < ddr_i2l ) eimage += ptrPair->single_coul(x[i], x_image [j], q[i], -q[j], f[i]);
      if(ddr_i2h > 0.0 && dr_j2h < ddr_i2h ) eimage += ptrPair->single_coul(x[i], x_image2[j], q[i], -q[j], f[i]);
      
      if(ddr_j2l > 0.0 && dr_i2l < ddr_j2l ) eimage += ptrPair->single_coul(x[j], x_image [i], q[j], -q[i], f[j]);
      if(ddr_j2h > 0.0 && dr_i2h < ddr_j2h ) eimage += ptrPair->single_coul(x[j], x_image2[i], q[j], -q[i], f[j]);

    }
  }

  ptrPair->eng_coul += eimage * 0.5;

  if (ptrPair->vflag_fdotr) ptrPair->virial_fdotr_compute();

  //  TIMER_CLICK(EVB_EffPair,compute_pair_mp);
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_pair_mp_6(int vflag)
{
#if defined (_OPENMP)
  //  TIMER_STAMP(EVB_EffPair,compute_pair_mp);

  PairElectrodeOMP *ptrPair = (PairElectrodeOMP*)(force->pair);

  int i,ii,j,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  evdwl = ecoul = 0.0;
  double eimage = 0.0;
  int eflag = 1;
  ptrPair->ev_setup(eflag,vflag);

  // Electrode part for ENV: based on PairElectrode::et_compute_setup()
  ptrPair->complex_atom = evb_engine->complex_atom;
  ptrPair->eimage = 0.0;
  ptrPair->et_setup(); // Is this necessary at this stage?
  if(universe->iworld == 0) {
    ptrPair->et_compute(); // Only master partition evaluates plane interactions (for now)
    eimage += ptrPair->eimage;
  }

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  double **cutsq = ptrPair->cutsq;
  double **cut_ljsq = ptrPair->cut_ljsq;
  double **lj1_orig = ptrPair->lj1;
  double **lj2_orig = ptrPair->lj2;
  double **lj3_orig = ptrPair->lj3;
  double **lj4_orig = ptrPair->lj4;
  double **offset = ptrPair->offset;
  double g_ewald = ptrPair->g_ewald;

  const double cut_coul = sqrt(cut_coulsq);
  const double pos_lo = ptrPair->pos_lo;
  const double pos_hi = ptrPair->pos_hi;
  double ** x_image = ptrPair->x_image;
  double ** x_image2 = ptrPair->x_image2;

  inum = ptrPair->list->inum;
  ilist = ptrPair->list->ilist;
  numneigh = ptrPair->list->numneigh;
  firstneigh = ptrPair->list->firstneigh;

  // Distribute work across partitions
  int nw = universe->nworlds;
  int istart, iend;
  if(nw > 1) {
    int nwm1 = nw - 1;
    int dn = inum / nw + 1;
    istart = universe->iworld * dn;
    iend   = istart + dn;
    if(universe->iworld == nwm1) iend = inum; // Last partition plays cleanup
  } else {
    istart = 0;
    iend = inum;
  }
  if(istart > inum-1) return; // No work for this slave partition
  if(iend > inum) iend = inum; // Make sure the last slave partition has the correct upper limit

  // loop over neighbors of my atoms
  for (ii = istart; ii < iend; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    const double dr_i2l = x[i][2] - pos_lo;
    const double dr_i2h = pos_hi - x[i][2];
    const double ddr_i2l = cut_coul - dr_i2l;
    const double ddr_i2h = cut_coul - dr_i2h;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;

        if (rsq < cut_coulsq) {
	  r = sqrt(rsq);
	  grij = g_ewald * r;
	  expm2 = exp(-grij*grij);
	  t = 1.0 / (1.0 + EWALD_P*grij);
	  erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	  prefactor = qqrd2e * qtmp*q[j]/r;
	  forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	  if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else forcecoul = 0.0;
	
        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1_orig[itype][jtype]*r6inv - lj2_orig[itype][jtype]);
        } else forcelj = 0.0;

        fpair = (forcecoul + factor_lj*forcelj) * r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
	
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

	if (rsq < cut_coulsq) {
	  ecoul = prefactor*erfc;
	  if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
	} else ecoul = 0.0;
	
	if (rsq < cut_ljsq[itype][jtype]) {
	  evdwl = r6inv*(lj3_orig[itype][jtype]*r6inv-lj4_orig[itype][jtype]) - offset[itype][jtype];
	  evdwl *= factor_lj;
	} else evdwl = 0.0;
	
        ptrPair->ev_tally(i,j,nlocal,newton_pair,evdwl,ecoul,fpair,delx,dely,delz);
      }

      // Image charge interactions
      const double dr_j2l = x[j][2] - pos_lo;
      const double dr_j2h = pos_hi - x[j][2];
      const double ddr_j2l = cut_coul - dr_j2l;
      const double ddr_j2h = cut_coul - dr_j2h;

      if(ddr_i2l > 0.0 && dr_j2l < ddr_i2l ) eimage += ptrPair->single_coul(x[i], x_image [j], q[i], -q[j], f[i]);
      if(ddr_i2h > 0.0 && dr_j2h < ddr_i2h ) eimage += ptrPair->single_coul(x[i], x_image2[j], q[i], -q[j], f[i]);
      
      if(ddr_j2l > 0.0 && dr_i2l < ddr_j2l ) eimage += ptrPair->single_coul(x[j], x_image [i], q[j], -q[i], f[j]);
      if(ddr_j2h > 0.0 && dr_i2h < ddr_j2h ) eimage += ptrPair->single_coul(x[j], x_image2[i], q[j], -q[i], f[j]);

    }
  }

  ptrPair->eng_coul += eimage * 0.5;

  if (ptrPair->vflag_fdotr) ptrPair->virial_fdotr_compute();

  //  TIMER_CLICK(EVB_EffPair,compute_pair_mp);
#endif
}

/* ---------------------------------------------------------------------- */

#endif