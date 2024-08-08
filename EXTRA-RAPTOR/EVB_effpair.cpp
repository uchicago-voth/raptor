/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
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
#include "EVB_matrix_sci.h"
#include "EVB_timer.h"

#define _CRACKER_PAIR_LJ_CUT_COUL_CUT
#define _CRACKER_PAIR_LJ_CHARMM_COUL_LONG
#define _CRACKER_PAIR_LJ_CHARMMFSW_COUL_LONG
#define _CRACKER_PAIR_LJ_CUT_COUL_LONG
#ifdef _RAPTOR_GPU
#define _CRACKER_PAIR_LJ_CUT_COUL_LONG_GPU
#endif
#if defined (_OPENMP)
#define _CRACKER_PAIR_LJ_CUT_COUL_LONG_OMP
#endif
  #include "EVB_cracker.h"
#undef _CRACKER_PAIR_LJ_CUT_COUL_CUT
#undef _CRACKER_PAIR_LJ_CHARMM_COUL_LONG
#undef _CRACKER_PAIR_LJ_CUT_COUL_LONG
#undef _CRACKER_PAIR_LJ_CHARMMFSW_COUL_LONG

#ifdef _RAPTOR_GPU
#undef _CRACKER_PAIR_LJ_CUT_COUL_LONG_GPU
#endif
#if defined (_OPENMP)
#undef _CRACKER_PAIR_LJ_CUT_COUL_LONG_OMP
#endif

//#include "pair_table_lj_cut_coul_long.h"
//#include "pair_gulp_coul_long.h"

//#include "pair_electrode.h"
//#include "pair_electrode_omp.h"

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define EA1       0.254829592
#define EA2      -0.284496736
#define EA3       1.421413741
#define EA4      -1.453152027
#define EA5       1.061405429

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_EffPair::EVB_EffPair(LAMMPS *lmp, EVB_Engine *engine) : Pointers(lmp), EVB_Pointers(engine)
{
  max_atom = max_pair = 0;
  q = q_offdiag = q_exch = NULL;
  lj1 = lj2 = NULL;
  
  pre_ecoul = pre_fcoul = r2inv = r6inv = NULL;
  pre_ecoul_exch = pre_fcoul_exch = NULL;
  
  cut_coul = cut_lj = NULL;
  
  is_exch = NULL;
  p_style = 0;
}

/* ---------------------------------------------------------------------- */

EVB_EffPair::~EVB_EffPair()
{
  memory->sfree(q);
  memory->sfree(q_offdiag);
  memory->sfree(is_exch);
  memory->sfree(q_exch);
  memory->destroy(lj1);
  memory->destroy(lj2);

  memory->sfree(r2inv);
  memory->sfree(r6inv);
  memory->sfree(pre_ecoul);
  memory->sfree(pre_fcoul);
  memory->sfree(pre_ecoul_exch);
  memory->sfree(pre_fcoul_exch);
  memory->sfree(cut_coul);
  memory->sfree(cut_lj);
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::init()
{
  if(comm->me==0 && screen) fprintf(screen,"Init EVB-Effective-Pair module...\n");
  
  if(strcmp(force->pair_style,"lj/cut/coul/long")==0)
  {
    PairLJCutCoulLong* ptrPair = (PairLJCutCoulLong*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_ljsq[1][1];
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 0;
    p_style = 1;
  }
  else if(strcmp(force->pair_style,"lj/charmm/coul/long")==0)
  {
    PairLJCharmmCoulLong* ptrPair = (PairLJCharmmCoulLong*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_ljsq;
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 0;
    p_style = 2;
  }
  else if(strcmp(force->pair_style,"lj/charmmfsw/coul/long")==0)
  {
    PairLJCharmmfswCoulLong* ptrPair = (PairLJCharmmfswCoulLong*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_ljsq;
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 0;
    p_style = 2;
  }

  /*
  else if(strcmp(force->pair_style,"electrode")==0)
  {
    PairElectrode* ptrPair = (PairElectrode*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_ljsq[1][1];
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 1;
    p_style = 5;
  }
#if defined (_OPENMP)
  else if(strcmp(force->pair_style,"electrode/omp")==0)
  {
    PairElectrodeOMP* ptrPair = (PairElectrodeOMP*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_ljsq[1][1];
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 1;
    p_style = 6;  // We do non-omp pairwise calculation with mp parallelism.
  }
#endif
  else if(strcmp(force->pair_style,"lj/cut/coul/cut")==0)
  {
    PairLJCutCoulCut* ptrPair = (PairLJCutCoulCut*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;
    cut_coulsq = ptrPair->cut_coul_global * ptrPair->cut_coul_global;
    cut_ljsq = ptrPair->cut_ljsq[1][1];
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 0;
    p_style = 0;
  }
  else if(strcmp(force->pair_style,"lj/charmm/coul/long")==0)
  {
    PairLJCharmmCoulLong* ptrPair = (PairLJCharmmCoulLong*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;    
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_ljsq;    
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 0;
    p_style = 2;
  }
  else if(strcmp(force->pair_style,"table/lj/cut/coul/long")==0)
  {
    PairTableLJCutCoulLong* ptrPair = (PairTableLJCutCoulLong*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;    
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_ljsq[1][1];
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 1;
    p_style = 3;
  }
  else if(strcmp(force->pair_style,"gulp/coul/long")==0)
  {
    PairGulpCoulLong* ptrPair = (PairGulpCoulLong*)(force->pair);
    ptrLJ1 = ptrPair->lj3_ljcut;
    ptrLJ2 = ptrPair->lj4_ljcut;
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_global * ptrPair->cut_global;
    ptrA = ptrPair->lj1_ljcut;
    ptrB = ptrPair->lj2_ljcut;
    evb_engine->flag_EFFPAIR_SUPP = 1;
    p_style = 4;
  }
#if defined (_OPENMP)
  else if(strcmp(force->pair_style,"lj/cut/coul/long/omp")==0)
  {
    PairLJCutCoulLongOMP* ptrPair = (PairLJCutCoulLongOMP*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;    
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_ljsq[1][1];
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 0;
    p_style = 1;  // We do non-omp pairwise calculation with mp parallelism.
  }
#endif
#ifdef _RAPTOR_GPU
  else if(strcmp(force->pair_style,"lj/cut/coul/long/gpu")==0)
  {
    PairLJCutCoulLongGPU* ptrPair = (PairLJCutCoulLongGPU*)(force->pair);
    ptrLJ1 = ptrPair->lj3;
    ptrLJ2 = ptrPair->lj4;
    cut_coulsq = ptrPair->cut_coulsq;
    cut_ljsq = ptrPair->cut_ljsq[1][1];
    ptrA = ptrPair->lj1;
    ptrB = ptrPair->lj2;
    evb_engine->flag_EFFPAIR_SUPP = 0;
    p_style = 1;
  }
#endif

  else if(strcmp(force->pair_style, "hybrid/overlay")==0)
  { p_style = -1;
  }
*/
  else
  {
    char errline[255];
    sprintf(errline,"[EVB] EffPair has not been checked with this pair style: %s", force->pair_style);
    error->all(FLERR,errline);
  }
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::setup()
{
  int nall = atom->nlocal + atom->nghost;
  
  if(nall > max_atom) {
    max_atom = nall;
    q = (double*) memory->srealloc(q,sizeof(double)*max_atom,"EVB_EffPair:q");
    q_offdiag = (double*) memory->srealloc(q_offdiag,sizeof(double)*max_atom,"EVB_EffPair:q_offdiag");
    is_exch = (int*) memory->srealloc(is_exch, sizeof(int)*max_atom,"EVB_EffPair:is_exch");
    q_exch = (double*) memory->srealloc(q_exch, sizeof(double)*max_atom,"EVB_EffPair:q_exch");
    
    int natp = evb_type->natp;
    memory->grow(lj1,max_atom,natp,"EVB_EffPair:lj1");
    memory->grow(lj2,max_atom,natp,"EVB_EffPair:lj2");
  }

  memcpy(q, atom->q, sizeof(double)*(nall));
  memcpy(q_offdiag, atom->q, sizeof(double)*(nall));
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_q_eff(bool q_offdiag, bool q_scale)
{
  double *Cs = evb_complex->Cs;
  double *Cs2 = evb_complex->Cs2;
  int *cplx_list = evb_complex->cplx_list;
  int *parent_id = evb_complex->parent_id;
  int cplx_id = evb_complex->id-1;
  
  GET_OFFDIAG_EXCH(evb_complex);
  
  int n;
  if(evb_engine->ncenter>1) n = evb_complex->natom_cplx;
  else n = evb_complex->nlocal_cplx;

  for(int i=0; i<n; i++) {
    int id = cplx_list[i];
    q[id] = 0.0;
  }
  
  for(int i=0; i<evb_complex->nstate; i++) {
    double * q_src = evb_complex->status[i].q;
    
    for(int j=0; j<n; j++) {
      int id = cplx_list[j];
      q[id] += Cs2[i] * q_src[j];
    }
  }
  
  // If real-space method used for off-diagonals, don't include off-diagonal contribution in effective k-space CPLX-ENV calculations.
  // If calculating effective charges for output, then always include off-diagonal contribution and undo scaling by A_Rq used for k-space methods.
  if(!q_offdiag && evb_engine->flag_DIAG_QEFF) return;

  for(int i=0; i<evb_complex->nstate-1; i++) {
    double scale = 2*Cs[i+1]*Cs[parent_id[i+1]];
    
    if(q_scale) {
      double A_Rq = evb_engine->all_matrix[cplx_id]->e_offdiag[i][EOFF_ARQ];
      if(fabs(A_Rq) < 0.0000001) A_Rq = 1.0; // Prevent division by zero. If A_Rq is zero, then C will be zero.
      scale /= A_Rq;
    }
    
    for(int j=0; j<nexch_off[i]; j++) {
      int id = iexch_off[i][j];
      q[id] += scale * qexch_off[i][j];
    }
  }

  for(int i=0; i<evb_complex->nextra_coupling; i++) {
    double scale = 2*Cs[extra_i[i]]*Cs[extra_j[i]];
    
    if(q_scale) {
      double A_Rq = evb_engine->all_matrix[cplx_id]->e_extra[i][EOFF_ARQ];
      if(fabs(A_Rq) < 0.0000001) A_Rq = 1.0; // Prevent division by zero. If A_Rq is zero, then C will be zero.
      scale /= A_Rq;
    }
    
    for(int j=0; j<nexch_extra[i]; j++) {
      int id = iexch_extra[i][j]; 
      q[id] += scale * qexch_extra[i][j];
    }
  }
  
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_q_eff_offdiag(bool assign_q)
{
  double *Cs = evb_complex->Cs;
  double *Cs2 = evb_complex->Cs2;
  int *cplx_list = evb_complex->cplx_list;
  int *parent_id = evb_complex->parent_id;
  int cplx_id = evb_complex->id-1;
  
  GET_OFFDIAG_EXCH(evb_complex);
  
  int n;
  if(evb_engine->ncenter>1) n = evb_complex->natom_cplx;
  else n = evb_complex->nlocal_cplx;

  // Load off-diagonal charges into q
  if(assign_q) {

    for(int i=0; i<n; i++) {
      int id = cplx_list[i];
      q[id] = 0.0;
    }
    
    for(int i=0; i<evb_complex->nstate-1; i++) {
      double scale = 2*Cs[i+1]*Cs[parent_id[i+1]];
      
      for(int j=0; j<nexch_off[i]; j++) {
	int id = iexch_off[i][j];
	q[id] += scale * qexch_off[i][j];
      }
    }
    
    for(int i=0; i<evb_complex->nextra_coupling; i++) {
      double scale = 2*Cs[extra_i[i]]*Cs[extra_j[i]];
      
      for(int j=0; j<nexch_extra[i]; j++) {
	int id = iexch_extra[i][j]; 
	q[id] += scale * qexch_extra[i][j];
      }
    }

    // Load off-diagonal charges into q_offdiag
  } else {

    for(int i=0; i<n; i++) {
      int id = cplx_list[i];
      q_offdiag[id] = 0.0;
    }
    
    for(int i=0; i<evb_complex->nstate-1; i++) {
      double scale = 2*Cs[i+1]*Cs[parent_id[i+1]];
      
      for(int j=0; j<nexch_off[i]; j++) {
	int id = iexch_off[i][j];
	q_offdiag[id] += scale * qexch_off[i][j];
      }
    }
    
    for(int i=0; i<evb_complex->nextra_coupling; i++) {
      double scale = 2*Cs[extra_i[i]]*Cs[extra_j[i]];
      
      for(int j=0; j<nexch_extra[i]; j++) {
	int id = iexch_extra[i][j]; 
	q_offdiag[id] += scale * qexch_extra[i][j];
      }
    }

  }
  
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_vdw_eff()
{
#if defined (_OPENMP)
  compute_vdw_eff_omp();
  return;
#endif

  double *Cs2 = evb_complex->Cs2;
  int *cplx_list = evb_complex->cplx_list;
  int *parent_id = evb_complex->parent_id;
  
  int natp = evb_type->natp;
  int *atp_list = evb_type->atp_list;

  for(int i=0; i<evb_complex->natom_cplx; i++) {
    int id = cplx_list[i];
    memset(lj1[id],0,sizeof(double)*natp);
    memset(lj2[id],0,sizeof(double)*natp);
  }

  for(int i=0; i<evb_complex->nstate; i++) {
    int* type = evb_complex->status[i].type;
    double c = Cs2[i];
    
    for(int j=0; j<evb_complex->natom_cplx; j++) {
      int id = cplx_list[j];
      int tj = type[j];

      for(int t=0; t<natp; t++) {
	lj1[id][t] += c * ptrLJ1[tj][atp_list[t]];
	lj2[id][t] += c * ptrLJ2[tj][atp_list[t]];
      }
    }
  }
}


/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_para()
{
  // Both diagonal and off-diagonal electrostatics treated with same method, load diag+offdiag charges into evb_effpair->q
  compute_q_eff(true,false);

  // Load new vdw terms
  compute_vdw_eff();
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_para_qeff()
{
  // fprintf(stdout,"\n\nInside compute_para_qeff()\n");

  // Load new diagonal charges into q
  compute_q_eff(false,false);
  
  // Load new off-diagonal charges into q_offdiag
  compute_q_eff_offdiag(false);
  
  // Load new vdw terms
  compute_vdw_eff();
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::pre_compute()
{
#if defined (_OPENMP)
  pre_compute_omp();
  return;
#endif
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

  double xi,yi,zi,dx,dy,dz,r2,r,t,erfc,grij,expm2,prefactor;
    
  double g_ewald = force->kspace->g_ewald;
  double qqrd2e = force->qqrd2e;
  double **x = atom->x;
  
  int nall = atom->nlocal + atom->nghost;

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  
  int id = 0;

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
   
  for(int i=0; i<inum; i++) {
    int atomi = ilist[i];
	
    xi = x[atomi][0];
    yi = x[atomi][1];
    zi = x[atomi][2];

    int jnum = numneigh[atomi];
    int *jlist = firstneigh[atomi];
	
    for(int j=0; j<jnum; j++) {
	int atomj = jlist[j] & NEIGHMASK;

	dx = xi - x[atomj][0];
	dy = yi - x[atomj][1];
	dz = zi - x[atomj][2];

	r2 = dx*dx + dy*dy + dz*dz;
	r2inv[id] = 1.0/r2;

	if(r2>cut_coulsq) cut_coul[id]=true;
	else {
	  cut_coul[id]=false;
          
	  r = sqrt(r2);
	  grij = g_ewald * r;
	  expm2 = exp(-grij*grij);
	  t = 1.0 / (1.0 + EWALD_P*grij);
	  erfc = t * (EA1+t*(EA2+t*(EA3+t*(EA4+t*EA5)))) * expm2;
	  prefactor = qqrd2e / r;
	  pre_ecoul[id] = prefactor*erfc;
	  pre_fcoul[id] = prefactor*(erfc+EWALD_F*grij*expm2);
	  
	  if(is_Vij_ex==1) {
	    pre_ecoul_exch[id] = pre_ecoul[id];
	    pre_fcoul_exch[id] = pre_fcoul[id];

	  } else if(is_Vij_ex==4 || is_Vij_ex==5) {
	    double rinv = 1.0 / r;
	  
	    pre_ecoul_exch[id] = rinv;
	    pre_fcoul_exch[id] = r2inv[id];
	  
	    if(r < cut_cgis) { 
	      pre_ecoul_exch[id] += cgis_const - B2 * (r2 - cut_cgis2);
	      pre_fcoul_exch[id] += B * r;
	    } else { 
	      double dr = r - cut;
	      double dr2 = dr * dr;
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
    }
  }  
}

/* ---------------------------------------------------------------------- */
// Calculate non-bond interaction between atoms of complex==id and (complex
// !=id && complex!=0) by effective non-bond method. Effective parameters are
// used for atoms of (complex!=id && complex!=0)
 
void EVB_EffPair::compute_cplx(int cplx_id)
{
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
  
  int id = 0;
  evdw = ecoul = 0.0;

  // Diagonal and off-diagonal electrostatics use different short-range potentials
  if(evb_engine->flag_DIAG_QEFF) {

    int count = 0;
    for(int i=0; i<inum; i++) {
      const int atomi = ilist[i];
      
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
      
      for(int j=0; j<jnum; j++) {
	const int atomj = jlist[j] & NEIGHMASK;
	
	bool flag = false;
	double lj3, lj4, qqe;
	
	if (cplx_index[atomi]==cplx_id && cplx_index[atomj]!=cplx_id) {   
	  flag = true;
	  const int index = atp_index[type[atomi]];
	  
	  lj3 = lj1[atomj][index];
	  lj4 = lj2[atomj][index];
	  qqe  = q_src[atomi] * q[atomj] * pre_ecoul[id]; // Diagonal
	  qqe += q_src[atomi] * q_offdiag[atomj] * pre_ecoul_exch[id]; // Off-diagonal term
	}
	else if (cplx_index[atomi]!=cplx_id && cplx_index[atomj]==cplx_id) {
	  flag = true;
	  const int index = atp_index[type[atomj]];
	  
	  lj3 = lj1[atomi][index];
	  lj4 = lj2[atomi][index];
	  qqe  = q_src[atomj] * q[atomi] * pre_ecoul[id];
	  qqe += q_src[atomj] * q_offdiag[atomi] * pre_ecoul_exch[id];
	}
	
	if(flag) {
	  if(!cut_lj[id]) evdw += r6inv[id]*(lj3*r6inv[id]-lj4);
	  
	  if(!cut_coul[id]) ecoul += qqe;
	}
	
	id++;
      }
    }

    // Diagonal and off-diagonal electrostatics use same short-range potentials
  } else {

    int count = 0;
    for(int i=0; i<inum; i++) {
      const int atomi = ilist[i];
      
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
      
      for(int j=0; j<jnum; j++) {
	const int atomj = jlist[j] & NEIGHMASK;
	
	bool flag = false;
	double lj3, lj4, qq;
	
	if (cplx_index[atomi]==cplx_id && cplx_index[atomj]!=cplx_id) {   
	  flag = true;
	  const int index = atp_index[type[atomi]];
	  
	  lj3 = lj1[atomj][index];
	  lj4 = lj2[atomj][index];
	  qq = q_src[atomi]*q[atomj];

	}
	else if (cplx_index[atomi]!=cplx_id && cplx_index[atomj]==cplx_id) {
	  flag = true;
	  const int index = atp_index[type[atomj]];
	  
	  lj3 = lj1[atomi][index];
	  lj4 = lj2[atomi][index];
	  qq = q_src[atomj]*q[atomi];
	  
	}
	
	if(flag) {
	  if(!cut_lj[id]) evdw += r6inv[id]*(lj3*r6inv[id]-lj4);
	  
	  if(!cut_coul[id]) ecoul += pre_ecoul[id]*qq;
	}
	
	id++;
      }
    }
    
  }

  energy = evdw + ecoul;
}

/* ---------------------------------------------------------------------- */
// Calculate supplemental non-bond interaction between atoms of complex==id and (complex
// !=id && complex!=0) by effective non-bond method. 
 
void EVB_EffPair::compute_cplx_supp(int cplx_id)
{
  int ptrPair_indx = 0;
  if(strcmp(force->pair_style,"table/lj/cut/coul/long") == 0) {
    ptrPair_1 = (PairTableLJCutCoulLong*)(force->pair);
    ptrPair_indx = 1;
  } /* else if(strcmp(force->pair_style,"gulp/coul/long") == 0) {
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
  } */else error->all(FLERR,"Unsupported pair_style in EVB_EffPair::compute_cplx_supp().");
  
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
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int i,itype,j,jnum,jtype;
  int *jlist;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq,factor_lj,factor_coul;
  double eng,epair;
  
  int icplx,jcplx;
  
  for(int ii=0; ii<inum; ii++) {
    i = ilist[ii];
    icplx = cplx_index[i];
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
      jcplx = cplx_index[j];
      
      bool flag = false;
      if (icplx==cplx_id && jcplx!=cplx_id) flag = true;
      else if(icplx!=cplx_id && jcplx==cplx_id) flag = true;
      
      // Compute only if atoms are in different cplx's
      if(flag) {

	delx = xtmp - x[j][0];
	dely = ytmp - x[j][1];
	delz = ztmp - x[j][2];
	rsq = delx*delx + dely*dely + delz*delz;	
	
	if(ptrPair_indx == 1 || ptrPair_indx == 4) {
	  
	  // Loop over states of each complex and compute energy
	  eng = 0.0;
	  if(rsq < cut_ljsq) {
	    if(jcplx!=cplx_id) {
	      itype = type[i];
	      EVB_Complex *cplx_B = evb_engine->all_complex[jcplx-1];
	      
	      for(int jstate=0; jstate<cplx_B->nstate; jstate++) {
		jtype = cplx_B->status[jstate].type[pos[j]];
		
		//if(ptrPair_indx == 1)      epair = ptrPair_1->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
		//else if(ptrPair_indx == 4) epair = ptrPair_4->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
		
		eng+= epair * cplx_B->Cs2[jstate];
	      }
	    }
	    
	    if(icplx!=cplx_id) {
	      jtype = type[j];
	      EVB_Complex *cplx_A = evb_engine->all_complex[icplx-1];
	      
	      for(int istate=0; istate<cplx_A->nstate; istate++) {
		itype = cplx_A->status[istate].type[pos[i]];
		
		//if(ptrPair_indx == 1)      epair = ptrPair_1->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
		//else if(ptrPair_indx == 4) epair = ptrPair_4->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
		
		eng+= epair * cplx_A->Cs2[istate];
	      }
	    } 	    
	  }

	} /*else if(ptrPair_indx == 5 || ptrPair_indx == 6) {
	  
	  // Loop over states of each complex and compute energy
	  eng = 0.0;
	  if(rsq < cut_coulsq) {
	    if(jcplx!=cplx_id) {
	      itype = type[i];
	      EVB_Complex *cplx_B = evb_engine->all_complex[jcplx-1];
	      
	      for(int jstate=0; jstate<cplx_B->nstate; jstate++) {
		jtype = cplx_B->status[jstate].type[pos[j]];
		
		if(ptrPair_indx == 5)      epair = ptrPair_5->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
#if defined (_OPENMP)
		else if(ptrPair_indx == 6) epair = ptrPair_6->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
#endif
		
		eng+= epair * cplx_B->Cs2[jstate];
	      }
	    }
	    
	    if(icplx!=cplx_id) {
	      jtype = type[j];
	      EVB_Complex *cplx_A = evb_engine->all_complex[icplx-1];
	      
	      for(int istate=0; istate<cplx_A->nstate; istate++) {
		itype = cplx_A->status[istate].type[pos[i]];
		
		if(ptrPair_indx == 5)      epair = ptrPair_5->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
#if defined (_OPENMP)
		else if(ptrPair_indx == 6) epair = ptrPair_6->single_ener_noljcoul(i,j,itype,jtype,rsq,factor_lj);
#endif
		
		eng+= epair * cplx_A->Cs2[istate];
	      }
	    } 
	  }

	} */ //if(ptrPair_indx == 5)

	energy += eng;
      }
      
    }
  }

  // Calculate particle-electrode energy
  //if(ptrPair_indx == 5) energy += ptrPair_5->et_compute_pln_eng();
#if defined (_OPENMP)
  //if(ptrPair_indx == 6) energy += ptrPair_6->et_compute_pln_eng();
#endif
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::init_exch(bool is_extra, int index)
{
  if(is_extra) {
    nexch = evb_complex->nexch_extra[index];
    iexch = evb_complex->iexch_extra[index];
    qexch = evb_complex->qexch_extra[index];
  } else {
    nexch = evb_complex->nexch_off[index];
    iexch = evb_complex->iexch_off[index];
    qexch = evb_complex->qexch_off[index];
  }
  
  memset(is_exch,0,sizeof(int)*(atom->nlocal+atom->nghost));

  for(int i=0; i<nexch; i++) {
    is_exch[iexch[i]] = 1;
    q_exch[iexch[i]] = qexch[i];
  }
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_exch(int cplx_id)
{
  int *cplx_index = evb_engine->complex_atom;
  int nall = atom->nlocal+atom->nghost;

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  
  int id = 0;
  double qq;
  
  ecoul = 0.0;

  // Diagonal and off-diagonal electrostatics use different short-range potentials
  if(evb_engine->flag_DIAG_QEFF) {

    int count = 0;
    for(int i=0; i<inum; i++) {
      const int atomi = ilist[i];
      if (cplx_index[atomi]==0) continue;
      
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
  
      for(int j=0; j<jnum; j++) {
   
	// q_exch (current complex) * q_eff (other complex) * V_S (real-space approximation)
	// q has diagonal contribution to effective charge
	// q_offdiag has off-diagonal contribution to effective charge
	if(!cut_coul[id]) {
	  const int atomj = jlist[j] & NEIGHMASK;
	  if (is_exch[atomi] && cplx_index[atomj]!=cplx_id)
	    ecoul += q_exch[atomi] * (q[atomj] * pre_ecoul_exch[id] + q_offdiag[atomj] * pre_ecoul_exch[id]);
	  else if (cplx_index[atomi]!=cplx_id && is_exch[atomj])
	    ecoul += q_exch[atomj] * (q[atomi] * pre_ecoul_exch[id] + q_offdiag[atomi] * pre_ecoul_exch[id]);
	}
	
	id++;
      }
    }

    // Diagonal and off-diagonal electrostatics use same short-range potentials
  } else {

    for(int i=0; i<inum; i++) {
      const int atomi = ilist[i];
      if (cplx_index[atomi]==0) continue;
      
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
  
      for(int j=0; j<jnum; j++) {
   
	// The array q here has both diagonal and off-diagonal contributions.
	if(!cut_coul[id]) {
	  const int atomj = jlist[j] & NEIGHMASK;
	  if (is_exch[atomi] && cplx_index[atomj]!=cplx_id)      ecoul += q_exch[atomi] * q[atomj] * pre_ecoul_exch[id];
	  else if (cplx_index[atomi]!=cplx_id && is_exch[atomj]) ecoul += q_exch[atomj] * q[atomi] * pre_ecoul_exch[id];
	}
	
	id++;
      }
    }

  }

}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_exch_supp(int cplx_id)
{
  int ptrPair_indx = 0;
  if(strcmp(force->pair_style,"table/lj/cut/coul/long") == 0) return;

  else if(strcmp(force->pair_style,"gulp/coul/long") == 0) return;

  else if(strcmp(force->pair_style,"electrode") == 0) {
    ptrPair_5 = (PairElectrode*)(force->pair);
    ptrPair_indx = 5;

    //ecoul += ptrPair_5->compute_exch_image_eng(cplx_id);

#if defined (_OPENMP)
  } else if(strcmp(force->pair_style,"electrode/omp") == 0) {
    //ptrPair_6 = (PairElectrodeOMP*)(force->pair);
    ptrPair_indx = 6;

    //ecoul += ptrPair_6->compute_exch_image_eng(cplx_id);
#endif

  } else error->all(FLERR,"Unsupported pair_style in EVB_EffPair::compute_cplx_supp().");
}

/* ---------------------------------------------------------------------- */

void EVB_EffPair::compute_finter(int vflag)
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
  
  // If using real-space method for off-diagonals, then only include diagonal contributions in q_eff.
  if(evb_engine->flag_DIAG_QEFF) {

    // When short-range potential used for exch charges

    // lj/cut/coul/long forces for diagonal-diagonal (d-d) charge interactions
    // real-space approximation forces for d-o, o-d, and o-o charge interactions
    // array q has diagonal contribution to effective charges
    // array q_offdiag has off-diagonal contribution to effective charges
    for(int i=0; i<inum; i++) {
      const int atomi = ilist[i];
    
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
    
      const double xtmp = x[atomi][0];
      const double ytmp = x[atomi][1];
      const double ztmp = x[atomi][2];
      
      const int cplx_indx_i = cplx_index[atomi];
      const EVB_Complex * cplx_A = evb_engine->all_complex[cplx_indx_i-1];

      for(int j=0; j<jnum; j++) {
	const int atomj = jlist[j]  & NEIGHMASK;
      
	const int cplx_indx_j = cplx_index[atomj];
	const EVB_Complex * cplx_B = evb_engine->all_complex[cplx_indx_j-1];
      
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

    // lj/cut/coul/long forces for all interactions
    for(int i=0; i<inum; i++) {
      const int atomi = ilist[i];
    
      const int jnum = numneigh[atomi];
      const int * jlist = firstneigh[atomi];
    
      const double xtmp = x[atomi][0];
      const double ytmp = x[atomi][1];
      const double ztmp = x[atomi][2];
    
      const EVB_Complex * cplx_A = evb_engine->all_complex[cplx_index[atomi]-1];

      for(int j=0; j<jnum; j++) {
	const int atomj = jlist[j]  & NEIGHMASK;
      
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

void EVB_EffPair::compute_finter_supp(int vflag)
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

  } else error->all(FLERR,"Unsupported pair_style in EVB_EffPair::compute_finter_supp().");
  
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
  
  for(int ii=0; ii<inum; ii++) {
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
      
      if(ptrPair_indx == 1 || ptrPair_indx == 4) {
	
	// Loop over states of each complex and compute energy
	if(rsq < cut_ljsq) {
	  fpair = 0.0;
	  for(int istate=0; istate<cplx_A->nstate; istate++) {
	    itype = cplx_A->status[istate].type[pos[i]];
	    
	    for(int jstate=0; jstate<cplx_B->nstate; jstate++) {
	      jtype = cplx_B->status[jstate].type[pos[j]];
	      
	      //if(ptrPair_indx == 1)      fp = ptrPair_1->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	      //else if(ptrPair_indx == 4) fp = ptrPair_4->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	      
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
	      
	      //if(ptrPair_indx == 5)      ptrPair_5->single_fpair_noljcoul(i,j,&(fi[0]),&(fj[0]));
#if defined (_OPENMP)
	      //else if(ptrPair_indx == 6) ptrPair_6->single_fpair_noljcoul(i,j,&(fi[0]),&(fj[0]));
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

void EVB_EffPair::compute_fenv(int vflag)
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
  
  // lj/cut/coul/long forces
  for(int i=0; i<inum; i++) {
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
	erfc = t*(EA1+t*(EA2+t*(EA3+t*(EA4+t*EA5))))*expm2;
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
  
    // CGIS forces
    for(int i=0; i<inum; i++) {
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
	
	if(r2<cut_coulsq) {
	  flag = true;
	  r = sqrt(r2);
	  const double rinv = 1.0 / r;
	  const double r2inv = rinv * rinv;

	  fpair = r2inv;

	  if(r < cut_cgis) fpair += B * r;
	  else {
	    const double dr = r - cut;
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

void EVB_EffPair::compute_fenv_supp(int vflag)
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
  
  for(int ii=0; ii<inum; ii++) {
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
	    
	    //if(ptrPair_indx == 1)      fp = ptrPair_1->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	    //else if(ptrPair_indx == 4) fp = ptrPair_4->single_fpair_noljcoul(i,j,itype,jtype,rsq,factor_lj);
	    
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

	  //if(ptrPair_indx == 5)      ptrPair_5->single_fpair_noljcoul(i,j,&(ffi[0]),&(ffj[0]));
#if defined (_OPENMP)
	  //else if(ptrPair_indx == 6) ptrPair_6->single_fpair_noljcoul(i,j,&(ffi[0]),&(ffj[0]));
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
  // if(ptrPair_indx == 5) ptrPair_5->et_compute_pln_frc();
#if defined (_OPENMP)
  // if(ptrPair_indx == 6) ptrPair_6->et_compute_pln_frc();
#endif
}
