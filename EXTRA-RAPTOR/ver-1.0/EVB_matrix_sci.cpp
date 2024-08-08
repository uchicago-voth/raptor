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
#include "mp_verlet_sci.h"

#include "EVB_engine.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_chain.h"
#include "EVB_complex.h"
#include "EVB_reaction.h"
#include "EVB_list.h"
#include "EVB_matrix_sci.h"
#include "EVB_repul.h"
#include "EVB_offdiag.h"
#include "EVB_kspace.h"
#include "EVB_effpair.h"

#define _CRACKER_PAIR_LJ_CUT_COUL_LONG
#include "EVB_cracker.h"

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
    a[k][l]=h+s*(g-h*tau);


/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

EVB_MatrixSCI::EVB_MatrixSCI(LAMMPS *lmp, EVB_Engine *engine) : EVB_Matrix(lmp,engine)
{
  first_time_setup = 1;

  max_comm_ek = 0;
  comm_ek = NULL;
}

/* ---------------------------------------------------------------------- */

EVB_MatrixSCI::~EVB_MatrixSCI()
{
  memory->destroy(comm_ek);
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::setup()
{
  nstate = cplx->nstate;
  nextra = cplx->nextra_coupling;
  natom = cplx->natom_cplx;
  list = cplx->cplx_list;
  
  bool inc_atom = false;
  bool inc_state = false;
  bool inc_extra = false;
  
  if(natom>max_atom) { max_atom = natom; inc_atom = true; }
  if(nstate>max_state) { max_state = nstate; inc_state = true; }
  if(nextra>max_extra) { max_extra = nextra; inc_extra = true; }
  
  // Always allocate arrays on first pass with leading dimensions of at least 1
  if(first_time_setup) {
    inc_atom = true;
    first_time_setup = 0;
  }
  
  // Reallocate memory space
  
  if(inc_extra || inc_atom) memory->grow(f_extra_coupling,max_extra,max_atom, 3,"EVB_MatrixSCI:f_extra_coupling");
  
  if(inc_state || inc_atom) {
    memory->grow(f_diagonal,max_state,max_atom,3,"EVB_MatrixSCI:f_diagonal");
    memory->grow(f_off_diagonal,max_state-1,max_atom,3,"EVB_MatrixSCI:f_off_diagonal");
  }
}

/* ---------------------------------------------------------------------- */
// Only master partition calls this function to setup for all complexes
/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::setup_mp()
{
  nstate = cplx->nstate;
  nextra = cplx->nextra_coupling;
  natom = cplx->natom_cplx;
  list = cplx->cplx_list;
  
  bool inc_atom = false;
  bool inc_state = false;
  bool inc_extra = false;
  
  if(natom>max_atom) { max_atom = natom; inc_atom = true; }
  if(nstate>max_state) { max_state = nstate; inc_state = true; }
  if(nextra>max_extra) { max_extra = nextra; inc_extra = true; }
  
  // Skip force allocation if complex not owned
  if(!evb_engine->lb_cplx_owned[cplx->id]) return;
  
  // Reallocate memory space
  
  if(inc_extra || inc_atom) memory->grow(f_extra_coupling,max_extra,max_atom, 3,"EVB_MatrixSCI:f_extra_coupling");
  
  if(inc_state || inc_atom) {
    memory->grow(f_diagonal,max_state,max_atom,3,"EVB_MatrixSCI:f_diagonal");
    memory->grow(f_off_diagonal,max_state-1,max_atom,3,"EVB_MatrixSCI:f_off_diagonal");
  }
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::clear(bool eflag, bool vflag, bool fflag)
{  
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
    if(natom>0) for(int i=0; i<nstate; i++) {
	memset(&(f_diagonal[i][0][0]),0,sizeof(double)*3*natom);
	if(i<nstate-1) memset(&(f_off_diagonal[i][0][0]),0,sizeof(double)*3*natom);
      }
    
    if(nextra>0 && natom>0) for(int i=0; i<nextra; i++) memset(&(f_extra_coupling[i][0][0]),0,sizeof(double)*3*natom);
  }
}

/* ---------------------------------------------------------------------- */
// Only master partition call this function
/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::clear_mp(bool eflag, bool vflag, bool fflag)
{  
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
  
  if(!evb_engine->lb_cplx_owned[cplx->id]) return;
  
  if(fflag) {
    if(natom>0) for(int i=0; i<nstate; i++) {
	memset(&(f_diagonal[i][0][0]),0,sizeof(double)*3*natom);
	if(i<nstate-1) memset(&(f_off_diagonal[i][0][0]),0,sizeof(double)*3*natom);
      }
    
    if(nextra>0 && natom>0) for(int i=0; i<nextra; i++) memset(&(f_extra_coupling[i][0][0]),0,sizeof(double)*3*natom);
  }
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::copy_ev(bool vflag)
{
  EVB_Matrix* src = (EVB_Matrix*)(evb_engine->full_matrix);
  
  memcpy(energy, src->energy, sizeof(double)*size_e);
  if(nstate>1) memcpy(e_offdiag, src->e_offdiag, sizeof(double)*(nstate-1)*EOFF_NITEM);
  if(nextra>0) memcpy(e_extra, src->e_extra, sizeof(double)*nextra*EOFF_NITEM);
  
  if(vflag) {
    memcpy(v_diagonal,src->v_diagonal,sizeof(double)*nstate*6);
    if(nstate>1) memcpy(v_offdiag,src->v_offdiag,sizeof(double)*(nstate-1)*6);
    if(nextra>0) memcpy(v_extra,src->v_extra,sizeof(double)*nextra*6);
  }
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::copy_force()
{
  EVB_Matrix* src = (EVB_Matrix*)(evb_engine->full_matrix);
  
  for(int i=0; i<natom; i++) {
    int id = list[i];
    
    for(int n=0; n<nstate; n++) { VECTOR_COPY(f_diagonal[n][i],src->f_diagonal[n][id]); }
    for(int n=0; n<nstate-1; n++) { VECTOR_COPY(f_off_diagonal[n][i],src->f_off_diagonal[n][id]);}
    for(int n=0; n<nextra; n++) {VECTOR_COPY(f_extra_coupling[n][i],src->f_extra_coupling[n][id]);}
  }
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::accumulate_force()
{
  EVB_Matrix* src = (EVB_Matrix*)(evb_engine->full_matrix);
  
  for(int i=0; i<natom; i++) {
    int id = list[i];
    
    for(int n=0; n<nstate; n++) { VECTOR_SELF_ADD(f_diagonal[n][i],src->f_diagonal[n][id]); }
    for(int n=0; n<nstate-1; n++) { VECTOR_SELF_ADD(f_off_diagonal[n][i],src->f_off_diagonal[n][id]);}
    for(int n=0; n<nextra; n++) {VECTOR_SELF_ADD(f_extra_coupling[n][i],src->f_extra_coupling[n][id]);}
  }
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::sci_save_ev_diag(int index, bool vflag)
{
  sci_e_diagonal[index][SCI_EDIAG_VDW] = evb_effpair->evdw;
  sci_e_diagonal[index][SCI_EDIAG_COUL] = evb_effpair->ecoul;
  if(evb_kspace) sci_e_diagonal[index][SCI_EDIAG_KSPACE] = evb_kspace->energy;
  else  sci_e_diagonal[index][SCI_EDIAG_KSPACE] = 0.0;
  
  sci_e_diagonal[index][SCI_EDIAG_POT] = evb_effpair->energy + sci_e_diagonal[index][SCI_EDIAG_KSPACE];
}

void EVB_MatrixSCI::sci_save_ev_offdiag(bool is_extra,int index, bool vflag)
{
  double *eoff,*voff;
  
  if(is_extra) eoff = sci_e_extra+index;
  else eoff = sci_e_offdiag+index;
  
  (*eoff) = evb_effpair->ecoul;
  if(evb_kspace) (*eoff) += evb_kspace->off_diag_energy;
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::sci_total_energy()
{
  int size = nstate*SCI_EDIAG_NITEM;
  MPI_Allreduce(sci_e_diagonal, sci_allreduce,size, MPI_DOUBLE,MPI_SUM,world);
  memcpy(sci_e_diagonal, sci_allreduce, sizeof(double)*size);
  
  if(nstate>1) {
    MPI_Allreduce(sci_e_offdiag, sci_allreduce,nstate-1, MPI_DOUBLE,MPI_SUM,world);
    memcpy(sci_e_offdiag, sci_allreduce, sizeof(double)*(nstate-1));
  }
  
  if(nextra>0) {
    MPI_Allreduce(sci_e_extra, sci_allreduce,nextra, MPI_DOUBLE,MPI_SUM,world);
    memcpy(sci_e_extra, sci_allreduce, sizeof(double)*nextra);
  }
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::compute_hellmann_feynman()
{
  double **f_des = atom->f;
  double *Cs = cplx->Cs;
  double *Cs2= cplx->Cs2;

  GET_OFFDIAG_EXCH(cplx);

  if(natom==0) return;

  /*** Diagonal elements ***/

  for (int i=0;i<nstate;i++) {
    double **f_src = f_diagonal[i];
    
    for(int j=0; j<natom; j++) {
      int id = list[j];

      f_des[id][0]+=f_src[j][0]*Cs2[i];
      f_des[id][1]+=f_src[j][1]*Cs2[i];
      f_des[id][2]+=f_src[j][2]*Cs2[i];
    }
  }
  
  /*** Off-Diagonal elements ***/
  
  int *parent = cplx->parent_id;
  
  for (int i=0; i<nstate-1; i++) {
    double **f_src = f_off_diagonal[i];
    double C = 2*Cs[i+1]*Cs[parent[i+1]];
  
    for(int j=0; j<natom; j++) {
      int id = list[j];
      f_des[id][0]+=f_src[j][0]*C;
      f_des[id][1]+=f_src[j][1]*C;
      f_des[id][2]+=f_src[j][2]*C;
    }
  }
  
  /*** Extra couplings ***/
  
  for (int i=0; i<nextra; i++) {
    double **f_src = f_extra_coupling[i];
    double C = 2*Cs[extra_j[i]]*Cs[extra_i[i]];
  
    for(int j=0; j<natom; j++) {
      int id = list[j];
      f_des[id][0]+=f_src[j][0]*C;
      f_des[id][1]+=f_src[j][1]*C;
      f_des[id][2]+=f_src[j][2]*C;
    }
  }
}

/* ---------------------------------------------------------------------- */
// Communicate energies to master partition for the writing of output
/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::sci_comm_energy_mp(int index)
{
  if(comm->me != 0) return;

  MPI_Status status;

  int is_master = evb_engine->mp_verlet_sci->is_master;
  MPI_Comm block = evb_engine->mp_verlet_sci->block;

  int part_id = evb_engine->lb_cplx_master[index];
  if(part_id == 0) return; // Master partition already has this complex.

  EVB_Complex *cplx = evb_engine->all_complex[index];

  // diagonal + off-diagonal + ground_state
  int size = cplx->nstate * (EDIAG_NITEM + 1) + (cplx->nstate-1 + cplx->nextra_coupling) * EOFF_NITEM + 1;

  if(size > max_comm_ek) {
    max_comm_ek = size;
    memory->grow(comm_ek, size, "EVB_Matrix_SCI:comm_ek");
  }

  if(is_master) {
    // Recieve data
    MPI_Recv(&(comm_ek[0]), size, MPI_DOUBLE, part_id, 0, block, &status);
    // Unpack data
    int n = 0;
    ground_state_energy = comm_ek[n++];
    for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) e_diagonal[j][k] = comm_ek[n++];
    for(int j=0; j<cplx->nstate; j++) e_repulsive[j] = comm_ek[n++];
    for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) e_offdiag[j][k] = comm_ek[n++];
    for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) e_extra[j][k] = comm_ek[n++];
  } else if (universe->iworld == part_id) {
    // Pack data
    int n = 0;
    comm_ek[n++] = ground_state_energy;
    for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) comm_ek[n++] = e_diagonal[j][k];
    for(int j=0; j<cplx->nstate; j++) comm_ek[n++] = e_repulsive[j];
    for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) comm_ek[n++] = e_offdiag[j][k];
    for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) comm_ek[n++] = e_extra[j][k];
    if(n != size) error->universe_one(FLERR,"n != size in sci_comm_energy_mp()");
    
    // Send data
    MPI_Send(&(comm_ek[0]), size, MPI_DOUBLE, 0, 0, block);
  }
  
}


/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::copy_ev_full(bool vflag)
{
  EVB_Matrix* src = (EVB_Matrix*)(evb_engine->full_matrix);
  
  memcpy(src->energy, energy, sizeof(double)*size_e);
  if(nstate>1) memcpy(src->e_offdiag, e_offdiag, sizeof(double)*(nstate-1)*EOFF_NITEM);
  if(nextra>0) memcpy(src->e_extra, e_extra, sizeof(double)*nextra*EOFF_NITEM);
  
  if(vflag) {
    memcpy(src->v_diagonal,v_diagonal,sizeof(double)*nstate*6);
    if(nstate>1) memcpy(src->v_offdiag,v_offdiag,sizeof(double)*(nstate-1)*6);
    if(nextra>0) memcpy(src->v_extra,v_extra,sizeof(double)*nextra*6);
  }
}

/* ---------------------------------------------------------------------- */

void EVB_MatrixSCI::copy_force_full()
{
  EVB_Matrix* src = (EVB_Matrix*)(evb_engine->full_matrix);
  
  for(int i=0; i<natom; i++) {
    int id = list[i];
    
    for(int n=0; n<nstate; n++) { VECTOR_COPY(src->f_diagonal[n][id], f_diagonal[n][i]); }
    for(int n=0; n<nstate-1; n++) { VECTOR_COPY(src->f_off_diagonal[n][id], f_off_diagonal[n][i]);}
    for(int n=0; n<nextra; n++) {VECTOR_COPY(src->f_extra_coupling[n][id],f_extra_coupling[n][i]);}
  }
}
