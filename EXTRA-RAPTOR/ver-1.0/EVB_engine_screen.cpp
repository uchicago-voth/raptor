/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Steve Tse
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "memory.h"
#include "error.h"
#include "update.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#define _CRACKER_PAIR  // cracked for GPU
#include "EVB_cracker.h"
#undef _CRACKER_PAIR
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"

#define _CRACKER_KSPACE
#include "EVB_cracker.h"
#undef _CRACKER_KSPACE

#define LMP_KSPACE_H
#include "ewald.h"
#include "pppm.h"

#define _CRACKER_NEIGHBOR
#include "EVB_cracker.h"
#undef _CRACKER_NEIGHBOR

#include "comm.h"
#include "universe.h"
#include "domain.h"

#include "EVB_engine.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_chain.h"
#include "EVB_reaction.h"
#include "EVB_list.h"
#include "EVB_matrix.h"
#include "EVB_matrix_full.h"
#include "EVB_matrix_sci.h"
#include "EVB_complex.h"
#include "EVB_cec.h"
#include "EVB_cec_v2.h"
#include "EVB_kspace.h"
#include "EVB_output.h"
#include "EVB_effpair.h"
#include "EVB_module_offdiag.h"
#include "EVB_module_rep.h"
#include "EVB_timer.h"

#include "mp_verlet.h"
#include "mp_verlet_sci.h"
#include "pair_evb.h"
#include "EVB_text.h"

#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif

#ifdef _RAPTOR_GPU
#include "timer.h"
#include "modify.h"
#include "fix_gpu.h"
#endif

using namespace LAMMPS_NS;


void EVB_Engine::screen_states(int vflag)
{
  TIMER_STAMP(EVB_Engine, screen_states);

  engine_indicator = ENGINE_INDICATOR_SCREEN;

  lmp_f  = atom->f;
  energy = 0.0;
  
  for(int i=0; i<6; i++) virial[i] = 0.0;
  clear_virial();
  
  /****************************************************************/
  /***    Single-complex subroutine   *****************************/
  /****************************************************************/
  
  for(int i=0; i<ncomplex; i++) {
    evb_complex = all_complex[i];
    evb_complex->build_cplx_map();
    evb_complex->setup_avec();
    evb_complex->setup_offdiag();
    evb_complex->update_pair_list(); // This is VERY important after pair-list got rebuilt
                                     // Because atom-special list isn't updated.
    
    check_for_special_atoms(); // AWGL : Identify if my rank contains complex atoms
    
    full_matrix->clear(true,vflag,true);
    
    evb_list->single_split();
    
    int iextra_coupling = 0;
    
    // Compute the pivot state
    if(evb_kspace) evb_kspace->energy = 0.0;
    evb_list->change_list(EVB_LIST);
    
    TIMER_STAMP(EVB_Engine, compute__Eii_compute_diagonal__pivot);
    if(!mp_verlet || mp_verlet->is_master==1) screen_compute_diagonal(vflag);
    TIMER_CLICK(EVB_Engine, compute__Eii_compute_diagonal__pivot);
    
    evb_matrix->save_ev_diag(MATRIX_PIVOT_STATE,vflag);
    
    evb_complex->save_avec(MATRIX_PIVOT_STATE);
    
    // Compute other states
    for(int j=1; j<evb_complex->nstate; j++) {

      // Build new state
      evb_complex->build_state(j);
      
      // Setup diagonal (force and repulsion) and calculate
      
      TIMER_STAMP(EVB_Engine, compute__Eii_compute_diagonal);
      if(!mp_verlet || mp_verlet->is_master==1) screen_compute_diagonal(vflag);
      TIMER_CLICK(EVB_Engine, compute__Eii_compute_diagonal);
      
      full_matrix->save_ev_diag(j,vflag);
      
      // Setup extra couplings and calculate
      int is_Vij_ex_save;
      if(evb_complex->extra_coupling[j]>0) {
	// Save original Zundel 
	int save_mol_A = evb_complex->molecule_A[j];
        
	// Build Zundels of extra couplings and compute
	for(int k=0; k<evb_complex->extra_coupling[j]; k++) {		                     
	  evb_complex->extra_i[iextra_coupling] = j-k-1;
	  evb_complex->extra_j[iextra_coupling] = j;
	  evb_complex->molecule_A[j] = evb_complex->molecule_B[j-k-1];
	  
	  int type_A=evb_reaction->reactant_B[evb_complex->reaction[j-k-1]-1];
	  int type_B=evb_reaction->product_B[evb_complex->reaction[j]-1];
          
	  for(int l=0; l<evb_reaction->nPair; l++)
	    if(evb_reaction->product_A[l]==type_A && evb_reaction->product_B[l]==type_B) {
	      evb_offdiag = all_offdiag[l];
	      break;
	    }
	  
	  SETUP_OFFDIAG_EXCH(extra,iextra_coupling);
          
	  evb_offdiag->index = evb_matrix->ndx_extra+iextra_coupling*10;
	  
	  is_Vij_ex_save = evb_offdiag->is_Vij_ex; // Save electrostatics flag
	  evb_offdiag->is_Vij_ex = 0; // Turn off electrostatics
	  
	  TIMER_STAMP(EVB_Engine, compute__evb_offdiag__compute__extra);
	  evb_offdiag->compute(vflag);
	  TIMER_CLICK(EVB_Engine, compute__evb_offdiag__compute__extra);
	  
	  evb_offdiag->is_Vij_ex = is_Vij_ex_save; // Reset electrostatics
	  
	  full_matrix->save_ev_offdiag(true,iextra_coupling,vflag);
	  
	  iextra_coupling++; 
	}
	
	// Restore original Zundel
	evb_complex->molecule_A[j] = save_mol_A;
      } // if(extra_coupling)
      
      // Setup off-diagonal and calculate
      evb_offdiag = all_offdiag[evb_complex->reaction[j]-1];
      SETUP_OFFDIAG_EXCH(off,j-1);
      
      evb_offdiag->index = evb_matrix->ndx_offdiag+10*(j-1);
      
      is_Vij_ex_save = evb_offdiag->is_Vij_ex; // Save electrostatics flag
      evb_offdiag->is_Vij_ex = 0; // Turn off electrostatics
      
      TIMER_STAMP(EVB_Engine, compute__evb_offdiag__compute);
      evb_offdiag->compute(vflag);
      TIMER_CLICK(EVB_Engine, compute__evb_offdiag__compute);
      
      evb_offdiag->is_Vij_ex = is_Vij_ex_save; // Reset electrostatics
      
      full_matrix->save_ev_offdiag(false,j-1,vflag);
      
      // Save status
      evb_complex->save_avec(j); 
    } // for(j<nstate)
    
    full_matrix->total_energy();

    full_matrix->diagonalize();
    evb_complex->cec_v2->compute();

    // Delete states with weight below user-specified tolerance
    screen_delete_states();

    if(ncomplex==1) {
      int pivot = 0; // Reload original parent state
      
      evb_complex->load_avec(pivot);
      evb_complex->update_mol_map();
      evb_complex->update_list();
      
      if(pivot!=0) { 
	nreact = 1;
	if(evb_kspace) qsqsum_sys = qsqsum_env+evb_complex->qsqsum;
      }
      
      energy = full_matrix->ground_state_energy + full_matrix->e_env[EDIAG_POT];
    } else {
      evb_complex->load_avec(0);
      evb_complex->update_mol_map();
      evb_complex->update_list();
    } // if(ncomplex==1)
    
    evb_list->change_list(SYS_LIST);
  }
  
  TIMER_CLICK(EVB_Engine, screen_states);
}

void EVB_Engine::screen_finalize()
{
  TIMER_STAMP(EVB_Engine, screen_finalize);

  /****************************************************************/
  /***   Get necessary variables and pointers *********************/
  /****************************************************************/
  
  int nlocal = atom->nlocal;
  natom      = nlocal + atom->nghost;
  nreact     = 0;
  evb_matrix = full_matrix;

  // Setup load-balancer for sci_mp simulations
  //if(mp_verlet_sci) setup_lb_mp();
  
  /****************************************************************/
  /***   Setup all EVB objects   **********************************/
  /****************************************************************/
  
  if(evb_effpair) evb_effpair->setup();
  evb_list->setup();

  full_matrix->setup();

  evb_reaction->setup();
  for(int i=0; i<ncomplex; i++) all_complex[i]->setup();
  
  //if(mp_verlet_sci) sci_comm_cplx_map();
  
  if(evb_kspace) {
    if(!mp_verlet_sci) for(int i=0; i<ncomplex; i++) all_complex[i]->build_cplx_map();
    evb_kspace->evb_setup();
  }

  TIMER_CLICK(EVB_Engine, screen_finalize);
}

/* ---------------------------------------------------------------------- */

void EVB_Engine::screen_compute_diagonal(int vflag)
{
  TIMER_STAMP(EVB_Engine, screen_compute_diagonal);
  
  int save_ago = neighbor->ago;
  
  if(force->pair) {
    force->pair->eng_vdwl = 0.0;
    force->pair->eng_coul = 0.0;
  }

  if(evb_kspace) force->kspace->energy = 0.0;
  
  if(force->bond) {
    if(bHybridBond) neighbor->ago = 0;else neighbor->ago = save_ago;
    force->bond->compute(true,vflag);
  }
  
  if(force->angle) {
    if(bHybridAngle) neighbor->ago = 0; else neighbor->ago = save_ago;
    force->angle->compute(true,vflag);
  }
  
  if(force->dihedral) { 
    if(bHybridDihedral) neighbor->ago = 0; else neighbor->ago = save_ago;
    force->dihedral->compute(true,vflag);
  }
  
  if(force->improper) {
    if(bHybridImproper) neighbor->ago = 0; else neighbor->ago = save_ago;
    force->improper->compute(true,vflag);
  }
  
  neighbor->ago = save_ago;

  TIMER_CLICK(EVB_Engine, screen_compute_diagonal);
}

void EVB_Engine::screen_delete_states()
{
  TIMER_STAMP(EVB_Engine, screen_delete_states);

  if(evb_complex->nstate > screen_max_list) {
    screen_max_list = evb_complex->nstate;
    memory->grow(screen_del_list, screen_max_list, "EVB_Engine:screen_del_list");
    memory->grow(screen_delsum_list, screen_max_list, "EVB_Engine:screen_delsum_list");
    memory->grow(screen_important_list, screen_max_list, "EVB_Engine:screen_important_list");
  }

  int del_size;
  
  del_size = 0;
  screen_important_list[0] = 1;
  for (int i=1;i<evb_complex->nstate;i++) {
    //   fprintf(screen,"i = %d  , Cs2[i] = %f by rank %d\n",i, evb_complex->Cs2[i], rank);                   
    if (evb_complex->Cs2[i]> screen_minP) {
      // fprintf(screen,"Important state %d by rank %d\n",i,rank);                   
      screen_important_list[i] = 1;
      int parentid = evb_complex->parent_id[i];
      while (screen_important_list[parentid]==0) {
	screen_important_list[parentid] = 1;
	parentid = evb_complex->parent_id[parentid];
      }

    }
    else screen_important_list[i] = 0;
  }
  int nextra_coupling = evb_complex->nextra_coupling;
      
  for (int i=0; i<nextra_coupling; i++) {
    int extra_j = evb_complex->extra_j[i];
    int extra_i = evb_complex->extra_i[i];
    double C = evb_complex->Cs[extra_j]*evb_complex->Cs[extra_i];
    
    if (C > screen_minP2) {
      screen_important_list[extra_i] = 1;
      screen_important_list[extra_j] = 1;
      int i_parentid = evb_complex->parent_id[extra_i];
      int j_parentid = evb_complex->parent_id[extra_j];
      while (screen_important_list[i_parentid]==0) {
	screen_important_list[i_parentid] = 1;
	i_parentid = evb_complex->parent_id[i_parentid];
      }

      while (screen_important_list[j_parentid]==0) {
	screen_important_list[j_parentid] = 1;
	j_parentid = evb_complex->parent_id[j_parentid];
      }
    } else {
      evb_complex->extra_coupling[extra_j] = 0;
      evb_complex->nextra_coupling--;
    }

  }

  for (int ii=0; ii<evb_complex->nstate; ii++) {
    if (ii==0) screen_delsum_list[0] = 0; 
    else screen_delsum_list[ii] = screen_delsum_list[ii-1];

    if(screen_important_list[ii]==0) {
      screen_delsum_list[ii]++;
      screen_del_list[del_size]=ii;
      del_size++;
    }
  }

#ifdef STATE_DECOMP
  MPI_Bcast(&del_size,1,MPI_INT,0,universe->uworld);
  MPI_Bcast(&(screen_del_list[0]),del_size,MPI_INT,0,universe->uworld);
#else
  MPI_Bcast(&del_size,1,MPI_INT,0,world);
  MPI_Bcast(&(screen_del_list[0]),del_size,MPI_INT,0,world);
#endif

  if (del_size>0) evb_complex->delete_multiplestates(screen_del_list,del_size);

  TIMER_CLICK(EVB_Engine, screen_delete_states);
}
