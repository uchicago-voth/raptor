/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight
     Based on EVB_engine_screen written by Steve Tse
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


void EVB_Engine::screen_states_mp(int vflag)
{
  TIMER_STAMP(EVB_Engine, screen_states);

  engine_indicator = ENGINE_INDICATOR_SCREEN;

  lmp_f  = atom->f;
  energy = 0.0;

  for(int i=0; i<6; i++) virial[i] = 0.0;
  clear_virial();

  if(!screen_del_list_count) {
    memory->grow(screen_del_list_count, ncomplex, "EVB_Engine:screen_del_list_count");
    memory->grow(screen_del_list_global, ncomplex*MAX_STATE, "EVB_Engine:screen_del_list_global");
  }
  
  memset(screen_del_list_count, 0, sizeof(int)*ncomplex);
  memset(screen_del_list_global, 0, sizeof(int)*ncomplex*MAX_STATE);
  
  // ********************************************************************************/
  // Computation phase
  // ********************************************************************************/

  int task_pos = 0; // Current position in lb_tasklist
  for(int i=0; i<ncomplex; i++) {
    evb_complex = all_complex[i];

    if(lb_cplx_owned[i]) evb_complex->build_cplx_map();
    else {
      evb_complex->natom_cplx  = sci_cplx_count[i*3];
      evb_complex->nghost_cplx = sci_cplx_count[i*3+1];
      evb_complex->nlocal_cplx = sci_cplx_count[i*3+2];
    }

    evb_complex->setup_avec();
    evb_complex->setup_offdiag();

    // Skip rest of work for this complex if not owned by partition
    if(!lb_cplx_owned[i]) {
      task_pos+= evb_complex->nstate; // Update position in tasklist
      continue;
    }

    evb_complex->update_pair_list(); // This is VERY important after pair-list got rebuilt
                                     // Because atom-special list isn't updated.
    
    check_for_special_atoms(); // AWGL : Identify if my rank contains complex atoms
    
    all_matrix[i]->setup();
    all_matrix[i]->clear(true,vflag,true);
    
    full_matrix->clear(true,vflag,true);
    
    evb_list->single_split();
    
    int iextra_coupling = 0;
    
    // Build states
    if(evb_kspace) evb_kspace->energy = 0.0;
    evb_list->change_list(EVB_LIST);
    evb_complex->save_avec(MATRIX_PIVOT_STATE);

    for(int j=1; j<evb_complex->nstate; j++) {
      evb_complex->build_state(j);

      if(evb_complex->extra_coupling[j]>0) {      
      	for(int k=0; k<evb_complex->extra_coupling[j]; k++) {
      	  evb_complex->extra_i[iextra_coupling] = j-k-1;
      	  evb_complex->extra_j[iextra_coupling] = j;
      	  iextra_coupling++;
      	}
      }
      evb_complex->save_avec(j);
    }

    iextra_coupling = 0;    
    
    // Compute other states
    for(int j=0; j<evb_complex->nstate; j++) {
      // Check tasklist if assigned to work on current state
      if(lb_tasklist[task_pos] != universe->iworld) {
	task_pos++;
	continue;
      }

      // Load current state
      evb_complex->load_avec(j);
      evb_complex->update_mol_map();
      evb_complex->update_list();
      
      // Setup diagonal (force and repulsion) and calculate
      
      TIMER_STAMP(EVB_Engine, compute__Eii_compute_diagonal);
      if(!mp_verlet || mp_verlet->is_master==1) screen_compute_diagonal(vflag);
      TIMER_CLICK(EVB_Engine, compute__Eii_compute_diagonal);
      
      full_matrix->save_ev_diag(j,vflag);
      
      // Off-diagonals are not defined w/r to initial parent state. Skip if parent
      if(j==0) {
	task_pos++;
	continue;
      }

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
      
      // Update position in tasklist for load-balancer
      task_pos++;
    } // for(j<nstate)
    
    full_matrix->total_energy();

    // Save energies for communication phase
    all_matrix[i]->clear(true,vflag,true);
    all_matrix[i]->copy_ev(vflag);

    evb_complex->load_avec(0);
    evb_complex->update_mol_map();
    evb_complex->update_list();
    
    evb_list->change_list(SYS_LIST);
  }

  // ********************************************************************************/
  // Communication phase
  // ********************************************************************************/

  task_pos = 0;
  for(int i=0; i<ncomplex; i++) {
    evb_complex = all_complex[i];

    // Skip rest of work for this complex if not owned by partition
    if(!lb_cplx_owned[i]) {
      task_pos+= evb_complex->nstate; // Update position in tasklist
      // if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Skipping communication of complex in compute. New task_pos is %i.\n",
      // 			       universe->iworld,comm->me,task_pos);
      continue;
    }

    // Restore energies to full_matrix
    all_matrix[i]->copy_ev_full(vflag);

    if(lb_cplx_split[i]) sci_comm_lb_mp_1();

    // Only the partition that owns the complex should diagonalize the Hamiltonian
    if(universe->iworld == lb_cplx_master[i]) {
      full_matrix->diagonalize();
      
      // Generate list of states with weight below user-specified tolerance, but don't delete yet.
      screen_delete_states_mp(i);
    }

    task_pos+= evb_complex->nstate; // Update position in tasklist
  }

  // Masters of complexes sync list of states to delete with all partitions
  // All partitions then delete states for all complexes.
  screen_delete_states_comm_mp();
  
  TIMER_CLICK(EVB_Engine, screen_states);
}

/* ---------------------------------------------------------------------- */

void EVB_Engine::screen_delete_states_mp(int indx)
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

  // Add list of states for current complex to global list of all complexes.
  screen_del_list_count[indx] = del_size;
  for(int i=0; i<del_size; i++) screen_del_list_global[indx*MAX_STATE+i] = screen_del_list[i];

  TIMER_CLICK(EVB_Engine, screen_delete_states);
}

void EVB_Engine::screen_delete_states_comm_mp()
{

  // Accumulate global list of states to delete from all complexes across all partitions
  if(comm->me == 0) {
    MPI_Allreduce(MPI_IN_PLACE, &(screen_del_list_count[0]), ncomplex, MPI_INT, MPI_SUM, mp_verlet_sci->block);
    MPI_Allreduce(MPI_IN_PLACE, &(screen_del_list_global[0]), ncomplex*MAX_STATE, MPI_INT, MPI_SUM, mp_verlet_sci->block);
  }

  MPI_Bcast(&(screen_del_list_count[0]), ncomplex, MPI_INT, 0, world);
  MPI_Bcast(&(screen_del_list_global[0]), ncomplex*MAX_STATE, MPI_INT, 0, world);
  
  // Loop over complexes and delete states
  for(int i=0; i<ncomplex; i++) if(screen_del_list_count[i] > 0) {
      evb_complex = all_complex[i];
      evb_complex->delete_multiplestates(screen_del_list_global+i*MAX_STATE,screen_del_list_count[i]);
    }
}
