/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight
------------------------------------------------------------------------- */

#ifdef RAPTOR_MPVERLET

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "memory.h"
#include "error.h"
#include "update.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "output.h"

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
//#include "pair_evb.h"
#include "EVB_text.h"

// Electrode model
//#include "pair_electrode.h"

#define MAX_CIRCLE 10
#define CNVG_LIMIT 1e-6

#define KSPACE_DEFAULT    0 // Hellman-Feynman forces for Ewald
#define PPPM_HF_FORCES    1 // Hellman-Feynman forces for PPPM
#define PPPM_ACC_FORCES   2 // Approximate (acc) forces for PPPM. 
#define PPPM_POLAR_FORCES 3 // ACC forces plus an additional polarization force on complex atoms for PPPM.

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   sci_mp enabled version of compute()
   ---------------------------------------------------------------------- */

void EVB_Engine::compute_sci_mp(int vflag)
{
  TIMER_STAMP(EVB_Engine, compute_sci_mp);

  engine_indicator = ENGINE_INDICATOR_COMPUTE;

  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    if(universe->me==0) fprintf(stdout,"(%i,%i)  \n\nInside compute_sci_mp()\n",universe->iworld,comm->me);
  }
  
  lmp_f      = atom->f;
  energy     = 0.0;
  
  for(int i=0; i<6; i++) virial[i] = 0.0;
  clear_virial();

  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    for(int iproc=0; iproc<universe->nprocs; iproc++) {
      if(iproc == universe->me) {
	if(iproc == 0) fprintf(stdout,"\n\nStates at start of compute_sci_mp()\n");
	fprintf(stdout,"(%i,%i)  nstates = ",universe->iworld,comm->me);
	for(int i=0; i<ncomplex; i++) fprintf(stdout," %i",all_complex[i]->nstate);
	fprintf(stdout,"   nextra = ");
	for(int i=0; i<ncomplex; i++) fprintf(stdout," %i",all_complex[i]->nextra_coupling);
	fprintf(stdout,"\n");
      }
      MPI_Barrier(universe->uworld);
    }
  }

  // ********************************************************************************/
  // Computation phase
  // ********************************************************************************/

  int task_pos = 0; // Current position in lb_tasklist
  for(int i=0; i<ncomplex; i++) {
    
    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Starting complex %i in compute: lb_cplx_owned= %i.\n",
			     universe->iworld,comm->me,i,lb_cplx_owned[i]);
    // ********************************************************************************/
    // All partitions currently do some initialization for all complexes
    // ********************************************************************************/

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
      // if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Skipping complex in compute. New task_pos is %i.\n",
      // 			       universe->iworld,comm->me,task_pos);
      continue;
    }

    evb_complex->update_pair_list(); // This is VERY important after pair-list got rebuilt
                                     // Because atom-special list isn't updated
    check_for_special_atoms();       // AWGL : Identify if my rank contains complex atoms

    all_matrix[i]->setup();
    
    all_matrix[i]->clear(true,vflag,true);
    full_matrix->clear(true,vflag,true);
    
    evb_list->single_split();
    int iextra_coupling = 0;

    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Setting up complex %i in compute.\n",universe->iworld,comm->me,i);    

    // ********************************************************************************/
    // The only quantity needed from the environment at this point is the discretized charged density
    // Energies and forces for environment are calculated in initialize_sci_mp().
    if(natom>0) atom->f = full_matrix->f_env;

    TIMER_STAMP(EVB_Engine, compute__evb_kspace__compute_env);
    if(!mp_verlet || mp_verlet->is_master==0) if(evb_kspace) evb_kspace->compute_env_density(vflag);
    TIMER_CLICK(EVB_Engine, compute__evb_kspace__compute_env);

    // Initialize some pair_styles; needed because not calling compute_diagonal() for environment.
    evb_list->change_list(ENV_LIST);
    evb_effpair->setup_pair_mp();

    evb_repulsive = NULL;
    full_matrix->save_ev_diag(MATRIX_ENV,vflag);
    // ********************************************************************************/

    if(evb_kspace) evb_kspace->energy = 0.0;
    evb_list->change_list(EVB_LIST);
    evb_complex->save_avec(MATRIX_PIVOT_STATE);
    for(int j=1; j<evb_complex->nstate; j++) {
      evb_complex->build_state(j);
      if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Building state %i/%i of complex %i in compute, nextra= %i.\n",
			       universe->iworld,comm->me,j,evb_complex->nstate-1,i,evb_complex->extra_coupling[j]);
      if(evb_complex->extra_coupling[j]>0) {      
      	for(int k=0; k<evb_complex->extra_coupling[j]; k++) {
      	  evb_complex->extra_i[iextra_coupling] = j-k-1;
      	  evb_complex->extra_j[iextra_coupling] = j;
      	  iextra_coupling++;
      	}
      }
      if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Building state %i/%i of complex %i in compute.\n",
			       universe->iworld,comm->me,j,evb_complex->nstate-1,i);
      evb_complex->save_avec(j);
    }

    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished building states for complex %i in compute: nstate= %i  nextra= %i  iextra_coupling= %i.\n",
			     universe->iworld,comm->me,i,evb_complex->nstate,evb_complex->nextra_coupling,iextra_coupling);

    iextra_coupling = 0;
    
    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Starting states in complex %i in compute.\n",universe->iworld,comm->me,i);

    // Compute states
    for(int j=0; j<evb_complex->nstate; j++) {
      // Check tasklist if assigned to work on current state
      if(lb_tasklist[task_pos] != universe->iworld) {
	task_pos++;
	continue;
      }

      //if(debug_sci_mp) fprintf(stdout,"(%i,%i)  i= %i  j= %i  task_pos= %i  lb_task_list= %i\n",
      // 			       universe->iworld,comm->me,i,j,task_pos,lb_tasklist[task_pos]);
      
      // Load current state
      evb_complex->load_avec(j);
      evb_complex->update_mol_map();
      evb_complex->update_list();

      // Setup diagonal and calculate
      if (natom>0) atom->f = full_matrix->f_diagonal[j];
      
      TIMER_STAMP(EVB_Engine, compute__Eii_compute_diagonal);
      if(!mp_verlet || mp_verlet->is_master==1) {
	compute_diagonal(vflag);
	if(EFIELD_flag) compute_efield(1,i+1);
      }
      TIMER_CLICK(EVB_Engine, compute__Eii_compute_diagonal);
      
#ifdef DLEVB_MODEL_SUPPORT
      if(!mp_verlet || mp_verlet->is_master==1) if(EVB14) compute_LJ14(vflag);
#endif
      if(!mp_verlet || mp_verlet->is_master==1) compute_repulsive(vflag);
      
      TIMER_STAMP(EVB_Engine, compute__Eii_evb_kspace_compute_cplx);
      if(!mp_verlet || mp_verlet->is_master==0) if(evb_kspace) evb_kspace->compute_cplx(vflag);
      TIMER_CLICK(EVB_Engine, compute__Eii_evb_kspace_compute_cplx);
      
      full_matrix->save_ev_diag(j,vflag);

      // Off-diagonals are not defined w/r to initial parent state. Skip rest of loop
      if(j==0) {
	task_pos++;
	continue;
      }

      // if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished building state %i for complex %i\n",
      // 			       universe->iworld,comm->me,j,i);

      // Setup extra couplings and calculate
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
	  
       	  if(natom>0) atom->f = full_matrix->f_extra_coupling[iextra_coupling];
       	  SETUP_OFFDIAG_EXCH(extra,iextra_coupling);
	  
       	  evb_offdiag->index = evb_matrix->ndx_extra+iextra_coupling*10;
	  
       	  TIMER_STAMP(EVB_Engine, compute__evb_offdiag__compute__extra);
       	  evb_offdiag->compute(vflag);
       	  TIMER_CLICK(EVB_Engine, compute__evb_offdiag__compute__extra);
	  
       	  full_matrix->save_ev_offdiag(true,iextra_coupling,vflag);
	  
       	  iextra_coupling++; 
       	}
	
       	// Restore original Zundel
       	evb_complex->molecule_A[j] = save_mol_A;
      } // if(extra_coupling)
      
      // Setup off-diagonal and calculate
      evb_offdiag = all_offdiag[evb_complex->reaction[j]-1];
      if (natom>0) atom->f = evb_matrix->f_off_diagonal[j-1];
      SETUP_OFFDIAG_EXCH(off,j-1);
      
      //evb_offdiag->finite_difference_test();
      evb_offdiag->index = evb_matrix->ndx_offdiag+10*(j-1);
      
      TIMER_STAMP(EVB_Engine, compute__evb_offdiag__compute);
      evb_offdiag->compute(vflag);
      TIMER_CLICK(EVB_Engine, compute__evb_offdiag__compute);
      
      full_matrix->save_ev_offdiag(false,j-1,vflag);

      // if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished building off-diagonal %i for complex %i\n",
      // 			       universe->iworld,comm->me,j,i);

      // Update position in tasklist for load-balancer
      task_pos++;
    } // for(j<nstate)

    full_matrix->total_energy();

    // Save energies and forces for communication phase
    all_matrix[i]->clear(true,vflag,true);
    all_matrix[i]->copy_ev(vflag);
    all_matrix[i]->copy_force();
    
    evb_complex->load_avec(0);
    evb_complex->update_mol_map();
    evb_complex->update_list();
    
    evb_list->change_list(SYS_LIST);

    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished with complex %i in compute.\n",universe->iworld,comm->me,i);
  } // for(i<ncomplex)

  // ********************************************************************************/
  // Communication phase
  // ********************************************************************************/

  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    if(universe->me==0) fprintf(stdout,"\n\n(%i,%i)  Beginning communication phase of initialization\n",universe->iworld,comm->me);
  }

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

    // Restore energies and forces from full_matrix
    all_matrix[i]->copy_ev_full(vflag);
    all_matrix[i]->copy_force_full(); // Exactly what forces are used later ??

    // If complex was split across partitions, then slave partitions send their info to 
    //   master rank of partition that owns complex. This master rank will calculate 
    //   ground state eigenvector for call to sci_comm_evec_mp().
    // Partitions wait here till all working on current complex are ready

    if(debug_sci_mp) fprintf(stdout,"(%i,%i) complex= %i  lb_cplx_split= %i\n",universe->iworld,comm->me,i,lb_cplx_split[i]);

    if(lb_cplx_split[i]) sci_comm_lb_mp_1();
   
    if(debug_sci_mp) {
      fprintf(stdout,"(%i,%i)  Communication of energies to diagonalize has completed for complex %i in compute. cplx_master= %i...\n",
	      universe->iworld,comm->me,i,lb_cplx_master[i]);
    }
    
    // Only the partition that owns the complex should diagonalize the Hamiltonian
    if(universe->iworld == lb_cplx_master[i]) {
      // if(comm->me==0) {
      // 	fprintf(stdout,"(%i,%i)  Hamiltonian for complex %i\n",universe->iworld,comm->me,i);
      // 	for(int j=0; j<evb_complex->nstate; j++) fprintf(stdout,"(%i,%i)  j= %i  e_diagonal= %f\n",universe->iworld,comm->me,j,evb_matrix->e_diagonal[j][0]);
      // 	for(int j=0; j<evb_complex->nstate-1; j++) fprintf(stdout,"(%i,%i)  j= %i  e_offdiag= %f\n",universe->iworld,comm->me,j,evb_matrix->e_offdiag[j][0]);
      // 	for(int j=0; j<evb_complex->nextra_coupling; j++) fprintf(stdout,"(%i,%i)  j= %i  e_extra= %f\n",universe->iworld,comm->me,j,evb_matrix->e_extra[j][0]);
      
      // // 	error->universe_one(FLERR,"Early Termination");
      // }

      if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Starting diagonalization\n",universe->iworld,comm->me);
      full_matrix->diagonalize();
      if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished diagonalization\n",universe->iworld,comm->me);
      evb_complex->cec_v2->compute();
    }

    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Diagonalization complete for complex %i in compute.\n",universe->iworld,comm->me,i);
    // if(i==2) error->universe_one(FLERR,"Early Termination");
    task_pos+= evb_complex->nstate; // Update position in tasklist
  }

  if(debug_sci_mp) {
    fprintf(stdout,"(%i,%i)  Preparing to communicate eigenvectors.\n",universe->iworld,comm->me);
    MPI_Barrier(universe->uworld);
  }

  sci_comm_evec_mp();

  TIMER_CLICK(EVB_Engine, compute_sci_mp);
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::post_process_sci_mp(int vflag)
{   
  TIMER_STAMP(EVB_Engine, post_process_sci_mp);
  
  /****************************************************************/
  /***    Multi-complex subroutine   ******************************/
  /****************************************************************/
  
  delete_overlap_mp();

  // Setup load-balancer for sci_mp simulations
  if(mp_verlet_sci) setup_lb_mp();

  sci_initialize_mp(vflag);
  sci_iteration_mp(vflag);
  sci_finalize_mp(vflag);
  
  qsqsum_sys = qsqsum_env;
  for(int i=0; i<ncomplex; i++) qsqsum_sys += all_complex[i]->qsqsum;
  
  if(mp_verlet_sci->is_master) for(int i=0; i<ncomplex; i++) {
      all_complex[i]->cec->compute();
      all_complex[i]->cec_v2->compute();
    }
  
  if(universe->me==0) evb_output->execute();
  
  TIMER_CLICK(EVB_Engine, post_process_sci_mp);
}

/* ---------------------------------------------------------------------- */

void EVB_Engine::delete_overlap_mp()
{
  TIMER_STAMP(EVB_Engine, delete_overlap_mp);
  
  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    for(int iproc=0; iproc<universe->nprocs; iproc++) {
      if(iproc == universe->me) {
	if(iproc == 0) fprintf(stdout,"\n\nStates at start of delete_overlap()\n");
	fprintf(stdout,"(%i,%i)  nstates = ",universe->iworld,comm->me);
	for(int i=0; i<ncomplex; i++) fprintf(stdout," %i",all_complex[i]->nstate);
	fprintf(stdout,"\n");
      }
      MPI_Barrier(universe->uworld);
    }
  }

  int nlocal = atom->nlocal;
  int *is_kernel = evb_type->is_kernel;
  
  // Clear all the maps
  memset(complex_molecule,0, sizeof(int)*nmolecule);
  memset(max_coeff,0,sizeof(double)*nmolecule);
  memset(complex_atom,0,sizeof(int)*natom);
  memset(kernel_atom,0,sizeof(int)*natom);
  
  // Loop all complex to check overlap and build complex_index map
  for(int i=0; i<ncomplex; i++) {
    EVB_Complex* cplx = all_complex[i];
    
    // Loop all states of this complex
    for(int n=0; n<cplx->nstate; n++) {
      int mol = cplx->molecule_B[n];
      
      // Check whether this molecule is belonging to this complex
      if(complex_molecule[mol]!=i+1) {
	// Find the maximum coeff of this molecule in this complex
	double new_C = cplx->Cs2[n];
	for(int m = n+1; m<cplx->nstate; m++)
	  if(cplx->molecule_B[m]==mol && cplx->Cs2[m]>new_C)
	    new_C = cplx->Cs2[m];
	
	// If this molecule has not been used, label it.
	if(complex_molecule[mol]==0) {
	  complex_molecule[mol] = i+1;
	  max_coeff[mol] = new_C;
	} else { // Overlap occurs
	  EVB_Complex* target;
	  
	  // Identify which complex this molecule should be deleted from
	  if(new_C>max_coeff[mol]) target = all_complex[complex_molecule[mol]-1];
	  else target = cplx;
	  
	  // Delete states from target complex
	  for(int m=0; m<target->nstate; m++) if(target->molecule_B[m]==mol) {
	      target->delete_state(m--);
	    }
	  
	  // If delete it from original one, update its complex_index
	  if(target!=cplx) {
	    complex_molecule[mol] = i+1;
	    max_coeff[mol] = new_C;
	  } else n--;
	} // if(compelx_molecule[mol]==0)
      } // if(complex_molecule[mol]!=i+1)
    } // for(n<nstate)
  } // for(i<ncomplex)

  // Update the extra-coupling for each complex
  
  for(int i=0; i<ncomplex; i++) {
    EVB_Complex* cplx = all_complex[i];
    cplx->nextra_coupling = 0;  
    memset(cplx->extra_coupling,0,sizeof(int)*cplx->nstate);
    continue; // DEBUG_ON 
    if(bExtraCouplings) for(int j=0; j<cplx->nstate; j++) {
	cplx->extra_coupling[j] = 0;
	
	for(int k=j-1; (k>0 &&
			cplx->molecule_A[k] == cplx->molecule_A[j] &&  // These lines are rules for searching extra-couplings
			cplx->parent_id[k] == cplx->parent_id[j] &&    // after delete_overlap. They are more than the same
			cplx->reaction[k] == cplx->reaction[j] &&      // mission in the single-proton problems.
			cplx->path[k] == cplx->path[j] &&              // Don't modify these ruls if not think about very very
			cplx->shell[k] == cplx->shell[j] &&            // carefully. Or it may cause problems when you simulate
			cplx->molecule_B[k] != cplx->molecule_B[j]     // multi-proton problems with MS-EVB3. 
			); k--) {
	  cplx->nextra_coupling ++;
	  cplx->extra_coupling[j] ++;
	}
      }
  } // for(i<ncomplex)
  
  // Re-normalize eigen-vector
  for(int i=0; i<ncomplex; i++) {
    double *Cs = all_complex[i]->Cs;
    double *Cs2 = all_complex[i]->Cs2;
    
    double total = 0.0;
    for(int j=0; j<all_complex[i]->nstate; j++) total+=Cs2[j];

    double factor = sqrt(total);
    for(int j=0; j<all_complex[i]->nstate; j++) {
      Cs[j] /= factor;
      Cs2[j] /= total;
    }
  }
  
  // Build complex_index map for atoms
  
  for(int i=0; i<ncomplex; i++)
    all_complex[i]->nlocal_cplx=all_complex[i]->nghost_cplx=all_complex[i]->natom_cplx=0;
  
  int *molecule = atom->molecule;
  for(int i=0; i<natom; i++) {
    if(mol_type[i]==0) continue;
    int icplx = complex_molecule[molecule[i]];
    
    if(icplx>0) {
      complex_atom[i]=icplx;
      EVB_Complex *cplx = all_complex[icplx-1];
      complex_pos[i] = cplx->natom_cplx;
      cplx->cplx_list[cplx->natom_cplx++]=i;
      if(i<nlocal) cplx->nlocal_cplx++;
      else cplx->nghost_cplx++;
      
      int start = evb_type->type_index[mol_type[i]-1];
      if(is_kernel[start+mol_index[i]-1] == 1) kernel_atom[i] = icplx;
    }
  }
  
  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    for(int iproc=0; iproc<universe->nprocs; iproc++) {
      if(iproc == universe->me) {
	if(iproc == 0) fprintf(stdout,"\n\nStates at end of delete_overlap()\n");
	fprintf(stdout,"(%i,%i)  nstates = ",universe->iworld,comm->me);
	for(int i=0; i<ncomplex; i++) fprintf(stdout," %i",all_complex[i]->nstate);
	fprintf(stdout,"\n");
      }
      MPI_Barrier(universe->uworld);
    }
  }
  
  // Compute qsum and qsqsum
  for(int i=0; i<ncomplex; i++) all_complex[i]->compute_qsqsum();

  TIMER_CLICK(EVB_Engine, delete_overlap_mp);
}
/* ----------------------------------------------------------------------*/

void EVB_Engine::sci_initialize_mp(int vflag)
{
  TIMER_STAMP(EVB_Engine,sci_initialize_mp);

  engine_indicator = ENGINE_INDICATOR_INITIALIZE;

  evb_matrix = full_matrix;
  full_matrix->clear(true,vflag,true);
  if(natom>0) atom->f = full_matrix->f_env;
  
  /****************************************************************
   * Compute the energy and force of env <-> env
   *
   * Energy component:
   *     E(env) = E(env,pair) + E(env,bond) + E(env,kspace)
   *
   * Note:
   * (1) Energy are stored in [full_matrix->e_env]
   * (2) Force are stored in [full_matrix->f_env]
   * (3) K-Space:
   *     a) Ewald: [sracrl(cim)_env] are kept for folowing steps 
   ***************************************************************/
  
  evb_list->sci_split_env();

  // ********************************************************************************/
  // Environment computation/communication
  // ********************************************************************************/
  evb_list->change_list(ENV_LIST);
  if(evb_kspace) evb_kspace->sci_compute_env(vflag);

  compute_diagonal_mp(vflag);
  if(universe->iworld == 0 && EFIELD_flag) compute_efield(0,0); // Only master computes env efield

  double env_evdwl = 0.0;
  double env_ecoul = 0.0;
  MPI_Reduce(&(force->pair->eng_vdwl), &env_evdwl, 1, MPI_DOUBLE, MPI_SUM, 0, mp_verlet_sci->block);
  MPI_Reduce(&(force->pair->eng_coul), &env_ecoul, 1, MPI_DOUBLE, MPI_SUM, 0, mp_verlet_sci->block);
  force->pair->eng_vdwl = env_evdwl;
  force->pair->eng_coul = env_ecoul;
  
  // Is this necessary to do here, or can it wait till sci_finalize_mp()?
  Force_Reduce();
  memset(&(lmp_f[0][0]), 0, sizeof(double)*natom*3);
  MPI_Reduce(&(full_matrix->f_env[0][0]), &(lmp_f[0][0]), natom*3, MPI_DOUBLE, MPI_SUM, 0, mp_verlet_sci->block);
  memcpy(&(full_matrix->f_env[0][0]), &(lmp_f[0][0]), natom*3*sizeof(double));
  Force_Clear(2);
  
  evb_repulsive = NULL;
  if(universe->iworld==0) {
    full_matrix->save_ev_diag(MATRIX_ENV,vflag);
    MPI_Allreduce(full_matrix->e_env,full_matrix->sci_e_env,EDIAG_NITEM,MPI_DOUBLE,MPI_SUM,world);
  }

  // ********************************************************************************/
  // Only master of complex build states
  // ********************************************************************************/
  for(int i=0; i<ncomplex; i++) {
    evb_complex = all_complex[i];
    evb_complex->cec->clear();

    // Should not be necessary to allocate force arrays for all complexes on all partitions
    // if(mp_verlet_sci->is_master) {
    //   all_matrix[i]->setup_mp();
    //   all_matrix[i]->clear_mp(true,vflag,true);
    // }
    all_matrix[i]->setup();
    all_matrix[i]->clear(true,vflag,true);

    evb_complex->save_avec(0);

    // if(lb_cplx_master[i] != universe->iworld) continue;
    if(!lb_cplx_owned[i]) continue;
  
    int iextra_coupling = 0;
  
    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Partition %i is working on complex %i in initialize: lb_cplx_owned= %i.  nstate= %i\n",
    			     universe->iworld,comm->me,universe->iworld,i,lb_cplx_owned[i],evb_complex->nstate);
    
    evb_complex->cec->compute_coc();
    for(int j=1; j<evb_complex->nstate; j++) {
      if(debug_sci_mp) fprintf(stdout,"(%i,%i)  building state %i\n",universe->iworld,comm->me,j);
      evb_complex->sci_build_state(j);
      evb_offdiag = all_offdiag[evb_complex->reaction[j]-1];
      if(evb_complex->extra_coupling[j]>0) {
  	int save_mol_A = evb_complex->molecule_A[j];
  	
  	for(int k=0; k<evb_complex->extra_coupling[j]; k++) {
  	  evb_complex->extra_i[iextra_coupling] = j-k-1;
  	  evb_complex->extra_j[iextra_coupling] = j;
  	  
  	  evb_complex->molecule_A[j] = evb_complex->molecule_B[j-k-1];
  	  
  	  SETUP_OFFDIAG_EXCH(extra,iextra_coupling);
  	  evb_offdiag->sci_setup_mp();
  	  
  	  iextra_coupling++;
  	}
  	
  	evb_complex->molecule_A[j] = save_mol_A;
      }
      SETUP_OFFDIAG_EXCH(off,j-1);
      evb_offdiag->sci_setup_mp();
   
      evb_complex->cec->compute_coc();
      evb_complex->save_avec(j);
    } // loop over states
    
    evb_complex->load_avec(0);
    evb_complex->update_mol_map();

  } // loop over complexes

  // Sync avec for all complexes on all partitions
  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    if(universe->me==0) fprintf(stdout,"(%i,%i)  Entering sci_comm_avec()\n",universe->iworld,comm->me);
  }

  sci_comm_avec();
  
  /****************************************************************
   * Compute the energy and force of cplx(i) <- cplx(i) + env
   *
   * Energy component:
   *     E[cplx(i)] = E[cplx(i)<-cplx(i)+env,pair]
   *                  + E[cplx(i)<-cplx(i)+env,bond]
   *                  + E[cplx(i)<-cplx(i),kspace]
   *
   * Note:
   * (1) Interaction include cplx(i) with itself and env
   * (2) cplx(i)<->cplx(j,j!=i) is not included
   * (3) Only force on cplx atoms is stored
   * (4) Energy and Force are stored in [all_matrix(i)]
   *      a) all_matrix[i]->e_diaongal
   *                      ->e_off_diagonal
   *                      ->e_extra_coupling
   *                      ->f_diagonal
   *                      ->f_off_diagonal
   *                      ->f_extra_coupling
   ****************************************************************/

  // ********************************************************************************/
  // Complex computation phase
  // ********************************************************************************/

  evb_list->change_list(SYS_LIST);

  int task_pos = 0; // Current position in lb_tasklist
  for(int i=0; i<ncomplex; i++) {
    // ********************************************************************************/
    // Skip energy/force evaluation if complex not owned by partition, but still 
    //  need to rebuild all states because of delete_overlap().

    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Partition %i is working on complex %i in initialize: %d.  nstate= %i\n",
    			     universe->iworld,comm->me,universe->iworld,i,lb_cplx_owned[i],all_complex[i]->nstate);

    // Skip rest of work for this complex if not owned by partition
    if(!lb_cplx_owned[i]) {
      task_pos+= all_complex[i]->nstate; // Update position in tasklist
      // if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Skipping computation of complex in initialize. New task_pos is %i.\n",
      // 			       universe->iworld,comm->me,task_pos);
      continue;
    }
    // ********************************************************************************/

    evb_complex = all_complex[i];

    all_matrix[i]->setup(); // Should not be necessary to allocate force arrays for all complexes on all partitions
    all_matrix[i]->clear(true,vflag,true);

    int iextra_coupling = 0;
    
    full_matrix->clear(true,vflag,true,false); // Is this the cleanest way to keep full_matrix->f_env intact?    
    if(evb_kspace) evb_kspace->sci_setup_init();

    // Build pair and bond list for complex
    evb_list->multi_split();
    evb_list->change_list(EVB_LIST);
    evb_complex->update_list();
    
    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Partition %i is starting loop over states for complex %i in initialize: %d.\n",
			     universe->iworld,comm->me,universe->iworld,i,lb_cplx_owned[i]);

    for(int j=0;j<evb_complex->nstate; j++) {

      if(debug_sci_mp) fprintf(stdout,"(%i,%i)  i= %i  j= %i  task_pos= %i  lb_task_list= %i in initialize\n",
			       universe->iworld,comm->me,i,j,task_pos,lb_tasklist[task_pos]);
      
      // Check tasklist if assigned to work on current state
      if(lb_tasklist[task_pos] != universe->iworld) {
	task_pos++;
	continue;
      }

      // Load current state
      evb_complex->load_avec(j);
      evb_complex->update_mol_map();
      evb_complex->update_list();
      
      // Setup diagonal and calculate
      if(natom>0) atom->f = full_matrix->f_diagonal[j];

      TIMER_STAMP(EVB_Engine, compute__Eii_compute_diagonal);
      compute_diagonal(vflag);
      if(EFIELD_flag) compute_efield(1,i+1);
      TIMER_CLICK(EVB_Engine, compute__Eii_compute_diagonal);

      TIMER_STAMP(EVB_Engine, compute__Eii_evb_kspace_compute_cplx);
      if(evb_kspace) evb_kspace->compute_cplx(vflag); 
      TIMER_CLICK(EVB_Engine, compute__Eii_evb_kspace_compute_cplx);
	
      compute_repulsive(vflag);

      full_matrix->save_ev_diag(j,vflag);
	
      Force_Reduce();
	
      // Update position in tasklist for load-balancer
      if(j==0) {
	task_pos++;
	continue;
      }

      // Setup extra-couplings and calculate
      if(evb_complex->extra_coupling[j]>0) {
	int save_mol_A = evb_complex->molecule_A[j];
	
	for(int k=0; k<evb_complex->extra_coupling[j]; k++) {
	  evb_complex->extra_i[iextra_coupling] = j-k-1;
	  evb_complex->extra_j[iextra_coupling] = j;
	  
	  evb_complex->molecule_A[j] = evb_complex->molecule_B[j-k-1];
	  if(natom>0) atom->f = full_matrix->f_extra_coupling[iextra_coupling];
	  
	  SETUP_OFFDIAG_EXCH(extra,iextra_coupling);
	  evb_offdiag->sci_setup(vflag);
	  full_matrix->save_ev_offdiag(true,iextra_coupling,vflag);
	  
	  Force_Reduce();
	  
	  iextra_coupling++;
	}
	
	evb_complex->molecule_A[j] = save_mol_A;
      }
      
      // Off-diagonal-element

      evb_offdiag = all_offdiag[evb_complex->reaction[j]-1];
      if(natom>0) atom->f = full_matrix->f_off_diagonal[j-1];
      SETUP_OFFDIAG_EXCH(off,j-1);
      evb_offdiag->sci_setup(vflag);
      full_matrix->save_ev_offdiag(false,j-1,vflag);
      
      Force_Reduce();
      if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished with state %i in initialize.\n",universe->iworld,comm->me,j);
      // Update position in tasklist for load-balancer
      task_pos++;
    } // for(j<nstate)

    // Save background force and energy;
    full_matrix->total_energy();

    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Preparing to diagonalize complex %i in initialize.  lb_cplx_split= %i\n",
			     universe->iworld,comm->me,i,lb_cplx_split[i]);

    // // If complex was split across partitions, then slave partitions send their info to 
    // //   master rank of partition that owns complex. This master rank will calculate 
    // //   ground state eigenvector for call to sci_comm_evec_mp().
    // // Partitions wait here till all working on current complex are ready
    // if(lb_cplx_split[i]) sci_comm_lb_mp_1();
    
    all_matrix[i]->clear(true,vflag,true);
    all_matrix[i]->copy_ev(vflag);
    all_matrix[i]->copy_force();

    // Change back to pivot state
    evb_complex->load_avec(0);
    evb_complex->update_mol_map();
    evb_complex->update_list();
    
    // Calculate the effective LJ parameters and charges
    if(flag_DIAG_QEFF) evb_effpair->compute_para_qeff();
    else evb_effpair->compute_para();
    
    // Init the iteration
    all_matrix[i]->E = 0.0;
    all_matrix[i]->dE = 0.0;
    all_matrix[i]->ground_state_energy = 0.0;
    
    // list
    evb_list->change_list(SYS_LIST);

    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished with complex %i in initialize.\n",universe->iworld,comm->me,i);
  } // for(i<ncomplex)

  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished with all complexes in initialize.\n",universe->iworld,comm->me);

  // ********************************************************************************/
  // Communication phase
  // ********************************************************************************/

  task_pos = 0;
  for(int i=0; i<ncomplex; i++) {
    evb_complex = all_complex[i];
    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Starting compute_para() for complex %i in initialize.\n",universe->iworld,comm->me,i);
    //evb_complex->load_avec(0);
    if(flag_DIAG_QEFF) evb_effpair->compute_para_qeff();
    else evb_effpair->compute_para();

    // Skip rest of work for this complex if not owned by partition
    if(!lb_cplx_owned[i]) {
      task_pos+= evb_complex->nstate; // Update position in tasklist
      if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Skipping communication of complex in initialize. New task_pos is %i.\n",
			       universe->iworld,comm->me,task_pos);
      continue;
    }

    // If complex was split across partitions, then slave partitions send their info to 
    //   master rank of partition that owns complex. This master rank will calculate 
    //   ground state eigenvector for call to sci_comm_evec_mp().
    // Partitions wait here till all working on current complex are ready
    
    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Starting comm for complex %i in initialize.\n",universe->iworld,comm->me,i);
    if(lb_cplx_split[i]) sci_comm_lb_mp_2();
    
    task_pos+= evb_complex->nstate; // Update position in tasklist
  }
  //error->universe_all(FLERR,"Early Termination");

  TIMER_CLICK(EVB_Engine,sci_initialize_mp);
}

/* ----------------------------------------------------------------------*/
// This procedure is used to calculated SCI energy and force, that is, the
// interaction between each complex pair. In each iteration loop, all complex
// will be looped, and when calculating a specific complex (id), it means
// calculating interaction between atoms of complex==id and (complex!=0 &&
// complex!=id). All procedures named ***_sci mean it.

void EVB_Engine::sci_iteration_mp(int vflag)
{
  TIMER_STAMP(EVB_Engine,sci_iteration_mp);

  engine_indicator = ENGINE_INDICATOR_ITERATION;

  if(debug_sci_mp && universe->me==0) fprintf(stdout,"\n\n(%i,%i)  Inside sci_iteration_mp()\n",universe->me,comm->me);
  
  evb_list->sci_split_inter();
  evb_list->change_list(CPL_LIST);
  evb_effpair->pre_compute();
  
  ncircle = 0;
  int num_converged;
  
  double **lmp_f = atom->f;
  atom->f = full_matrix->f_env;
  
  int is_Vij_ex;
  
  while(true) {
    
    // ********************************************************************************/
    // Computation phase
    // ********************************************************************************/
    
    num_converged = 0;
    int task_pos = 0; // Current position in lb_tasklist
    for(int i=0; i<ncomplex; i++) {
      
      evb_complex = all_complex[i];

      // Skip rest of work for this complex if not owned by partition
      if(!lb_cplx_owned[i]) {
	task_pos+= evb_complex->nstate; // Update position in tasklist
	if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Skipping computation of complex in iteration. New task_pos is %i.\n",
				 universe->iworld,comm->me,task_pos);
	continue;
      }

      // Zero sci_e arrays. Needed when multiple partitions working on same complex. 
      for(int j=0; j<evb_complex->nstate; j++) for(int k=0; k<SCI_EDIAG_NITEM; k++) all_matrix[i]->sci_e_diagonal[j][k] = 0.0;
      for(int j=0; j<evb_complex->nstate-1; j++) all_matrix[i]->sci_e_offdiag[j] = 0.0;
      for(int j=0; j<evb_complex->nextra_coupling; j++) all_matrix[i]->sci_e_extra[j] = 0.0;

      if(evb_kspace) evb_kspace->sci_setup_iteration();

      // Extra-couplings; The partition that owns the complex is currently responsible for these.
      // Should these be added as entries to the tasklist?
      // How many extra couplings are there in a typical SCI simulation?
      if(lb_cplx_master[i] == universe->iworld) {
	for(int j=0; j<evb_complex->nextra_coupling; j++) {
	  error->universe_one(FLERR,"Need to double check if current position of extra-couplings is OK");


	  is_Vij_ex = all_offdiag[evb_complex->reaction[j]-1]->is_Vij_ex;

	  evb_effpair->ecoul = 0.0;
	  if(evb_kspace) evb_kspace->off_diag_energy = 0.0;

	  if(is_Vij_ex > 0) {
	    evb_effpair->init_exch(true,j);
	    evb_effpair->compute_exch(evb_complex->id);
	    if(flag_EFFPAIR_SUPP) evb_effpair->compute_exch_supp(evb_complex->id);
	    if(evb_kspace && is_Vij_ex == 1) evb_kspace->sci_compute_exch(vflag);
	  }
	  
	  all_matrix[i]->sci_save_ev_offdiag(true,j,vflag);
	}
      }
      
      // Diagonal elements
      for(int j=0; j<evb_complex->nstate; j++) {

	// Check tasklist if assigned to work on current state
	if(lb_tasklist[task_pos] != universe->iworld) {
	  task_pos++;
	  continue;
	}
	
	// if(debug_sci_mp) fprintf(stdout,"(%i,%i)  i= %i  j= %i  task_pos= %i  lb_task_list= %i\n",
	// 			 universe->iworld,comm->me,i,j,task_pos,lb_tasklist[task_pos]);
	
	evb_complex->load_avec(j);
	evb_effpair->compute_cplx(evb_complex->id);
	if(flag_EFFPAIR_SUPP) evb_effpair->compute_cplx_supp(evb_complex->id);
	if(evb_kspace) evb_kspace->sci_compute_cplx(vflag);
	all_matrix[i]->sci_save_ev_diag(j,vflag);
	if(debug_sci_mp && i==0) fprintf(stdout,"(%i,%i)  i= %i  j= %i  evb_kspace->energy= %f  evb_effpair->energy= %f\n",
					 universe->iworld,comm->me,i,j,evb_kspace->energy,evb_effpair->energy);
	
	// Off-diagonal elements
	if(j>0) {
	  is_Vij_ex = all_offdiag[evb_complex->reaction[j]-1]->is_Vij_ex;

	  evb_effpair->ecoul = 0.0;
	  if(evb_kspace) evb_kspace->off_diag_energy = 0.0;

	  if(is_Vij_ex > 0) {
	    evb_effpair->init_exch(false,j-1);
	    evb_effpair->compute_exch(evb_complex->id);
	    if(flag_EFFPAIR_SUPP) evb_effpair->compute_exch_supp(evb_complex->id);
	    if(evb_kspace && is_Vij_ex == 1) evb_kspace->sci_compute_exch(vflag);
	  }

	  all_matrix[i]->sci_save_ev_offdiag(false,j-1,vflag);
	}
	
	// Update position in tasklist for load-balancer
	task_pos++;
      }
      
      // total energy to full_matrix to get diagonalize
      
      all_matrix[i]->sci_total_energy();
	
      evb_complex->load_avec(0);
    } // for(i<ncomplex)


    // ********************************************************************************/
    // Communication phase
    // ********************************************************************************/

    task_pos = 0;
    for(int i=0; i<ncomplex; i++) {
      evb_complex = all_complex[i];

      // Skip rest of work for this complex if not owned by partition
      if(!lb_cplx_owned[i]) {
	task_pos+= evb_complex->nstate; // Update position in tasklist
	// if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Skipping communication of complex in iteration. New task_pos is %i.\n",
	// 			 universe->iworld,comm->me,task_pos);
	continue;
      }


      // If complex was split across partitions, then slave partitions send their info to 
      //   master rank of partition that owns complex. The master rank will calculate 
      //   ground state eigenvector for call to sci_comm_evec_mp().
      if(lb_cplx_split[i]) sci_comm_lb_mp_3();

      for(int j=0; j<evb_complex->nstate; j++) {  
	full_matrix->e_diagonal[j][EDIAG_POT] = all_matrix[i]->e_diagonal[j][EDIAG_POT]
	  +all_matrix[i]->sci_e_diagonal[j][SCI_EDIAG_POT];
	if(debug_sci_mp && i==0) fprintf(stdout,"(%i,%i)  i= %i  j= %i  all->e_diagonal= %f  all->sci_e_diagonal= %f\n",
					 universe->iworld,comm->me,i,j,all_matrix[i]->e_diagonal[j][EDIAG_POT],all_matrix[i]->sci_e_diagonal[j][SCI_EDIAG_POT]);

      }
      
      for(int j=0; j<evb_complex->nstate-1; j++) { 
	full_matrix->e_offdiag[j][EOFF_ENE] =
	  all_matrix[i]->e_offdiag[j][EOFF_ENE]+all_matrix[i]->sci_e_offdiag[j];
      }
      
      for(int j=0; j<evb_complex->nextra_coupling; j++) {
	full_matrix->e_extra[j][EOFF_ENE] = 
	  all_matrix[i]->e_extra[j][EOFF_ENE]+all_matrix[i]->sci_e_extra[j];
      }
      
      // Only the master partition for this complex should diagonalize the Hamiltonian
      if(universe->iworld == lb_cplx_master[i]) {
	full_matrix->diagonalize();
    
	all_matrix[i]->ground_state = full_matrix->ground_state;
	all_matrix[i]->pivot_state = full_matrix->pivot_state;
	all_matrix[i]->E = full_matrix->ground_state_energy;
	all_matrix[i]->dE = all_matrix[i]->E - all_matrix[i]->ground_state_energy;
	all_matrix[i]->ground_state_energy = all_matrix[i]->E;

	if(debug_sci_mp) fprintf(stdout,"(%i,%i)  ncircle= %i  i= %i  E= %f  dE= %f\n",universe->iworld,comm->me,ncircle,i,all_matrix[i]->E,all_matrix[i]->dE);

	if(sci_overlap_tol > 0.0) {
	  double overlap = 0.0;
	  if(ncircle > 0) for(int j=0; j<evb_complex->nstate; j++) overlap+= evb_complex->Cs_prev[j] * evb_complex->Cs[j];
	  if(overlap > sci_overlap_tol) num_converged++;
	} else if(fabs(all_matrix[i]->dE) < CNVG_LIMIT) num_converged++;
      }
      
      task_pos += evb_complex->nstate;
    }

    /*************************************************
      Sync up eigenvectors on all partitions
    *************************************************/

    sci_comm_evec_mp();

    /*************************************************
      Update charges and effective vdw parameters
      Currently, all partitions do this for all complexes instead of communicating updated parameters
      This is why states were built for all complexes at beginning of initialize_sci_mp().
     *************************************************/
    for(int i=0; i<ncomplex; i++) {
      evb_complex = all_complex[i];
      if(flag_DIAG_QEFF) evb_effpair->compute_para_qeff();
      else evb_effpair->compute_para();

      // Save eigenvector for next simulation step
      memcpy(evb_complex->Cs_prev, evb_complex->Cs, sizeof(double)*evb_complex->nstate);
    }

    /************************************************
      Test for convergence
     ************************************************/

    ncircle++;
    if(ncircle==MAX_CIRCLE) break;
    
    // The following is safe only because partitions that own a complex update the value of dE
    //  for the complexes that they own.
    int nC = 0;
    if(comm->me==0) MPI_Allreduce(&num_converged, &nC, 1, MPI_INT, MPI_SUM, mp_verlet_sci->block);
    MPI_Bcast(&nC, 1, MPI_INT, 0, world);
    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  ncircle= %i  num_converged= %i  nC= %i\n",
			     universe->iworld,comm->me,ncircle,num_converged,nC);
    if(nC==ncomplex && ncircle>0) break;
  }
  
  TIMER_CLICK(EVB_Engine,sci_iteration_mp);
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::sci_finalize_mp(int vflag)
{
  TIMER_STAMP(EVB_Engine,sci_finalize_mp);

  engine_indicator = ENGINE_INDICATOR_FINALIZE;

  if(debug_sci_mp && universe->me==0) fprintf(stdout,"(%i,%i)  Inside sci_finalize_mp()\n",universe->iworld,comm->me);
  
  atom->f = full_matrix->f_env;

  // Zero environment force on slave partitions
  if(!mp_verlet_sci->is_master) memset(&(atom->f[0][0]), 0, sizeof(double)*3*natom);
  
  /************************************************************/
  /******** Energy from each complex **************************/
  /************************************************************/
  
  cplx_energy = 0.0;
  inter_energy = 0.0;
  env_energy = full_matrix->sci_e_env[EDIAG_POT];

  int istart = universe->iworld;
  int di = universe->nworlds;
  
  // Interaction energy between each pair of complexes
  // Only the master of a complex evaluate this contribution
  for(int i=0; i<ncomplex; i++) {
    if(lb_cplx_master[i] != universe->iworld) continue;

    EVB_Complex* cplx = all_complex[i];
    EVB_MatrixSCI* mtx = all_matrix[i];

    cplx_energy += mtx->E;
    
    double *Cs = cplx->Cs;
    double *Cs2 = cplx->Cs2;
    int nstate = cplx->nstate;
    int nextra = cplx->nextra_coupling;
    int *parent = cplx->parent_id;
    int *ex_i = cplx->extra_i;
    int *ex_j = cplx->extra_j;
    
    for(int j=0; j<nstate; j++) inter_energy += Cs2[j]*mtx->sci_e_diagonal[j][SCI_EDIAG_POT];
    for(int j=1; j<nstate; j++) inter_energy += 2.0 * Cs[j]*Cs[parent[j]]*mtx->sci_e_offdiag[j-1];
    for(int j=0; j<nextra; j++) inter_energy += 2.0 * Cs[ex_i[j]]*Cs[ex_j[j]]*mtx->sci_e_extra[j];
  }
  
  inter_energy /= 2.0;
  
  double etmp = cplx_energy;
  MPI_Allreduce(&etmp, &cplx_energy, 1, MPI_DOUBLE, MPI_SUM, mp_verlet_sci->block);
  etmp = inter_energy;
  MPI_Allreduce(&etmp, &inter_energy, 1, MPI_DOUBLE, MPI_SUM, mp_verlet_sci->block);

  // The env_energy is only correct on master partition
  energy  = cplx_energy - inter_energy + env_energy;

  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  cplx_energy= %f  inter_energy= %f  env_energy= %f  energy= %f\n",universe->iworld,comm->me,cplx_energy,inter_energy,env_energy,energy);

  /************************************************************/
  // Hellmann-Feynman for each complex
  // Only partitions that own part of a complex should 
  //  calculate HF forces for that complex.
  /************************************************************/
  for (int i=0; i<ncomplex; i++) if(lb_cplx_owned[i]) all_matrix[i]->compute_hellmann_feynman();
  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished with Hellman-Feynman\n",universe->iworld,comm->me);

  /************************************************************/
  // Geometry force for off-diagonals
  // Only masters of a complex evaluate this contribution
  /************************************************************/

  for (int i=0; i<ncomplex; i++) {
    if(lb_cplx_master[i] != universe->iworld) continue; // Skip if partition is not master of complex

    evb_complex = all_complex[i];
    int iextra = 0;

    // Only master rank currently has correct off-diagonal energy for complex
    // Pack Vij's and send to rank that owns complex.
    int size = evb_complex->nstate - 1 + evb_complex->nextra_coupling;

    if(size > max_list_Vij) {
      max_list_Vij = size;
      memory->grow(list_Vij, size, "EVB_Engine:list_Vij");
    }
    memset(&(list_Vij[0]), -1, sizeof(double)*size);

    int ilist = 0;
    double arq;
    for(int j=1; j<evb_complex->nstate; j++) {
  
      // Extra-coupling-elements
      if(evb_complex->extra_coupling[j]>0) for(int k=0; k<evb_complex->extra_coupling[j]; k++) {
	  list_Vij[ilist++] = all_matrix[i]->e_extra[iextra][EOFF_VIJ];
	  arq = all_matrix[i]->e_extra[iextra][EOFF_ARQ];
	  if(arq>0.0) list_Vij[ilist-1]+= all_matrix[i]->sci_e_extra[iextra]/arq; // prevent division by zero
	  iextra++;
	}
	
      // Off-diagonal-element  
      list_Vij[ilist++] = all_matrix[i]->e_offdiag[j-1][EOFF_VIJ];
      arq = all_matrix[i]->e_offdiag[j-1][EOFF_ARQ];
      if(arq>0.0) list_Vij[ilist-1] += all_matrix[i]->sci_e_offdiag[j-1]/arq; // prevent division by zero
    }
    
    // If master rank doesn't already own complex, then send Vij to rank that does.
    if(rc_rank[i] != 0) {
      MPI_Status status;
      if(comm->me == 0) MPI_Send(&(list_Vij[0]), size, MPI_DOUBLE, rc_rank[i], 0, world);
      if(comm->me == rc_rank[i]) MPI_Recv(&(list_Vij[0]), size, MPI_DOUBLE, 0, 0, world, &status);
    }

    iextra = 0;
    ilist = 0;
    for(int j=0;j<evb_complex->nstate; j++) {
      evb_complex->load_avec(j);
      evb_complex->update_mol_map();
  
      // V_rep
  
      evb_repulsive = NULL;
      int mol_id = evb_complex->molecule_B[j];
      int target_atom = molecule_map[mol_id][1];
  
      if(target_atom==-1 || target_atom>=atom->nlocal) {}
      else {
      
      	int target_etp = mol_type[target_atom];
      	for(int irep=0; irep<nrepulsive; irep++)
      	  if(all_repulsive[irep]->etp_center == target_etp) {
      	    evb_repulsive = all_repulsive[irep];
      	    break;
      	  }
      
      	if (evb_repulsive) {
      	  evb_repulsive->center_mol_id = mol_id;
      	  evb_repulsive->sci_compute(vflag);
      	}
      }
  
      // V_ij
  
      if (j==0) continue;
      evb_offdiag = all_offdiag[evb_complex->reaction[j]-1];
  
      // Extra-coupling-elements
      if(evb_complex->extra_coupling[j]>0) {
  	int save_mol_A = evb_complex->molecule_A[j];
  	
  	for(int k=0; k<evb_complex->extra_coupling[j]; k++) {
  	  evb_complex->molecule_A[j] = evb_complex->molecule_B[j-k-1];
  	  evb_offdiag->Vij = list_Vij[ilist++];  	  
  	  evb_offdiag->sci_compute(vflag);
  	  iextra++;
  	}
  	
  	evb_complex->molecule_A[j] = save_mol_A;
      }
  
      // Off-diagonal-element
      evb_offdiag->Vij = list_Vij[ilist++];
      evb_offdiag->sci_compute(vflag);

    } // for(j<nstate)
  
    evb_complex->load_avec(0);
    evb_complex->update_mol_map();
  } // for(i<ncomplex)
  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished with off-diagonals\n",universe->iworld,comm->me);
  
  /************************************************************/
  // Effective short-range force between complexes
  /************************************************************/

  evb_list->change_list(CPL_LIST);
  evb_effpair->compute_finter_mp(vflag);
  if(flag_EFFPAIR_SUPP) evb_effpair->compute_finter_supp_mp(vflag);
  if(debug_sci_mp && mp_verlet_sci->is_master2) fprintf(stdout,"(%i,%i)  Finished with compute_finter\n",universe->iworld,comm->me);
  
  evb_list->change_list(SYS_LIST);
  
  /************************************************************/
  // Effective short-range force with environment
  /************************************************************/
  evb_effpair->compute_fenv_mp(vflag);
  if(flag_EFFPAIR_SUPP) evb_effpair->compute_fenv_supp_mp(vflag);
  if(debug_sci_mp && mp_verlet_sci->is_master2) fprintf(stdout,"(%i,%i)  Finished with compute_fenv\n",universe->iworld,comm->me);
  
  /************************************************************/
  // Effective kspace force
  /************************************************************/
  
  double *q_save = atom->q;
  atom->q = evb_effpair->q;
  if(evb_kspace) {
    if(SCI_KSPACE_flag == KSPACE_DEFAULT)  evb_kspace->sci_compute_eff_mp(vflag);        // HF forces for both Ewald
    else if(SCI_KSPACE_flag == PPPM_HF_FORCES) {
      evb_kspace->sci_compute_eff_mp(vflag);                                             // HF forces for PPPM (ENV due to CPLX)
      evb_kspace->sci_compute_eff_cplx_mp(vflag);                                        // HF forces for PPPM (CPLX due to CPLX)
    } else if(SCI_KSPACE_flag == PPPM_ACC_FORCES) evb_kspace->sci_compute_eff_mp(vflag); // ACC forces for PPPM
    else if(SCI_KSPACE_flag == PPPM_POLAR_FORCES) sci_pppm_polar();                      // ACC + Polar forces for PPPM
  }
  Force_Reduce();
  atom->q = q_save;
  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished with sci_compute_eff_mp\n",universe->iworld,comm->me);
  
  /************************************************************/
  /****************** End of Force ****************************/
  /************************************************************/

  // Bcast avecs for complexes where reaction occurred to all partitions
  sci_comm_avec_pivot();

  nreact = 0;
  for(int i=0; i<ncomplex; i++) {
    evb_complex = all_complex[i];
    int pivot = all_matrix[i]->pivot_state;
    // evb_complex->load_avec(pivot);
    // if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Pivot loaded= %i for complex %i\n",universe->iworld,comm->me,pivot,i);
    
    // Full calculation of effective charges for output.
    if(mp_verlet_sci->is_master && flag_DIAG_QEFF) evb_effpair->compute_q_eff(true,true);
    if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Finished with compute_q_eff: i= %i\n",universe->iworld,comm->me,i);

    evb_complex->update_mol_map();
    if(pivot!=0) {
      nreact++;

      // Ensure order of complexes doesn't change on next step
      rc_molecule_prev[i] = evb_complex->molecule_B[pivot];
    }
  }

  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    for(int i=0; i<universe->nprocs; i++) {
      if(universe->me == i) {
	fprintf(stdout,"(%i,%i)  Force completed on rank %i\n",universe->iworld,comm->me,i);
	for(int j=0; j<ncomplex; j++) fprintf(stdout,"(%i,%i)  j= %i  pivot_state= %i\n",universe->iworld,comm->me,j,all_matrix[j]->pivot_state);
      }
      MPI_Barrier(universe->uworld);
    }
  }

  // Send energies from slave partitions to rank 0 for output
  // This should only be necessary on steps where output is written
  // This following if() logic was copied from EVB_Output::execute().
  int flag = true;
  if(universe->me == 0) {
    int freq = evb_output->freq;
    int ifreq = evb_output->ifreq;
    if(freq && (++ifreq==freq)) flag=false;
    else if(freq==0 && update->ntimestep == output->next) flag=false;
    else if(evb_output->react && nreact) flag=false;
  }
  MPI_Bcast(&flag, 1, MPI_INT, 0, mp_verlet_sci->block);

  if(!flag) for(int i=0; i<ncomplex; i++) all_matrix[i]->sci_comm_energy_mp(i);
  
  // Reset force pointer and accumulate forces on master partition.
  atom->f = lmp_f;
  memset(&(atom->f[0][0]), 0, sizeof(double)*3*natom);
  MPI_Reduce(&(full_matrix->f_env[0][0]), &(atom->f[0][0]), natom*3, MPI_DOUBLE, MPI_SUM, 0, mp_verlet_sci->block);

  if(debug_sci_mp && universe->me==0) fprintf(stdout,"(%i,%i)  Leaving sci_finalize_mp()\n",universe->iworld,comm->me);
  TIMER_CLICK(EVB_Engine,sci_finalize_mp);
}

/* ----------------------------------------------------------------------*/
// The large cost of sci_comm_evec_mp() is currently due to load-imbalance
//  between partitions with varying number of states. FIX ME!!!
// If only the master partition becomes responsible for calculating environment
//  contribution, then the master partition could be assigned fewer states than
//  the rest to further balance the partition workload.
/* ----------------------------------------------------------------------*/

void EVB_Engine::sci_comm_evec_mp()
{
  TIMER_STAMP(EVB_Engine, sci_comm_evec_mp);

  // MPI_Barrier(universe->uworld);
  // for(int ipart=0; ipart<universe->nworlds; ipart++) {
  //   if((universe->iworld == ipart) && (comm->me == 0)) {
  //     fprintf(stdout,"\n\nOriginal eigenvectors on partition %i\n",ipart);
  //     for(int i=0; i<ncomplex; i++) {
  // 	fprintf(stdout,"i = %i : nstate = %i  C = ",i,all_complex[i]->nstate);
  // 	int n = all_complex[i]->nstate;
  // 	if(n>10) n = 10;
  // 	for(int j=0; j<n; j++) fprintf(stdout," %f",all_complex[i]->Cs2[j]);
  // 	fprintf(stdout,"\n");
  //     }
  //   }
  //   MPI_Barrier(universe->uworld);
  // }

  int is_master = mp_verlet_sci->is_master;
  int size = 0;
  for(int i=0; i<ncomplex; i++) size+= all_complex[i]->nstate;

  // Pack eigenvectors that each partition owns into array
  int n = 0;

  if(size*2 > max_cc2) {
    max_cc2 = size * 2;
    memory->grow(cc2, max_cc2, "EVB_Engine:cc2");
  }
  memset(&(cc2[0]), 0, sizeof(double)*size*2);

  if(comm->me == 0) {
    // Only partitions that own a complex pack the eigenvectors
    for(int i=0; i<ncomplex; i++) {
      int part_id = lb_cplx_master[i];
      int nstate = all_complex[i]->nstate;
      if(part_id == universe->iworld) for(int j=0; j<nstate; j++) {
	  cc2[n]   = all_complex[i]->Cs[j];
	  cc2[size + n++] = all_complex[i]->Cs2[j];
	} else n += nstate;
    }
    
    // Accumulate eigenvectors on master ranks for all partitions
    MPI_Allreduce(MPI_IN_PLACE, &(cc2[0]),  size*2, MPI_DOUBLE, MPI_SUM, mp_verlet_sci->block);
  }
  
  // Broadcast eigenvectors from each master rank to rest of respective partition
  MPI_Bcast(&(cc2[0]),  size*2, MPI_DOUBLE, 0, world);

  // Unpack eigenvectors, calculate square of eigenvector, and identify parent state
  n = 0;
  for(int i=0; i<ncomplex; i++) {
    double *Cs = all_complex[i]->Cs;
    double *Cs2 = all_complex[i]->Cs2;
    int nstate = all_complex[i]->nstate;
    for(int j=0; j<nstate; j++) {
      Cs[j]  = cc2[n];
      Cs2[j] = cc2[size + n++];
    }

    int part_id = lb_cplx_master[i];

    // Find state with maximal amplitude
    int pivot_state = 0;
    double max_c = Cs2[0];
    for(int j=1; j<nstate; j++) 
      if(Cs2[j] > max_c) {
	max_c = Cs2[j];
	pivot_state = j;
      }
    all_matrix[i]->pivot_state = pivot_state;
  }
  
  // Check that eigenvectors on all partitions are synced
  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    for(int iproc=0; iproc<universe->nprocs; iproc++) {
      if(iproc == universe->me) {
	fprintf(stdout,"\n\nNew eigenvectors on rank %i of partition %i\n",iproc,universe->iworld);
	for(int i=0; i<ncomplex; i++) {
	  fprintf(stdout,"i = %i : nstate = %i  pivot= %i  C2 = ",i,all_complex[i]->nstate,all_matrix[i]->pivot_state);
	  int n = all_complex[i]->nstate;
	  if(n>16) n = 16;
	  for(int j=0; j<n; j++) fprintf(stdout," %f",all_complex[i]->Cs2[j]);
	  fprintf(stdout,"\n");
	}
      }
      MPI_Barrier(universe->uworld);
    }
  }

  TIMER_CLICK(EVB_Engine, sci_comm_evec_mp);
}

/* ----------------------------------------------------------------------*/
// Setup load-balancer to distribute work to all partitions.
// Called anytime state search is updated: 
//    EVB_engine::state_search and EVB_engine::delete_overlap
/* ----------------------------------------------------------------------*/

void EVB_Engine::setup_lb_mp()
{
/* ----------------------------------------------------------------------*/
// Initialize arrays
/* ----------------------------------------------------------------------*/

  int size = ncomplex * MAX_STATE;
  if(debug_sci_mp && universe->me==0) fprintf(stdout,"(%i,%i)  ncomplex= %i  MAX_STATE= %i  size= %i\n",universe->iworld,comm->me,ncomplex,MAX_STATE,size);

  if(!lb_tasklist) lb_tasklist = new int [size];
  memset(&lb_tasklist[0], -1, sizeof(int)*size);

  if(!lb_cplx_master) lb_cplx_master = new int [ncomplex];
  memset(&lb_cplx_master[0], -1, sizeof(int)*ncomplex);

  if(!lb_cplx_split) lb_cplx_split = new int [ncomplex];
  memset(&lb_cplx_split[0], -1, sizeof(int)*ncomplex);

  if(!lb_cplx_owned) lb_cplx_owned = new int [ncomplex];
  memset(&lb_cplx_owned[0], 0, sizeof(int)*ncomplex);

  if(!lb_cplx_block) {
    lb_cplx_block = new MPI_Comm [ncomplex];
    for(int i=0; i<ncomplex; i++) lb_cplx_block[i] = MPI_COMM_NULL;
  }

  if(!num_tasks_per_part) num_tasks_per_part = new int[universe->nworlds];
  for(int i=0; i<universe->nworlds; i++) num_tasks_per_part[i] = 0;

  if(!cplx_owner_list) cplx_owner_list = new int[universe->nworlds];
  if(!num_part_per_complex) num_part_per_complex = new int[ncomplex];
  if(!num_states_per_complex) num_states_per_complex = new int[ncomplex];

  if(!lb_num_owners) lb_num_owners = new int [ncomplex];
  memset(&lb_num_owners[0], 0, sizeof(int)*ncomplex);

  if(!lb_cplx_owner_list) lb_cplx_owner_list = new int [ncomplex * universe->nworlds];
  memset(&lb_cplx_owner_list[0], -1, sizeof(int)*ncomplex*universe->nworlds);

  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    for(int iproc=0; iproc<universe->nprocs; iproc++) {
      if(iproc == universe->me) {
	if(iproc == 0) fprintf(stdout,"\n\nStates at start of setup_lb_mp()\n");
	fprintf(stdout,"(%i,%i)  nstates = ",universe->iworld,comm->me);
	for(int i=0; i<ncomplex; i++) fprintf(stdout," %i",all_complex[i]->nstate);
	fprintf(stdout,"\n");
      }
      MPI_Barrier(universe->uworld);
    }
  }

/* ----------------------------------------------------------------------*/
// Generate tasklist
// 1: Complexes are distributed to partitions in round-robin fashion.
//     A partition completely owns a complex.
//
// 2: States are distributed to partitions in round-robin fashion.
//     Complexes can be split across partitions.
//     MPI comms are created after every update (bottleneck)
//
// 3: Complexes are pinned to group of partitions.
//     States are evenly distributed across paritions that own complex.
//     MPI comms are created only once.
//     Only makes sense if the number of partitions is a multiple of the number of complexes.
//
// 4: Same as 2, except communicators are not used. Instead, a new set of 
//     sci_comm_lb_mp_#() functions are written with direct Send/Recv with
//     complex master.
/* ----------------------------------------------------------------------*/

  // Count total number of current states spanning all complexes
  int ns = 0;
  for(int i=0; i<ncomplex; i++) ns+= all_complex[i]->nstate;

  // The strategy to distribute work is hard-coded.
  // The user will only have access to the optimal strategy.
  int strategy;
  if(universe->nworlds >= ncomplex) strategy = 5;
  else strategy = 4;

  if(debug_sci_mp)  {
    MPI_Barrier(universe->uworld);
    if(universe->me==0) fprintf(stdout,"(%i,%i)  Setting up tasklist  ns= %i\n",universe->iworld,comm->me,ns);
  }

  if(strategy == 1) {
    int ipart = 0;
    int n = 0;
    for(int i=0; i<ncomplex; i++) {
      if(ipart > universe->nworlds-1) ipart = 0;
      for(int j=0; j<all_complex[i]->nstate; j++) lb_tasklist[n++] = ipart;
      lb_cplx_master[i] = ipart;
      lb_cplx_split[i] = 0; // All states for this complex owned by single partition
      if(universe->iworld == ipart) lb_cplx_owned[i] = 1; // Partition owns part of complex
      num_tasks_per_part[ipart]++;
      ipart++;
    }
    if(ns != n) error->universe_all(FLERR,"Inconsistent number of states in tasklist");
  }

  /* ----------------------------------------------------------------------*/

  if(strategy == 2) {
    int num_states_per_part = ns / universe->nworlds;
    int num_extra_states = ns - universe->nworlds * num_states_per_part; // The remaining states to be distributed across the initial set of partitions
    
    if(debug_sci_mp) {
      fprintf(stdout,"(%i,%i) ns= %i  num_states_per_part= %i num_extra_states= %i\n",
	      universe->iworld,comm->me,ns,num_states_per_part,num_extra_states);
      MPI_Barrier(universe->uworld);
    }

    // Create group for everyone in block
    MPI_Group block_group;
    MPI_Comm_group(mp_verlet_sci->block, &block_group);

    int n = 0;
    int count = 0;
    int ipart = -1;
    for(int i=0; i<ncomplex; i++) {
      int num_owners = 0;
      memset(&cplx_owner_list[0], 0, sizeof(int)*universe->nworlds);
      for(int j=0; j<all_complex[i]->nstate; j++) {
	lb_cplx_split[i] = 0;              // Initially assume complex is completely owned by single partition
	if(count == 0) { // Assign states to next partition
	  ipart++;
	  if(ipart == universe->nworlds) ipart = universe->nworlds-1; // Last partition plays clean up
	}

	if(j==0) {
	  lb_cplx_master[i] = ipart;       // Partition with state 0 is owner of complex
	  cplx_owner_list[num_owners++] = ipart; // MPI rank in verlet block
	}
	lb_tasklist[n++] = ipart;
	if(universe->iworld == ipart) lb_cplx_owned[i] = 1; // Partion owns part of complex
	num_tasks_per_part[ipart]++;
	
	int id = ipart;
	if(id != cplx_owner_list[num_owners-1]) cplx_owner_list[num_owners++] = id;
	
	count++;
	
	// Reset counter if enough work was allocated to partition
	if(num_extra_states == 0) { // states evenly distributed across all partitions
	  if(count == num_states_per_part) count = 0;
	} else {
	  if(ipart < num_extra_states) {
	    if(count == num_states_per_part+1) count = 0; // partitions at start of list assigned an extra state
	  } else if(count == num_states_per_part) count = 0;
	}
	
	if(debug_sci_mp && universe->me == 0) {
	  fprintf(stdout,"(%i,%i)  n= %i  i= %i  j= %i  lb_tasklist= %i  num_owners= %i  cplx_owner_list= ",
		  universe->iworld,comm->me,n-1,i,j,ipart,num_owners);
	  for(int k=0; k<num_owners; k++) fprintf(stdout," %i",cplx_owner_list[k]);
	  fprintf(stdout,"\n");
	}

	// if(debug_sci_mp && universe->me==0) fprintf(stdout,"(%i,%i)  i= %i  nstate= %i  j= %i  ipart= %i  num_owners= %i  cplx_owner= %i\n",universe->iworld,comm->me,i,all_complex[i]->nstate,j,ipart,num_owners,cplx_owner_list[num_owners-1]);
      } // Loop over states

      // If several partitions own complex, setup sub-communicator between them.
      // This assumes that all partitions are same size and shape.
      if(num_owners>1) {
      	lb_cplx_split[i] = 1;

      	// First check if the members of this group have changed, if not, then no need to recreate the comm
      	bool comm_changed = false;
      	if(num_owners != lb_num_owners[i]) comm_changed = true; // Automatically know that comm has to be updated
      	else {
	  for(int j=0; j<num_owners; j++) {                     // Loop through list of owners and compare, update comm if different
	    if(lb_cplx_owner_list[i*universe->nworlds + j] != cplx_owner_list[j]) {
	      comm_changed = true;
	    }
	  }
	}
	comm_changed = true;

	//fprintf(stdout,"(%i,%i)  creating comm for complex %i.\n",universe->iworld,comm->me,i);

	count_comm_total++; // DEBUG
      	if(comm_changed) {
	  count_comm_create++; // DEBUG

      	  // Update list of owners
      	  lb_num_owners[i] = num_owners;
      	  for(int j=0; j<num_owners; j++) lb_cplx_owner_list[i*universe->nworlds + j] = cplx_owner_list[j];

      	  // Free old comm
      	  if(lb_cplx_block[i] != MPI_COMM_NULL) MPI_Comm_free(&(lb_cplx_block[i]));

      	  // Create new comm
      	  MPI_Group cplx_owner_group;
      	  MPI_Group_incl(block_group, num_owners, cplx_owner_list, &cplx_owner_group);
      	  MPI_Comm_create(mp_verlet_sci->block, cplx_owner_group, &(lb_cplx_block[i]));
      	  MPI_Group_free(&cplx_owner_group);
      	}
      } else { // If complex was originally split and now it isn't, free comm.
	if(lb_num_owners[i] != num_owners) {
	  lb_num_owners[i] = 1;
      	  if(lb_cplx_block[i] != MPI_COMM_NULL) MPI_Comm_free(&(lb_cplx_block[i]));
	}
      }

      if(debug_sci_mp) {
      	for(int iproc=0; iproc<universe->nprocs; iproc++) {
	  MPI_Barrier(universe->uworld);
      	  if(universe->me==iproc) {
      	    if(iproc==0) fprintf(stdout,"\ncomm blocks:\n");
      	    fprintf(stdout,"(%i,%i)  i= %i  nstate= %i  num_owners= %i  cplx_owner_list=",
      		    universe->iworld,comm->me,i,all_complex[i]->nstate,num_owners);
      	    for(int j=0; j<num_owners; j++) fprintf(stdout," %i",cplx_owner_list[j]);
      	    fprintf(stdout,"\n");
	    
	    int size = 0;
	    if(lb_cplx_block[i] != MPI_COMM_NULL) MPI_Comm_size(lb_cplx_block[i], &size);
	    if(comm->me==0) fprintf(stdout,"(%i,%i)  i= %i  size= %i\n",universe->iworld,comm->me,i,size);
      	  }
	  MPI_Barrier(universe->uworld);
	}
      }
	
    } // Loop over complexes

    if(ns != n) error->universe_all(FLERR,"Inconsistent number of states in tasklist");
  }

  /* ----------------------------------------------------------------------*/

  if(strategy == 3) {
    if(universe->nworlds%ncomplex != 0) error->universe_all(FLERR,"Partitions must be integer number of complexes for strategy 3.");

    int num_owners = universe->nworlds / ncomplex;
    memset(&cplx_owner_list[0], 0, sizeof(int)*universe->nworlds);
    
    // Create group for everyone in block
    MPI_Group block_group;
    if(lb_comm_update) MPI_Comm_group(mp_verlet_sci->block, &block_group);
    
    int n = 0;
    for(int i=0; i<ncomplex; i++) {
      int count = 0;
      int ipart_master = i * num_owners;
      int ipart = ipart_master - 1;
      
      if(num_owners > 1) lb_cplx_split[i] = 1;
      else lb_cplx_split[i] = 0;
      
      for(int j=0; j<num_owners; j++) {
	cplx_owner_list[j] = ipart_master + j; // MPI rank in verlet block
	if(universe->iworld == cplx_owner_list[j]) lb_cplx_owned[i] = 1; // Partion owns part of complex
      }
      
      int num_states_per_part = all_complex[i]->nstate / num_owners;
      int num_extra_states = all_complex[i]->nstate - num_owners * num_states_per_part; // The remaining states to be distributed across the initial set of partitions
      
      if(debug_sci_mp) {
	MPI_Barrier(universe->uworld);
	if(universe->me==0) fprintf(stdout,"(%i,%i) i= %i  nstate= %i  num_states_per_part= %i num_extra_states= %i\n",
		universe->iworld,comm->me,i,all_complex[i]->nstate,num_states_per_part,num_extra_states);
      }
      
      for(int j=0; j<all_complex[i]->nstate; j++) {
	if(count == 0) ipart++;
	if(j==0) lb_cplx_master[i] = ipart;       // Partition with state 0 is owner of complex
	
	lb_tasklist[n++] = ipart;
	num_tasks_per_part[ipart]++;
	
	count++;
	
	// Reset counter if enough work was allocated to partition
	if(num_extra_states == 0) { // states evenly distributed across all partitions
	  if(count == num_states_per_part) count = 0;
	} else {
	  if(ipart-ipart_master < num_extra_states) {
	    if(count == num_states_per_part+1) count = 0; // partitions at start of list assigned an extra state
	  } else if(count == num_states_per_part) count = 0;
	}
	
	if(debug_sci_mp && universe->me == 0) {
	  fprintf(stdout,"(%i,%i)  n= %i  i= %i  j= %i  lb_tasklist= %i  num_owners= %i  cplx_owner_list= ",
		  universe->iworld,comm->me,n-1,i,j,ipart,num_owners);
	  for(int k=0; k<num_owners; k++) fprintf(stdout," %i",cplx_owner_list[k]);
	  fprintf(stdout,"\n");
	}
	
	//if(debug_sci_mp && universe->me==0) fprintf(stdout,"(%i,%i)  i= %i  nstate= %i  j= %i  ipart= %i  num_owners= %i  cplx_owner= %i\n",universe->iworld,comm->me,i,all_complex[i]->nstate,j,ipart,num_owners,cplx_owner_list[num_owners-1]);
      } // Loop over states
      
	// If several partitions own complex, setup sub-communicator between them.
	// This assumes that all partitions are same size and shape.
      if(num_owners>1 && lb_comm_update) {	  
	// Update list of owners
	lb_num_owners[i] = num_owners;
	for(int j=0; j<num_owners; j++) lb_cplx_owner_list[i*universe->nworlds + j] = cplx_owner_list[j];
	
	// Free old comm
	if(lb_cplx_block[i] != MPI_COMM_NULL) MPI_Comm_free(&(lb_cplx_block[i]));
	
	// Create new comm
	MPI_Group cplx_owner_group;
	MPI_Group_incl(block_group, num_owners, cplx_owner_list, &cplx_owner_group);
	MPI_Comm_create(mp_verlet_sci->block, cplx_owner_group, &(lb_cplx_block[i]));
	MPI_Group_free(&cplx_owner_group);
      }
      
      // if(debug_sci_mp) {
      // 	MPI_Barrier(universe->uworld);
      // 	for(int iproc=0; iproc<universe->nprocs; iproc++) {
      // 	  if(universe->me==iproc) {
      // 	    if(iproc==0) fprintf(stdout,"\ncomm blocks:\n");
      // 	    fprintf(stdout,"(%i,%i)  i= %i  nstate= %i  num_owners= %i  cplx_owner_list=",
      // 		    universe->iworld,comm->me,i,all_complex[i]->nstate,num_owners);
      // 	    for(int j=0; j<num_owners; j++) fprintf(stdout," %i",cplx_owner_list[j]);
      // 	    fprintf(stdout,"\n");
      // 	  }
      // 	  MPI_Barrier(universe->uworld);
      // 	int size = 0;
      // 	if(lb_cplx_block[i] != MPI_COMM_NULL) MPI_Comm_size(lb_cplx_block[i], &size);
      // 	if(comm->me==0) fprintf(stdout,"(%i,%i)  i= %i  size= %i\n",universe->iworld,comm->me,i,size);
      // 	MPI_Barrier(universe->uworld);
      // }
      
    } // Loop over complexes
    
    lb_comm_update = 0; // Turn off updating of comms. Use the old ones from here on out.
    if(ns != n) error->universe_all(FLERR,"Inconsistent number of states in tasklist");
  }

  /* ----------------------------------------------------------------------*/

  if(strategy == 4) {
    int num_states_per_part = ns / universe->nworlds;
    int num_extra_states = ns - universe->nworlds * num_states_per_part; // The remaining states to be distributed across the initial set of partitions
    
    if(debug_sci_mp) {
      fprintf(stdout,"(%i,%i) ns= %i  num_states_per_part= %i num_extra_states= %i\n",
	      universe->iworld,comm->me,ns,num_states_per_part,num_extra_states);
      MPI_Barrier(universe->uworld);
    }

    int n = 0;
    int count = 0;
    int ipart = -1;
    for(int i=0; i<ncomplex; i++) {
      int num_owners = 0;
      memset(&cplx_owner_list[0], 0, sizeof(int)*universe->nworlds);
      for(int j=0; j<all_complex[i]->nstate; j++) {
	lb_cplx_split[i] = 0;              // Initially assume complex is completely owned by single partition
	if(count == 0) { // Assign states to next partition
	  ipart++;
	  if(ipart == universe->nworlds) ipart = universe->nworlds-1; // Last partition plays clean up
	}

	if(j==0) {
	  lb_cplx_master[i] = ipart;       // Partition with state 0 is owner of complex
	  cplx_owner_list[num_owners++] = ipart; // MPI rank in verlet block
	}
	lb_tasklist[n++] = ipart;
	if(universe->iworld == ipart) lb_cplx_owned[i] = 1; // Partion owns part of complex
	num_tasks_per_part[ipart]++;
	
	int id = ipart;
	if(id != cplx_owner_list[num_owners-1]) cplx_owner_list[num_owners++] = id;
	
	count++;
	
	// Reset counter if enough work was allocated to partition
	if(num_extra_states == 0) { // states evenly distributed across all partitions
	  if(count == num_states_per_part) count = 0;
	} else {
	  if(ipart < num_extra_states) {
	    if(count == num_states_per_part+1) count = 0; // partitions at start of list assigned an extra state
	  } else if(count == num_states_per_part) count = 0;
	}
	
	if(debug_sci_mp && universe->me == 0) {
	  fprintf(stdout,"(%i,%i)  n= %i  i= %i  j= %i  lb_tasklist= %i  num_owners= %i  cplx_owner_list= ",
		  universe->iworld,comm->me,n-1,i,j,ipart,num_owners);
	  for(int k=0; k<num_owners; k++) fprintf(stdout," %i",cplx_owner_list[k]);
	  fprintf(stdout,"\n");
	}

	// if(debug_sci_mp && universe->me==0) fprintf(stdout,"(%i,%i)  i= %i  nstate= %i  j= %i  ipart= %i  num_owners= %i  cplx_owner= %i\n",universe->iworld,comm->me,i,all_complex[i]->nstate,j,ipart,num_owners,cplx_owner_list[num_owners-1]);
      } // Loop over states

      // If several partitions own complex, setup sub-communicator between them.
      // This assumes that all partitions are same size and shape.
      if(num_owners>1) {
      	lb_cplx_split[i] = 1;

	// Update list of owners
	lb_num_owners[i] = num_owners;
	for(int j=0; j<num_owners; j++) lb_cplx_owner_list[i*universe->nworlds + j] = cplx_owner_list[j];
      }

      if(debug_sci_mp) {
      	for(int iproc=0; iproc<universe->nprocs; iproc++) {
	  MPI_Barrier(universe->uworld);
      	  if(universe->me==iproc) {
      	    if(iproc==0) fprintf(stdout,"\ncomm blocks:\n");
      	    fprintf(stdout,"(%i,%i)  i= %i  nstate= %i  num_owners= %i  cplx_owner_list=",
      		    universe->iworld,comm->me,i,all_complex[i]->nstate,num_owners);
      	    for(int j=0; j<num_owners; j++) fprintf(stdout," %i",cplx_owner_list[j]);
      	    fprintf(stdout,"\n");
      	  }
	  MPI_Barrier(universe->uworld);
	}
      }
	
    } // Loop over complexes

    if(ns != n) error->universe_all(FLERR,"Inconsistent number of states in tasklist");
  }

  if(strategy == 5) {
    
    // Determine number of partitions to assign each complex.
    for(int i=0; i<ncomplex; i++) num_part_per_complex[i] = 1;

    // Keep track of number of states of a complex per partiton as number of assigned partitions increases.
    for(int i=0; i<ncomplex; i++) num_states_per_complex[i] = all_complex[i]->nstate;

    int num_remaining_part = universe->nworlds - ncomplex;
    while(num_remaining_part > 0) {
      
      // Determine maximum states currently assigned to a partition
      int max = 0;
      for(int i=0; i<ncomplex; i++) if(num_states_per_complex[i] > max) max = num_states_per_complex[i];

      // Assign additional partitions to those complexes with the most states
      for(int i=0; i<ncomplex; i++) {
	if(num_states_per_complex[i] == max) {
	  num_part_per_complex[i]++;
	  num_remaining_part--;
	  num_states_per_complex[i] = all_complex[i]->nstate / num_part_per_complex[i];
	  if( (all_complex[i]->nstate)%num_part_per_complex[i] > 0) num_states_per_complex[i]++;

	  if(num_remaining_part == 0) break; // All partitions have been assigned.
	}
      }
    } // while(num_remaining_part > 0)

    int n = 0;
    int count = 0;
    int ipart = -1;
    for(int i=0; i<ncomplex; i++) {
      lb_num_owners[i] = num_part_per_complex[i];

      if(num_part_per_complex[i] > 1) lb_cplx_split[i] = 1;
      else lb_cplx_split[i] = 0;
      
      // The remaining states to be distributed across the initial set of partitions
      int avg_states_per_part = all_complex[i]->nstate / num_part_per_complex[i];
      int num_extra_states = all_complex[i]->nstate - num_part_per_complex[i] * avg_states_per_part; 

      int total = 0;
      for(int j=0; j<num_part_per_complex[i]; j++) {
	ipart++;
	if(j == 0) lb_cplx_master[i] = ipart;
	lb_cplx_owner_list[i * universe->nworlds + j] = ipart;
	int indx_state = 0;

	int count = avg_states_per_part;
	if(j < num_extra_states) count++;
	for(int k=0; k<count; k++) {
	    lb_tasklist[n++] = ipart;
	    num_tasks_per_part[ipart]++;
	    total++;
	}
	if(universe->iworld == ipart) lb_cplx_owned[i] = 1; // Partition owns part of complex
      }
      
      if(total != all_complex[i]->nstate) error->universe_all(FLERR,"Inconsistent number of states assigned to complex");

      if(debug_sci_mp) {
      	for(int iproc=0; iproc<universe->nprocs; iproc++) {
	  MPI_Barrier(universe->uworld);
      	  if(universe->me==iproc) {
      	    if(iproc==0) fprintf(stdout,"\ncomm blocks:\n");
      	    fprintf(stdout,"(%i,%i)  i= %i  nstate= %i  num_owners= %i  cplx_owner_list=",
      		    universe->iworld,comm->me,i,all_complex[i]->nstate,lb_num_owners[i]);
      	    for(int j=0; j<lb_num_owners[i]; j++) fprintf(stdout," %i",lb_cplx_owner_list[i*universe->nworlds + j]);
      	    fprintf(stdout,"\n");
      	  }
	  MPI_Barrier(universe->uworld);
	}
      }
	
    } // Loop over complexes

    if(ns != n) error->universe_all(FLERR,"Inconsistent number of states in tasklist");
  }

  /* ----------------------------------------------------------------------*/

  if(debug_sci_mp && universe->me == 0) {
    fprintf(stdout,"\n\n(%i,%i)  Total number of states %i\n",universe->iworld,comm->me,ns);
    for(int i=0; i<universe->nworlds; i++) fprintf(stdout,"(%i,%i)  i= %i  num_tasks_per_part= %i\n",
						   universe->iworld,comm->me,i,num_tasks_per_part[i]);
  }

  if(debug_sci_mp) {
    MPI_Barrier(universe->uworld);
    if(universe->iworld==0 && comm->me==0) {
      for(int i=0; i<ncomplex; i++) fprintf(stdout,"(%i,%i)  i= %i  cplx_master= %i  cplx_split= %i\n",
   					    universe->iworld,comm->me,i,lb_cplx_master[i],lb_cplx_split[i]);
    }
    MPI_Barrier(universe->uworld);
    
    if(strategy==2 || strategy==3) {
      for(int i=0; i<universe->nprocs; i++) {
	MPI_Barrier(universe->uworld);
   	if(universe->me==i) {
	  fprintf(stdout,"\n\n%i: Rank %i of %i in partition %i.\n",i,comm->me,comm->nprocs,universe->iworld);
	  for(int j=0; j<ncomplex; j++) {
	    int size = 0;
	    if(lb_cplx_owned[j] && lb_cplx_split[j]) MPI_Comm_size(lb_cplx_block[j], &size);
	    if(size == 0) fprintf(stdout,"(%i,%i)  j= %i  owned= %i  split= %i\n",universe->iworld,comm->me,j,lb_cplx_owned[j],lb_cplx_split[j]);
	    else fprintf(stdout,"(%i,%i)  j= %i  owned= %i  split= %i  size= %i  master= %i\n",universe->iworld,comm->me,j,lb_cplx_owned[j],lb_cplx_split[j],size,lb_cplx_master[j]);
	  }
	}
   	MPI_Barrier(universe->uworld);
      }
    }
  }

  // double ratio = double(count_comm_create) / double(count_comm_total); // DEBUG
  // if(universe->me==0) fprintf(stdout,"(%i,%i)  count_comm_create= %i/%i  = %f\n",universe->iworld,comm->me,
  // 			      count_comm_create,count_comm_total,ratio); // DEBUG
}

/* ---------------------------------------------------------------------- */
// Communicate necessary info to partition that owns complex
//  This is called in compute_sci_mp()
/* ---------------------------------------------------------------------- */

// void EVB_Engine::sci_comm_lb_mp_1()
// {
//   int cplx_id = evb_complex->id - 1;
//   MPI_Comm block = lb_cplx_block[cplx_id];

//   if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Inside sci_comm_lb_mp_1() for complex %i\n",universe->iworld,comm->me,cplx_id);
 
//   // Only master ranks on partitions will communicate energies
//   if(comm->me != 0) return;

//   // All partitions in block wait here till cplx_id's match
//   if(debug_sci_mp) {
//     MPI_Barrier(block);
//     int size_block;
//     MPI_Comm_size(block, &size_block);
//     int rank_block;
//     MPI_Comm_rank(block, &rank_block);
//     fprintf(stdout,"(%i,%i)  Processor %i is rank %i in block of size %i working on complex %i.\n",
// 	    universe->iworld,comm->me,universe->me,rank_block,size_block,cplx_id);
//   }
  
//   EVB_Complex *cplx = all_complex[cplx_id];

//   // diagonal + off-diagonal
//   int size = cplx->nstate * (EDIAG_NITEM + 1) + (cplx->nstate-1 + cplx->nextra_coupling) * EOFF_NITEM;
//   double ek[size];

//   // if(cplx_id == 6) {
//   //   for(int k=5; k<8; k++) {
//   //     if(universe->iworld==k) for(int j=0; j<cplx->nstate;j++) 
//   // 				fprintf(stdout,"(%i,%i) j= %i  e_diagonal= %f\n",universe->iworld,comm->me,j,evb_matrix->e_diagonal[j][0]);
//   //     MPI_Barrier(block);
//   //   }
//   // }

//   // Pack data. Assumes that entries are zero for those states owned by other partitions
//   int n = 0;
//   for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) ek[n++] = evb_matrix->e_diagonal[j][k];
//   for(int j=0; j<cplx->nstate; j++) ek[n++] = evb_matrix->e_repulsive[j];
//   for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) ek[n++] = evb_matrix->e_offdiag[j][k];
//   for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) ek[n++] = evb_matrix->e_extra[j][k];

//   // Send energies to master rank on partition that owns complex  double ek[size];
//   double ek2[size];
//   memset(&ek2, 0, sizeof(double)*size);
//   MPI_Reduce(&ek, &ek2, size, MPI_DOUBLE, MPI_SUM, 0, block);

//   // Unpack data
//   n = 0;
//   for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) evb_matrix->e_diagonal[j][k] = ek2[n++];
//   for(int j=0; j<cplx->nstate; j++) evb_matrix->e_repulsive[j] = ek2[n++];
//   for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) evb_matrix->e_offdiag[j][k] = ek2[n++];
//   for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) evb_matrix->e_extra[j][k] = ek2[n++];

//   if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Leaving sci_comm_lb_mp_1() for complex %i\n",universe->iworld,comm->me,cplx_id);
//   MPI_Barrier(block);
// }

void EVB_Engine::sci_comm_lb_mp_1()
{
  int cplx_id = evb_complex->id - 1;

  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Inside sci_comm_lb_mp_1() for complex %i  lb_cplx_owned= %i\n",universe->iworld,comm->me,cplx_id,lb_cplx_owned[cplx_id]);

  // Only master ranks on partitions that own part of a complex will communicate energies
  if(comm->me != 0) return;
  if(!lb_cplx_owned[cplx_id]) return;

  EVB_Complex *cplx = all_complex[cplx_id];
  MPI_Status status;

  // diagonal + off-diagonal/ nextra_coupling can be -1, so add 1 in case
  int size = cplx->nstate * (EDIAG_NITEM + 1) + (cplx->nstate + cplx->nextra_coupling) * EOFF_NITEM;
  if(size > max_comm_ek) {
    max_comm_ek = size;
    memory->grow(comm_ek, size, "EVB_Engine:comm_ek");
    memory->grow(comm_ek2, size, "EVB_Engine:comm_ek2");
  }

  // Pack data. Assumes that entries are zero for those states owned by other partitions
  int n = 0;
  for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) comm_ek[n++] = evb_matrix->e_diagonal[j][k];
  for(int j=0; j<cplx->nstate; j++) comm_ek[n++] = evb_matrix->e_repulsive[j];
  for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) comm_ek[n++] = evb_matrix->e_offdiag[j][k];
  for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) comm_ek[n++] = evb_matrix->e_extra[j][k];

  size = n; // Actual size of buffer to communicate

  // Send data to master
  if(universe->iworld != lb_cplx_master[cplx_id]) {
    int master = lb_cplx_owner_list[cplx_id * universe->nworlds];
    MPI_Send(&(comm_ek[0]), size, MPI_DOUBLE, master, 0, mp_verlet_sci->block);
    return;
  }

  // Receive data from slaves
  for(int i=1; i<lb_num_owners[cplx_id]; i++) {
    int sender = lb_cplx_owner_list[cplx_id * universe->nworlds + i];
    MPI_Recv(&(comm_ek2[0]), size, MPI_DOUBLE, sender, 0, mp_verlet_sci->block, &status);
    for(int j=0; j<size; j++) comm_ek[j]+= comm_ek2[j];
  }

  // Unpack data
  n = 0;
  for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) evb_matrix->e_diagonal[j][k] = comm_ek[n++];
  for(int j=0; j<cplx->nstate; j++) evb_matrix->e_repulsive[j] = comm_ek[n++];
  for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) evb_matrix->e_offdiag[j][k] = comm_ek[n++];
  for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) evb_matrix->e_extra[j][k] = comm_ek[n++];

  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Leaving sci_comm_lb_mp_1() for complex %i\n",universe->iworld,comm->me,cplx_id);
}

/* ---------------------------------------------------------------------- */
// Communicate necessary info to partition that owns complex
//  This is called in initialize_sci_mp().
/* ---------------------------------------------------------------------- */

// void EVB_Engine::sci_comm_lb_mp_2()
// {
//   int cplx_id = evb_complex->id - 1;
//   MPI_Comm block = lb_cplx_block[cplx_id];

//   if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Inside sci_comm_lb_mp_2() for complex %i\n",universe->iworld,comm->me,cplx_id);
 
//   // Only master ranks on partitions will communicate energies
//   if(comm->me != 0) return;

//   // All partitions in block wait here till cplx_id's match
//   // if(debug_sci_mp) {
//   //   MPI_Barrier(block);
//   //   int size_block;
//   //   MPI_Comm_size(block, &size_block);
//   //   int rank_block;
//   //   MPI_Comm_rank(block, &rank_block);
//   //   fprintf(stdout,"(%i,%i)  Processor %i is rank %i in block of size %i working on complex %i.\n",
//   // 	    universe->iworld,comm->me,universe->me,rank_block,size_block,cplx_id);
//   // }
  
//   EVB_Complex *cplx = all_complex[cplx_id];

//   // diagonal + off-diagonal
//   int size = cplx->nstate * (EDIAG_NITEM + 1) + (cplx->nstate-1 + cplx->nextra_coupling) * EOFF_NITEM;
//   double ek[size];

//   // if(cplx_id == 6) {
//   //   for(int k=5; k<8; k++) {
//   //     if(universe->iworld==k) for(int j=0; j<cplx->nstate;j++) 
//   // 				fprintf(stdout,"(%i,%i) j= %i  e_diagonal= %f\n",universe->iworld,comm->me,j,evb_matrix->e_diagonal[j][0]);
//   //     MPI_Barrier(block);
//   //   }
//   // }

//   // Pack data. Assumes that entries are zero for those states owned by other partitions
//   int n = 0;
//   for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) ek[n++] = all_matrix[cplx_id]->e_diagonal[j][k];
//   for(int j=0; j<cplx->nstate; j++) ek[n++] = all_matrix[cplx_id]->e_repulsive[j];
//   for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) ek[n++] = all_matrix[cplx_id]->e_offdiag[j][k];
//   for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) ek[n++] = all_matrix[cplx_id]->e_extra[j][k];

//   // Send energies to master rank on partition that owns complex  double ek[size];
//   double ek2[size];
//   memset(&ek2, 0, sizeof(double)*size);
//   MPI_Reduce(&ek, &ek2, size, MPI_DOUBLE, MPI_SUM, 0, block);

//   // Unpack data
//   n = 0;
//   for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) all_matrix[cplx_id]->e_diagonal[j][k] = ek2[n++];
//   for(int j=0; j<cplx->nstate; j++) all_matrix[cplx_id]->e_repulsive[j] = ek2[n++];
//   for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) all_matrix[cplx_id]->e_offdiag[j][k] = ek2[n++];
//   for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) all_matrix[cplx_id]->e_extra[j][k] = ek2[n++];

//   if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Leaving sci_comm_lb_mp_2() for complex %i\n",universe->iworld,comm->me,cplx_id);
//   MPI_Barrier(block);
// }

void EVB_Engine::sci_comm_lb_mp_2()
{
  int cplx_id = evb_complex->id - 1;

  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Inside sci_comm_lb_mp_2() for complex %i  lb_cplx_owned= %i\n",
			   universe->iworld,comm->me,cplx_id,lb_cplx_owned[cplx_id]);
 
  // Only master ranks on partitions will communicate energies
  if(comm->me != 0) return;
  if(!lb_cplx_owned[cplx_id]) return;
  
  EVB_Complex *cplx = all_complex[cplx_id];
  MPI_Status status;

  // diagonal + off-diagonal
  int size = cplx->nstate * (EDIAG_NITEM + 1) + (cplx->nstate-1 + cplx->nextra_coupling) * EOFF_NITEM;
  if(size > max_comm_ek) {
    max_comm_ek = size;
    memory->grow(comm_ek, size, "EVB_Engine:comm_ek");
    memory->grow(comm_ek2, size, "EVB_Engine:comm_ek2");
  }

  // Pack data. Assumes that entries are zero for those states owned by other partitions
  int n = 0;
  for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) comm_ek[n++] = all_matrix[cplx_id]->e_diagonal[j][k];
  for(int j=0; j<cplx->nstate; j++) comm_ek[n++] = all_matrix[cplx_id]->e_repulsive[j];
  for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) comm_ek[n++] = all_matrix[cplx_id]->e_offdiag[j][k];
  for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) comm_ek[n++] = all_matrix[cplx_id]->e_extra[j][k];

  // Send data to master
  if(universe->iworld != lb_cplx_master[cplx_id]) {
    int master = lb_cplx_owner_list[cplx_id * universe->nworlds];
    MPI_Send(&(comm_ek[0]), size, MPI_DOUBLE, master, 0, mp_verlet_sci->block);
    return;
  }
  
  // Receive data from slaves
  for(int i=1; i<lb_num_owners[cplx_id]; i++) {
    int sender = lb_cplx_owner_list[cplx_id * universe->nworlds + i];
    MPI_Recv(&(comm_ek2[0]), size, MPI_DOUBLE, sender, 0, mp_verlet_sci->block, &status);
    for(int j=0; j<size; j++) comm_ek[j]+= comm_ek2[j];
  }

  // Unpack data
  n = 0;
  for(int j=0; j<cplx->nstate; j++) for(int k=0; k<EDIAG_NITEM; k++) all_matrix[cplx_id]->e_diagonal[j][k] = comm_ek[n++];
  for(int j=0; j<cplx->nstate; j++) all_matrix[cplx_id]->e_repulsive[j] = comm_ek[n++];
  for(int j=0; j<cplx->nstate-1; j++) for(int k=0; k<EOFF_NITEM; k++) all_matrix[cplx_id]->e_offdiag[j][k] = comm_ek[n++];
  for(int j=0; j<cplx->nextra_coupling; j++) for(int k=0; k<EOFF_NITEM; k++) all_matrix[cplx_id]->e_extra[j][k] = comm_ek[n++];

  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Leaving sci_comm_lb_mp_2() for complex %i\n",universe->iworld,comm->me,cplx_id);
}

/* ---------------------------------------------------------------------- */
// Communicate necessary info to partition that owns complex
//  This is called in iteration_sci_mp().
/* ---------------------------------------------------------------------- */

// void EVB_Engine::sci_comm_lb_mp_3()
// {
//   int cplx_id = evb_complex->id - 1;
//   MPI_Comm block = lb_cplx_block[cplx_id];

//   if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Inside sci_comm_lb_mp_3() for complex %i\n",universe->iworld,comm->me,cplx_id);
 
//   // Only master ranks on partitions will communicate energies
//   if(comm->me != 0) return;

//   EVB_Complex *cplx = all_complex[cplx_id];

//   // diagonal + off-diagonal
//   int size = cplx->nstate * SCI_EDIAG_NITEM + cplx->nstate-1 + cplx->nextra_coupling;
//   double ek[size];

//   // if(cplx_id == 6) {
//   //   for(int k=5; k<8; k++) {
//   //     if(universe->iworld==k) for(int j=0; j<cplx->nstate;j++) 
//   // 				fprintf(stdout,"(%i,%i) j= %i  e_diagonal= %f\n",universe->iworld,comm->me,j,evb_matrix->e_diagonal[j][0]);
//   //     MPI_Barrier(block);
//   //   }
//   // }

//   // Pack data. Assumes that entries are zero for those states owned by other partitions
//   int n = 0;
//   for(int j=0; j<cplx->nstate; j++) for(int k=0; k<SCI_EDIAG_NITEM; k++) ek[n++] = all_matrix[cplx_id]->sci_e_diagonal[j][k];
//   for(int j=0; j<cplx->nstate-1; j++) ek[n++] = all_matrix[cplx_id]->sci_e_offdiag[j];
//   for(int j=0; j<cplx->nextra_coupling; j++) ek[n++] = all_matrix[cplx_id]->sci_e_extra[j];

//   // Send energies to master rank on partition that owns complex  double ek[size];
//   double ek2[size];
//   memset(&ek2, 0, sizeof(double)*size);
//   MPI_Reduce(&ek, &ek2, size, MPI_DOUBLE, MPI_SUM, 0, block);

//   // Unpack data
//   n = 0;
//   for(int j=0; j<cplx->nstate; j++) for(int k=0; k<SCI_EDIAG_NITEM; k++) all_matrix[cplx_id]->sci_e_diagonal[j][k] = ek2[n++];
//   for(int j=0; j<cplx->nstate-1; j++) all_matrix[cplx_id]->sci_e_offdiag[j] = ek2[n++];
//   for(int j=0; j<cplx->nextra_coupling; j++) all_matrix[cplx_id]->sci_e_extra[j] = ek2[n++];

//   if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Leaving sci_comm_lb_mp_2()\n",universe->iworld,comm->me); 
// }

void EVB_Engine::sci_comm_lb_mp_3()
{
  int cplx_id = evb_complex->id - 1;

  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Inside sci_comm_lb_mp_3() for complex %i  lb_cplx_owned= %i\n",
			   universe->iworld,comm->me,cplx_id,lb_cplx_owned[cplx_id]);
 
  // Only master ranks on partitions will communicate energies
  if(comm->me != 0) return;
  if(!lb_cplx_owned[cplx_id]) return;

  EVB_Complex *cplx = all_complex[cplx_id];
  MPI_Status status;

  // diagonal + off-diagonal
  int size = cplx->nstate * SCI_EDIAG_NITEM + cplx->nstate-1 + cplx->nextra_coupling;
  if(size > max_comm_ek) {
    max_comm_ek = size;
    memory->grow(comm_ek, size, "EVB_Engine:comm_ek");
    memory->grow(comm_ek2, size, "EVB_Engine:comm_ek2");
  }

  // Pack data. Assumes that entries are zero for those states owned by other partitions
  int n = 0;
  for(int j=0; j<cplx->nstate; j++) for(int k=0; k<SCI_EDIAG_NITEM; k++) comm_ek[n++] = all_matrix[cplx_id]->sci_e_diagonal[j][k];
  for(int j=0; j<cplx->nstate-1; j++) comm_ek[n++] = all_matrix[cplx_id]->sci_e_offdiag[j];
  for(int j=0; j<cplx->nextra_coupling; j++) comm_ek[n++] = all_matrix[cplx_id]->sci_e_extra[j];

  // Send energies to master
  if(universe->iworld != lb_cplx_master[cplx_id]) {
    int master = lb_cplx_owner_list[cplx_id * universe->nworlds];
    MPI_Send(&(comm_ek[0]), size, MPI_DOUBLE, master, 0, mp_verlet_sci->block);
    return;
  }
  
  // Receive data from slaves
  for(int i=1; i<lb_num_owners[cplx_id]; i++) {
    int sender = lb_cplx_owner_list[cplx_id * universe->nworlds + i];
    MPI_Recv(&(comm_ek2[0]), size, MPI_DOUBLE, sender, 0, mp_verlet_sci->block, &status);
    for(int j=0; j<size; j++) comm_ek[j]+= comm_ek2[j];
  }

  // Unpack data
  n = 0;
  for(int j=0; j<cplx->nstate; j++) for(int k=0; k<SCI_EDIAG_NITEM; k++) all_matrix[cplx_id]->sci_e_diagonal[j][k] = comm_ek[n++];
  for(int j=0; j<cplx->nstate-1; j++) all_matrix[cplx_id]->sci_e_offdiag[j] = comm_ek[n++];
  for(int j=0; j<cplx->nextra_coupling; j++) all_matrix[cplx_id]->sci_e_extra[j] = comm_ek[n++];

  if(debug_sci_mp) fprintf(stdout,"(%i,%i)  Leaving sci_comm_lb_mp_3()\n",universe->iworld,comm->me); 
}


/* ---------------------------------------------------------------------- */
// This function syncs avec for all states/complexes across all partitions
//  There is potential for lots of optimization here by eliminating 
//  unnecessary elements in the buffer input to Allreduce.
/* ---------------------------------------------------------------------- */

void EVB_Engine::sci_comm_avec()
{
  EVB_Complex *status = (EVB_Complex*)(evb_complex->status);

  if(!start_indx_int) start_indx_int = new int[ncomplex];
  if(!start_indx_double) start_indx_double = new int[ncomplex];
  if(!size_total_int) size_total_int = new int [ncomplex];
  if(!size_total_double) size_total_double = new int [ncomplex];

  memset(&(size_total_int[0]),    0, sizeof(int)*ncomplex);
  memset(&(size_total_double[0]), 0, sizeof(int)*ncomplex);

  // Determine size of arrays complexes owned by each partition.
  for(int icplx=0; icplx<ncomplex; icplx++) {
    if(lb_cplx_master[icplx] != universe->iworld) continue;
    
    evb_complex = all_complex[icplx];
    for(int i=0; i<evb_complex->nstate; i++) {
      int size_int = 0;
      int size_double = 0;
      for(int j=0; j<evb_complex->natom_cplx; j++) {
	int id = evb_complex->cplx_list[j];
	
	size_int++;  // size_int+= 4;
	size_double++;
	
	if(id>=atom->nlocal) continue;
	
	if(kernel_atom[id]==0) continue;
	
	// size_int+= 4;
	// size_int+= 2 * evb_complex->status[i].num_bond[j];
	// size_int+= 4 * evb_complex->status[i].num_angle[j];
	// size_int+= 5 * evb_complex->status[i].num_dihedral[j];
	// size_int+= 5 * evb_complex->status[i].num_improper[j];
      } // for(j<natom_cplx)

      // if(evb_kspace) size_double++;
      // size_int+= 4;

      if(i<evb_complex->nstate-1) {
	size_int++;                              // nexch_off
	size_int+= evb_complex->nexch_off[i];    // iexch_off
	size_double+= evb_complex->nexch_off[i]; // qexch_off
      }

      size_double+= 3; // r_coc
      
      size_total_int[icplx]+= size_int;
      size_total_double[icplx]+= size_double;
    } // for(i<nstate)
    
    if(evb_complex->nextra_coupling > 0) {
      for(int i=0; i<evb_complex->nextra_coupling; i++) {
    	size_total_int[icplx]++;                                 // nexch_extra
    	size_total_int[icplx]+= evb_complex->nexch_extra[i];     // iexch_extra
    	size_total_double[icplx]+= evb_complex->nexch_extra[i];  // qexch_extra
      }
    }

    if(debug_sci_mp && universe->me==0) fprintf(stdout,"id= %2i  nstate= %2i  natom_cplx= %3i  size_total_int= %6i  size_total_double= %6i\n",
						icplx,evb_complex->nstate,evb_complex->natom_cplx,size_total_int[icplx],size_total_double[icplx]);
  }

  // Collect buffer size of complexes on all partitions to determine total memory required as well as starting position for each complex.
  MPI_Allreduce(MPI_IN_PLACE, &(size_total_int[0]),    ncomplex, MPI_INT, MPI_SUM, mp_verlet_sci->block);
  MPI_Allreduce(MPI_IN_PLACE, &(size_total_double[0]), ncomplex, MPI_INT, MPI_SUM, mp_verlet_sci->block);

  int size_gtotal_int = 0;
  int size_gtotal_double = 0;
  for(int icplx=0; icplx<ncomplex; icplx++) {
    start_indx_int[icplx]    = size_gtotal_int;
    start_indx_double[icplx] = size_gtotal_double;

    size_gtotal_int+= size_total_int[icplx];
    size_gtotal_double+= size_total_double[icplx];
  }

  if(debug_sci_mp && universe->me==0) {
    fprintf(stdout,"\n\n");
    for(int icplx=0; icplx<ncomplex; icplx++) fprintf(stdout,"icplx= %i  nstate= %i  natom_cplx= %i  size_total_int= %6i  size_total_double= %6i  start_indx_int= %8i  start_indx_double= %8i\n",
						      icplx,all_complex[icplx]->nstate,all_complex[icplx]->natom_cplx,size_total_int[icplx],size_total_double[icplx],
						      start_indx_int[icplx],start_indx_double[icplx]);

    fprintf(stdout,"\n\nsize_gtotal_int= %8i  size_gtotal_double= %8i\n",size_gtotal_int,size_gtotal_double);
    fprintf(stdout,"size_gtotal_int= %8i  size_gtotal_double= %8i Bytes\n",
	    sizeof(int)*size_gtotal_int,sizeof(double)*size_gtotal_double);
    double s1 = sizeof(int)   *size_gtotal_int    /1024.0;
    double s2 = sizeof(double)*size_gtotal_double /1024.0;
    fprintf(stdout,"size_gtotal_int= %f  size_gtotal_double= %f KB\n",s1,s2);
    s1 = sizeof(int)   *size_gtotal_int    / 1024.0 / 1024.0;
    s2 = sizeof(double)*size_gtotal_double / 1024.0 / 1024.0;
    fprintf(stdout,"size_gtotal_int= %f  size_gtotal_double= %f MB\n\n\n",s1,s2);
  }

  // Grow buffers if needed
  if(size_gtotal_int > old_size_gtotal_int) {
    old_size_gtotal_int = size_gtotal_int;
    memory->grow(comm_avec_buf_int,     size_gtotal_int, "EVB_Engine::comm_avec_buf_int");
  }

  if(size_gtotal_double > old_size_gtotal_double) {
    old_size_gtotal_double = size_gtotal_double;
    memory->grow(comm_avec_buf_double,     size_gtotal_double, "EVB_Engine::comm_avec_buf_double");
  }

  // Pack avec into tmp buffer
  memset(&(comm_avec_buf_int[0]),    0, sizeof(int)*size_gtotal_int);
  memset(&(comm_avec_buf_double[0]), 0, sizeof(double)*size_gtotal_double);
  
  for(int icplx=0; icplx<ncomplex; icplx++) {
    if(lb_cplx_master[icplx] != universe->iworld) continue;
    
    // Starting position for this complex
    int pos_int = start_indx_int[icplx];
    int pos_double = start_indx_double[icplx];

    evb_complex = all_complex[icplx];
    for(int i=0; i<evb_complex->nstate; i++) {
      for(int j=0; j<evb_complex->natom_cplx; j++) {
	int id = evb_complex->cplx_list[j];
	
	comm_avec_buf_int[pos_int++]       = evb_complex->status[i].type[j];
	// comm_avec_buf_int[pos_int++]       = evb_complex->status[i].mol_type[j];
	// comm_avec_buf_int[pos_int++]       = evb_complex->status[i].mol_index[j];
	// comm_avec_buf_int[pos_int++]       = evb_complex->status[i].molecule[j];
	comm_avec_buf_double[pos_double++] = evb_complex->status[i].q[j];

	if(id >= atom->nlocal) continue;

	if(kernel_atom[id]==0) continue;

	// comm_avec_buf_int[pos_int++] = evb_complex->status[i].num_bond[j];
	// comm_avec_buf_int[pos_int++] = evb_complex->status[i].num_angle[j];
	// comm_avec_buf_int[pos_int++] = evb_complex->status[i].num_dihedral[j];
	// comm_avec_buf_int[pos_int++] = evb_complex->status[i].num_improper[j];

	// int n = evb_complex->status[i].num_bond[j];
	// for(int k=0; k<n; k++) {
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].bond_type[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].bond_atom[j][k];
	// }
	
	// n = evb_complex->status[i].num_angle[j];
	// for(int k=0; k<n; k++) {
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].angle_type[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].angle_atom1[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].angle_atom2[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].angle_atom3[j][k];
	// }

	// n = evb_complex->status[i].num_dihedral[j];
	// for(int k=0; k<n; k++) {
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].dihedral_type[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].dihedral_atom1[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].dihedral_atom2[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].dihedral_atom3[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].dihedral_atom4[j][k];
	// }

	// n = evb_complex->status[i].num_improper[j];
	// for(int k=0; k<n; k++) {
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].improper_type[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].improper_atom1[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].improper_atom2[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].improper_atom3[j][k];
	//   comm_avec_buf_int[pos_int++] = evb_complex->status[i].improper_atom4[j][k];
	// }
      } // loop over cplx_list
      
      // if(evb_kspace) comm_avec_buf_double[pos_double++] = evb_complex->status[i].qsqsum_cplx;
      
      // comm_avec_buf_int[pos_int++] = evb_complex->status[i].nbonds;
      // comm_avec_buf_int[pos_int++] = evb_complex->status[i].nangles;
      // comm_avec_buf_int[pos_int++] = evb_complex->status[i].ndihedrals;
      // comm_avec_buf_int[pos_int++] = evb_complex->status[i].nimpropers;

      // Off-diagonal exch charges
      if(i<evb_complex->nstate-1) {
	comm_avec_buf_int[pos_int++] = evb_complex->nexch_off[i];
	for(int j=0; j<evb_complex->nexch_off[i]; j++) {
	  comm_avec_buf_int[pos_int++]       = evb_complex->iexch_off[i][j];
	  comm_avec_buf_double[pos_double++] = evb_complex->qexch_off[i][j];
	}
      }

      // coordinates for center of charge
      comm_avec_buf_double[pos_double++] = evb_complex->cec->r_coc[i][0];
      comm_avec_buf_double[pos_double++] = evb_complex->cec->r_coc[i][1];
      comm_avec_buf_double[pos_double++] = evb_complex->cec->r_coc[i][2];
      
    } // loop over state

    // Extra off-diagonal exch charges
    if(evb_complex->nextra_coupling > 0) {
      for(int j=0; j<evb_complex->nextra_coupling; j++) {
    	comm_avec_buf_int[pos_int++] = evb_complex->nexch_extra[j];
    	for(int k=0; k<evb_complex->nexch_extra[j]; k++) {
    	  comm_avec_buf_int[pos_int++]       = evb_complex->iexch_extra[j][k];
    	  comm_avec_buf_double[pos_double++] = evb_complex->qexch_extra[j][k];
    	}
      }
    }
  } // loop over complex

  // Collect avec on all partitions
  MPI_Allreduce(MPI_IN_PLACE, &(comm_avec_buf_int[0]),    size_gtotal_int,    MPI_INT,    MPI_SUM, mp_verlet_sci->block);
  MPI_Allreduce(MPI_IN_PLACE, &(comm_avec_buf_double[0]), size_gtotal_double, MPI_DOUBLE, MPI_SUM, mp_verlet_sci->block);

  if(debug_sci_mp && universe->me==0) fprintf(stdout,"\n\n(%i,%i)  Unpacking avec.\n",universe->iworld,comm->me);

  // Unpack avec from buffer
  int pos_int = 0;
  int pos_double = 0;
  for(int icplx=0; icplx<ncomplex; icplx++) { 
    evb_complex = all_complex[icplx];
    for(int i=0; i<evb_complex->nstate; i++) {
      for(int j=0; j<evb_complex->natom_cplx; j++) {
      	int id = evb_complex->cplx_list[j];
	
	evb_complex->status[i].type[j]      = comm_avec_buf_int[pos_int++];
	// evb_complex->status[i].mol_type[j]  = comm_avec_buf_int[pos_int++];
	// evb_complex->status[i].mol_index[j] = comm_avec_buf_int[pos_int++];
	// evb_complex->status[i].molecule[j]  = comm_avec_buf_int[pos_int++];
	evb_complex->status[i].q[j] = comm_avec_buf_double[pos_double++];
	
       	if(id >= atom->nlocal) {
       	  evb_complex->status[i].num_bond[j]     = 0;
       	  evb_complex->status[i].num_angle[j]    = 0;
       	  evb_complex->status[i].num_dihedral[j] = 0;
       	  evb_complex->status[i].num_improper[j] = 0;
	  continue;
       	}
	
       	if(kernel_atom[id]==0) continue;

       	// evb_complex->status[i].num_bond[j]     = comm_avec_buf_int[pos_int++];
       	// evb_complex->status[i].num_angle[j]    = comm_avec_buf_int[pos_int++];
       	// evb_complex->status[i].num_dihedral[j] = comm_avec_buf_int[pos_int++];
       	// evb_complex->status[i].num_improper[j] = comm_avec_buf_int[pos_int++];

       	// int n = evb_complex->status[i].num_bond[j];
       	// for(int k=0; k<n; k++) {
       	//   evb_complex->status[i].bond_type[j][k] = comm_avec_buf_int[pos_int++];
	//   evb_complex->status[i].bond_atom[j][k] = comm_avec_buf_int[pos_int++];
	// }
	
	// n = evb_complex->status[i].num_angle[j];
	// for(int k=0; k<n; k++) {
	//   evb_complex->status[i].angle_type[j][k]  = comm_avec_buf_int[pos_int++];
	//   evb_complex->status[i].angle_atom1[j][k] = comm_avec_buf_int[pos_int++];
	//   evb_complex->status[i].angle_atom2[j][k] = comm_avec_buf_int[pos_int++];
	//   evb_complex->status[i].angle_atom3[j][k] = comm_avec_buf_int[pos_int++];
	// }
	
       	// n = evb_complex->status[i].num_dihedral[j];
       	// for(int k=0; k<n; k++) {
       	//   evb_complex->status[i].dihedral_type[j][k]  = comm_avec_buf_int[pos_int++];
       	//   evb_complex->status[i].dihedral_atom1[j][k] = comm_avec_buf_int[pos_int++];
       	//   evb_complex->status[i].dihedral_atom2[j][k] = comm_avec_buf_int[pos_int++];
       	//   evb_complex->status[i].dihedral_atom3[j][k] = comm_avec_buf_int[pos_int++];
       	//   evb_complex->status[i].dihedral_atom4[j][k] = comm_avec_buf_int[pos_int++];
       	// }

       	// n = evb_complex->status[i].num_improper[j];
       	// for(int k=0; k<n; k++) {
       	//   evb_complex->status[i].improper_type[j][k]  = comm_avec_buf_int[pos_int++];
       	//   evb_complex->status[i].improper_atom1[j][k] = comm_avec_buf_int[pos_int++];
       	//   evb_complex->status[i].improper_atom2[j][k] = comm_avec_buf_int[pos_int++];
       	//   evb_complex->status[i].improper_atom3[j][k] = comm_avec_buf_int[pos_int++];
       	//   evb_complex->status[i].improper_atom4[j][k] = comm_avec_buf_int[pos_int++];
       	// }
      } // loop over cplx_list

      // if(evb_kspace) evb_complex->status[i].qsqsum_cplx = comm_avec_buf_double[pos_double++];
      
      // evb_complex->status[i].nbonds     = comm_avec_buf_int[pos_int++];
      // evb_complex->status[i].nangles    = comm_avec_buf_int[pos_int++];
      // evb_complex->status[i].ndihedrals = comm_avec_buf_int[pos_int++];
      // evb_complex->status[i].nimpropers = comm_avec_buf_int[pos_int++];

      // Off-diagonal exch charges
      if(i<evb_complex->nstate-1) {
	evb_complex->nexch_off[i] = comm_avec_buf_int[pos_int++];
	for(int j=0; j<evb_complex->nexch_off[i]; j++) {
	  evb_complex->iexch_off[i][j] = comm_avec_buf_int[pos_int++];
	  evb_complex->qexch_off[i][j] = comm_avec_buf_double[pos_double++];
	}
      }

      // coordinates for center of charge
      evb_complex->cec->r_coc[i][0] = comm_avec_buf_double[pos_double++];
      evb_complex->cec->r_coc[i][1] = comm_avec_buf_double[pos_double++];
      evb_complex->cec->r_coc[i][2] = comm_avec_buf_double[pos_double++];

    } // loop over state

    // Extra off-diagonal exch charges
    if(evb_complex->nextra_coupling > 0) {
      for(int j=0; j<evb_complex->nextra_coupling; j++) {
    	evb_complex->nexch_extra[j] = comm_avec_buf_int[pos_int++];
    	for(int k=0; k<evb_complex->nexch_extra[j]; k++) {
    	  evb_complex->iexch_extra[j][k] = comm_avec_buf_int[pos_int++];
    	  evb_complex->qexch_extra[j][k] = comm_avec_buf_double[pos_double++];
    	}
      }
    }

  } // loop over complex
}


/* ---------------------------------------------------------------------- */
// This function syncs avec for new pivot state for all complexes across partitions
/* ---------------------------------------------------------------------- */

void EVB_Engine::sci_comm_avec_pivot()
{
  EVB_Complex *status = (EVB_Complex*)(evb_complex->status);

  memset(&(size_total_int[0]),    0, sizeof(int)*ncomplex);
  memset(&(size_total_double[0]), 0, sizeof(int)*ncomplex);

  // Determine size of arrays complexes owned by each partition.
  for(int icplx=0; icplx<ncomplex; icplx++) {
    if(lb_cplx_master[icplx] != universe->iworld) continue;
    
    evb_complex = all_complex[icplx];
    int pivot = all_matrix[icplx]->pivot_state;
    if(pivot==0) continue;
    int size_int = 0;
    int size_double = 0;
    for(int j=0; j<evb_complex->natom_cplx; j++) {
      int id = evb_complex->cplx_list[j];
	
      size_int+= 4;
      size_double++;
	
      if(id>=atom->nlocal) continue;
      
      if(kernel_atom[id]==0) continue;
      
      size_int+= 4;
      size_int+= 2 * evb_complex->status[pivot].num_bond[j];
      size_int+= 4 * evb_complex->status[pivot].num_angle[j];
      size_int+= 5 * evb_complex->status[pivot].num_dihedral[j];
      size_int+= 5 * evb_complex->status[pivot].num_improper[j];
    } // for(j<natom_cplx)
    if(evb_kspace) size_double++;
    size_int+= 4;
      
    size_total_int[icplx]+= size_int;
    size_total_double[icplx]+= size_double;

    if(debug_sci_mp && universe->me==0) fprintf(stdout,"id= %2i  nstate= %2i  natom_cplx= %3i  size_total_int= %6i  size_total_double= %6i\n",
						icplx,evb_complex->nstate,evb_complex->natom_cplx,size_total_int[icplx],size_total_double[icplx]);
  }

  // Collect buffer size of complexes on all partitions to determine total memory required as well as starting position for each complex.
  MPI_Allreduce(MPI_IN_PLACE, &(size_total_int[0]),    ncomplex, MPI_INT, MPI_SUM, mp_verlet_sci->block);
  MPI_Allreduce(MPI_IN_PLACE, &(size_total_double[0]), ncomplex, MPI_INT, MPI_SUM, mp_verlet_sci->block);

  int size_gtotal_int = 0;
  int size_gtotal_double = 0;
  for(int icplx=0; icplx<ncomplex; icplx++) {
    start_indx_int[icplx]    = size_gtotal_int;
    start_indx_double[icplx] = size_gtotal_double;

    size_gtotal_int+= size_total_int[icplx];
    size_gtotal_double+= size_total_double[icplx];
  }

  if(debug_sci_mp && universe->me==0) {
    fprintf(stdout,"\n\n");
    for(int icplx=0; icplx<ncomplex; icplx++) fprintf(stdout,"icplx= %i  nstate= %i  natom_cplx= %i  size_total_int= %6i  size_total_double= %6i  start_indx_int= %8i  start_indx_double= %8i\n",
						      icplx,all_complex[icplx]->nstate,all_complex[icplx]->natom_cplx,size_total_int[icplx],size_total_double[icplx],
						      start_indx_int[icplx],start_indx_double[icplx]);

    fprintf(stdout,"\n\nsize_gtotal_int= %8i  size_gtotal_double= %8i\n",size_gtotal_int,size_gtotal_double);
    fprintf(stdout,"size_gtotal_int= %8i  size_gtotal_double= %8i Bytes\n",
	    sizeof(int)*size_gtotal_int,sizeof(double)*size_gtotal_double);
    double s1 = sizeof(int)   *size_gtotal_int    /1024.0;
    double s2 = sizeof(double)*size_gtotal_double /1024.0;
    fprintf(stdout,"size_gtotal_int= %f  size_gtotal_double= %f KB\n",s1,s2);
    s1 = sizeof(int)   *size_gtotal_int    / 1024.0 / 1024.0;
    s2 = sizeof(double)*size_gtotal_double / 1024.0 / 1024.0;
    fprintf(stdout,"size_gtotal_int= %f  size_gtotal_double= %f MB\n\n\n",s1,s2);
  }

  // Grow buffers if needed
  if(size_gtotal_int > old_size_gtotal_int) {
    old_size_gtotal_int = size_gtotal_int;
    memory->grow(comm_avec_buf_int,     size_gtotal_int, "EVB_Engine::comm_avec_buf_int");
  }

  if(size_gtotal_double > old_size_gtotal_double) {
    old_size_gtotal_double = size_gtotal_double;
    memory->grow(comm_avec_buf_double,     size_gtotal_double, "EVB_Engine::comm_avec_buf_double");
  }

  // Pack avec into tmp buffer
  memset(&(comm_avec_buf_int[0]),    0, sizeof(int)*size_gtotal_int);
  memset(&(comm_avec_buf_double[0]), 0, sizeof(double)*size_gtotal_double);
  
  for(int icplx=0; icplx<ncomplex; icplx++) {
    if(lb_cplx_master[icplx] != universe->iworld) continue;
    
    // Starting position for this complex
    int pos_int = start_indx_int[icplx];
    int pos_double = start_indx_double[icplx];

    evb_complex = all_complex[icplx];
    int pivot = all_matrix[icplx]->pivot_state;
    if(pivot==0) continue;
    for(int j=0; j<evb_complex->natom_cplx; j++) {
      int id = evb_complex->cplx_list[j];
	
      comm_avec_buf_int[pos_int++]       = evb_complex->status[pivot].type[j];
      comm_avec_buf_int[pos_int++]       = evb_complex->status[pivot].mol_type[j];
      comm_avec_buf_int[pos_int++]       = evb_complex->status[pivot].mol_index[j];
      comm_avec_buf_int[pos_int++]       = evb_complex->status[pivot].molecule[j];
      comm_avec_buf_double[pos_double++] = evb_complex->status[pivot].q[j];
      
      if(id >= atom->nlocal) continue;
      
      if(kernel_atom[id]==0) continue;
      
      comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].num_bond[j];
      comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].num_angle[j];
      comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].num_dihedral[j];
      comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].num_improper[j];

      int n = evb_complex->status[pivot].num_bond[j];
      for(int k=0; k<n; k++) {
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].bond_type[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].bond_atom[j][k];
      }
	
      n = evb_complex->status[pivot].num_angle[j];
      for(int k=0; k<n; k++) {
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].angle_type[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].angle_atom1[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].angle_atom2[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].angle_atom3[j][k];
      }

      n = evb_complex->status[pivot].num_dihedral[j];
      for(int k=0; k<n; k++) {
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].dihedral_type[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].dihedral_atom1[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].dihedral_atom2[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].dihedral_atom3[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].dihedral_atom4[j][k];
      }

      n = evb_complex->status[pivot].num_improper[j];
      for(int k=0; k<n; k++) {
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].improper_type[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].improper_atom1[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].improper_atom2[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].improper_atom3[j][k];
	comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].improper_atom4[j][k];
      }
    } // loop over cplx_list
      
    if(evb_kspace) comm_avec_buf_double[pos_double++] = evb_complex->status[pivot].qsqsum_cplx;
      
    comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].nbonds;
    comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].nangles;
    comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].ndihedrals;
    comm_avec_buf_int[pos_int++] = evb_complex->status[pivot].nimpropers;
  } // loop over complex

  // Collect avec on all partitions
  MPI_Allreduce(MPI_IN_PLACE, &(comm_avec_buf_int[0]),    size_gtotal_int,    MPI_INT,    MPI_SUM, mp_verlet_sci->block);
  MPI_Allreduce(MPI_IN_PLACE, &(comm_avec_buf_double[0]), size_gtotal_double, MPI_DOUBLE, MPI_SUM, mp_verlet_sci->block);

  if(debug_sci_mp && universe->me==0) fprintf(stdout,"\n\n(%i,%i)  Unpacking avec.\n",universe->iworld,comm->me);

  // Unpack avec from buffer  
  int pos_int = 0;
  int pos_double = 0;
  for(int icplx=0; icplx<ncomplex; icplx++) { 
    evb_complex = all_complex[icplx];
    if(all_matrix[icplx]->pivot_state==0) continue;
    for(int j=0; j<evb_complex->natom_cplx; j++) {
      int id = evb_complex->cplx_list[j];
	
      evb_complex->type[id]      = comm_avec_buf_int[pos_int++];
      evb_complex->mol_type[id]  = comm_avec_buf_int[pos_int++];
      evb_complex->mol_index[id] = comm_avec_buf_int[pos_int++];
      evb_complex->molecule[id]  = comm_avec_buf_int[pos_int++];
      evb_complex->q[id]         = comm_avec_buf_double[pos_double++];
      
      if(id >= atom->nlocal) {
	evb_complex->num_bond[id]     = 0;
	evb_complex->num_angle[id]    = 0;
	evb_complex->num_dihedral[id] = 0;
	evb_complex->num_improper[id] = 0;
	continue;
      }
      
      if(kernel_atom[id]==0) continue;
      
      evb_complex->num_bond[id]     = comm_avec_buf_int[pos_int++];
      evb_complex->num_angle[id]    = comm_avec_buf_int[pos_int++];
      evb_complex->num_dihedral[id] = comm_avec_buf_int[pos_int++];
      evb_complex->num_improper[id] = comm_avec_buf_int[pos_int++];
      
      for(int k=0; k<evb_complex->num_bond[id]; k++) {
	evb_complex->bond_type[id][k] = comm_avec_buf_int[pos_int++];
	evb_complex->bond_atom[id][k] = comm_avec_buf_int[pos_int++];
      }
      
      for(int k=0; k<evb_complex->num_angle[id]; k++) {
	evb_complex->angle_type[id][k]  = comm_avec_buf_int[pos_int++];
	evb_complex->angle_atom1[id][k] = comm_avec_buf_int[pos_int++];
	evb_complex->angle_atom2[id][k] = comm_avec_buf_int[pos_int++];
	evb_complex->angle_atom3[id][k] = comm_avec_buf_int[pos_int++];
      }

      for(int k=0; k<evb_complex->num_dihedral[id]; k++) {
	evb_complex->dihedral_type[id][k]  = comm_avec_buf_int[pos_int++];
	evb_complex->dihedral_atom1[id][k] = comm_avec_buf_int[pos_int++];
	evb_complex->dihedral_atom2[id][k] = comm_avec_buf_int[pos_int++];
	evb_complex->dihedral_atom3[id][k] = comm_avec_buf_int[pos_int++];
	evb_complex->dihedral_atom4[id][k] = comm_avec_buf_int[pos_int++];
      }

      for(int k=0; k<evb_complex->num_improper[id]; k++) {
	evb_complex->improper_type[id][k]  = comm_avec_buf_int[pos_int++];
	evb_complex->improper_atom1[id][k] = comm_avec_buf_int[pos_int++];
	evb_complex->improper_atom2[id][k] = comm_avec_buf_int[pos_int++];
	evb_complex->improper_atom3[id][k] = comm_avec_buf_int[pos_int++];
	evb_complex->improper_atom4[id][k] = comm_avec_buf_int[pos_int++];
      }
    } // loop over cplx_list
    
    if(evb_kspace) evb_complex->qsqsum = comm_avec_buf_double[pos_double++];
    
    atom->nbonds     = comm_avec_buf_int[pos_int++];
    atom->nangles    = comm_avec_buf_int[pos_int++];
    atom->ndihedrals = comm_avec_buf_int[pos_int++];
    atom->nimpropers = comm_avec_buf_int[pos_int++];
  } // loop over complex
}

/* ----------------------------------------------------------------------*/
// This function parallelizes the per-complex state_search across partitions
/* ---------------------------------------------------------------------- */

void EVB_Engine::state_search_sci_mp()
{
  int me = comm->me;
  int iworld = universe->iworld;
  int nw = universe->nworlds;
  int max_shells = ncomplex * evb_chain->max_shell;
  MPI_Status status;

  if(!buf_size_all) buf_size_all = new int[ncomplex];
  if(!buf_nextra_all) buf_nextra_all = new int[ncomplex];
  if(!buf_shell_all) buf_shell_all = new int[ncomplex*max_shells];
  if(!start_indx) start_indx = new int[ncomplex];

  memset(&(buf_size_all[0]), 0, sizeof(int)*ncomplex);
  memset(&(buf_nextra_all[0]), 0, sizeof(int)*ncomplex);
  memset(&(buf_shell_all[0]), 0, sizeof(int)*ncomplex*max_shells);
  memset(&(start_indx[0]), 0, sizeof(int)*ncomplex);

  // Search states at each proc
  // Complexes are distributed across partitions round-robin.
  for(int i=0; i<ncomplex; i++) { 
    if(i%nw == iworld) {
      // fprintf(stdout,"(%i,%i)  state_search() on complex %i\n",universe->iworld,comm->me,i);
      
      // State search for this complex
      if(me == rc_rank[i]) {
	all_complex[i]->rc_start = rc_molecule[i];
	all_complex[i]->search_state();
	all_complex[i]->pack_state();
      }
      
      // If necessary, send states to master rank
      if(rc_rank[i] != 0) {
	if(me == rc_rank[i]) {
	  MPI_Send(&(all_complex[i]->buf_size),        1,                        MPI_INT, 0, 0, world);
	  MPI_Send(  all_complex[i]->state_buf,        all_complex[i]->buf_size, MPI_INT, 0, 0, world);
	  MPI_Send(&(all_complex[i]->nextra_coupling), 1,                        MPI_INT, 0, 0, world);
	  MPI_Send(all_complex[i]->state_per_shell,    evb_chain->max_shell,     MPI_INT, 0, 0, world);
	}
	
	if(me == 0) {
	  MPI_Recv(&(all_complex[i]->buf_size),        1,                        MPI_INT, rc_rank[i], 0, world, &status);
	  MPI_Recv(  all_complex[i]->state_buf,        all_complex[i]->buf_size, MPI_INT, rc_rank[i], 0, world, &status);
	  MPI_Recv(&(all_complex[i]->nextra_coupling), 1,                        MPI_INT, rc_rank[i], 0, world, &status);
	  MPI_Recv(all_complex[i]->state_per_shell,    evb_chain->max_shell,     MPI_INT, rc_rank[i], 0, world, &status);
	  all_complex[i]->unpack_state();
	}
      }
      
      // Master rank populates per-complex portion of global arrays
      if(me == 0) {
	buf_size_all[i]   = all_complex[i]->buf_size;
	buf_nextra_all[i] = all_complex[i]->nextra_coupling;
	for(int j=0; j<evb_chain->max_shell; j++) buf_shell_all[i*evb_chain->max_shell+j]  = all_complex[i]->state_per_shell[j];
      }
    }
    
  }
  
  // Allreduce buf_size, nextra_coupling, state_per_shell on master ranks
  if(me == 0) {
    MPI_Allreduce(MPI_IN_PLACE, &(buf_size_all[0]),   ncomplex,            MPI_INT, MPI_SUM, mp_verlet_sci->block);
    MPI_Allreduce(MPI_IN_PLACE, &(buf_nextra_all[0]), ncomplex,            MPI_INT, MPI_SUM, mp_verlet_sci->block);
    MPI_Allreduce(MPI_IN_PLACE, &(buf_shell_all[0]),  ncomplex*max_shells, MPI_INT, MPI_SUM, mp_verlet_sci->block);
  }

  // Each master rank Bcasts buf_size, nextra_coupling, state_per_shell to rest of partitions
  MPI_Bcast(&(buf_size_all[0]),   ncomplex,            MPI_INT, 0, world);
  MPI_Bcast(&(buf_nextra_all[0]), ncomplex,            MPI_INT, 0, world);
  MPI_Bcast(&(buf_shell_all[0]),  ncomplex*max_shells, MPI_INT, 0, world);

  // Starting positions for each complex for state_buf.
  start_indx[0] = 0;
  for(int i=1; i<ncomplex; i++) start_indx[i]+= start_indx[i-1] + buf_size_all[i-1];
  int size = start_indx[ncomplex-1] + buf_size_all[ncomplex-1];
  
  // Master ranks pack per-complex state_buf into global buffer
  if(size > max_tmp_state_buf) {
    max_tmp_state_buf = size;
    memory->grow(tmp_state_buf, size, "EVB_Engine:tmp_state_buf");
  }
  memset(&(tmp_state_buf[0]), 0, sizeof(int)*size);

  for(int i=0; i<ncomplex; i++) {
    if(i%nw != iworld) continue;

    int pos= start_indx[i];
    for(int j=0; j<buf_size_all[i]; j++) tmp_state_buf[pos++] = all_complex[i]->state_buf[j];
  }

  // Allreduce state_buf on master ranks
  if(comm->me == 0) MPI_Allreduce(MPI_IN_PLACE, &(tmp_state_buf[0]), size, MPI_INT, MPI_SUM, mp_verlet_sci->block);

  // Each master rank Bcasts state_buf to rest of partitions
  MPI_Bcast(&(tmp_state_buf[0]), size, MPI_INT, 0, world);

  // Everybody unpacks state_search
  int pos = 0;
  for(int i=0; i<ncomplex; i++) {
    all_complex[i]->buf_size        = buf_size_all[i];
    all_complex[i]->nextra_coupling = buf_nextra_all[i];
    for(int j=0; j<evb_chain->max_shell; j++) all_complex[i]->state_per_shell[j] = buf_shell_all[i*evb_chain->max_shell+j];
    
    for(int j=0; j<buf_size_all[i]; j++) all_complex[i]->state_buf[j] = tmp_state_buf[pos++];
    all_complex[i]->unpack_state();
    all_complex[i]->compute_qsqsum();
  }

  if(debug_sci_mp) {
    // MPI_Barrier(universe->uworld);
    // for(int i=0; i<universe->nprocs; i++) {
    //   if(universe->me == i) {
    // 	fprintf(stdout,"(%i,%i)  rc_molecule in state_search on rank %i\n",universe->iworld,comm->me,i);
    // 	for(int j=0; j<ncomplex; j++) fprintf(stdout,"(%i,%i)  j= %i  rc_molecule= %i\n",universe->iworld,comm->me,j,rc_molecule[j]);
    //   }
    //   MPI_Barrier(universe->uworld);
    // }
  
    MPI_Barrier(universe->uworld);
    for(int iproc=0; iproc<universe->nprocs; iproc++) {
      if(iproc == universe->me) {
	if(iproc == 0) fprintf(stdout,"\n\nStates at end of state_search_sci_mp()\n");
	fprintf(stdout,"(%i,%i)  nstates = ",universe->iworld,comm->me);
	for(int i=0; i<ncomplex; i++) fprintf(stdout," %i",all_complex[i]->nstate);
	fprintf(stdout,"\n");
      }
      MPI_Barrier(universe->uworld);
    }
  }

}

/* ---------------------------------------------------------------------- */

void EVB_Engine::compute_diagonal_mp(int vflag)
{
  TIMER_STAMP(EVB_Engine, compute_diagonal);

  int save_ago = neighbor->ago;
 
  if(force->pair) evb_effpair->compute_pair_mp(vflag); // All partitions assist with environment calculation

  if(!mp_verlet_sci->is_master) return;

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

  TIMER_CLICK(EVB_Engine, compute_diagonal);
}

void EVB_Engine::sci_comm_cplx_map()
{
  int nall = atom->nlocal + atom->nghost;
  int size = 3 * ncomplex;
  // fprintf(stdout,"(%i,%i) size= %i  %f MB\n",universe->iworld,comm->me,size,size*sizeof(int)/1024.0/1024.0);
  
  memset(&(sci_cplx_count[0]), 0, size*sizeof(int));
  int n;
  for(int i=0; i<ncomplex; i++) { 
    evb_complex = all_complex[i];
    if(universe->iworld == lb_cplx_master[i]) {
      evb_complex->build_cplx_map();
      
      sci_cplx_count[i*3]   = evb_complex->natom_cplx;
      sci_cplx_count[i*3+1] = evb_complex->nghost_cplx;
      sci_cplx_count[i*3+2] = evb_complex->nlocal_cplx;
    } else {
      if(nall > evb_complex->natom) {
	evb_complex->natom = nall;
	evb_complex->cplx_list = (int*) memory->srealloc(evb_complex->cplx_list, nall*sizeof(int), "EVB_Complex:cplx_list");
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &(sci_cplx_count[0]), size, MPI_INT, MPI_SUM, mp_verlet_sci->block);

  // Update nlocal_cplx and nghost_cplx for evb_kspace->evb_setup()
  for(int i=0; i<ncomplex; i++) {
    all_complex[i]->natom_cplx  = sci_cplx_count[i*3];
    all_complex[i]->nghost_cplx = sci_cplx_count[i*3+1];
    all_complex[i]->nlocal_cplx = sci_cplx_count[i*3+2];
  }
}

#endif