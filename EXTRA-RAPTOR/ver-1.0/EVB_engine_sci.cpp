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
#include "error.h"
#include "update.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "neighbor.h"
#include "comm.h"
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
#include "EVB_kspace.h"
#include "EVB_output.h"
#include "EVB_effpair.h"
#include "EVB_offdiag.h"
#include "EVB_repul.h"
#include "EVB_timer.h"

#define MAX_CIRCLE 10
#define CNVG_LIMIT 1e-6

#define KSPACE_DEFAULT    0 // Hellman-Feynman forces for Ewald
#define PPPM_HF_FORCES    1 // Hellman-Feynman forces for PPPM
#define PPPM_ACC_FORCES   2 // Approximate (acc) forces for PPPM. 
#define PPPM_POLAR_FORCES 3 // ACC forces plus an additional polarization force on complex atoms for PPPM.

//#define __DEBUG

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

void EVB_Engine::delete_overlap()
{  
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
	  if(cplx->molecule_B[m]==mol && cplx->Cs2[m]>new_C) new_C = cplx->Cs2[m];
	
	// If this molecule has not been used, label it.
	if(complex_molecule[mol]==0) {
	  complex_molecule[mol] = i+1;
	  max_coeff[mol] = new_C;
	} else {    // Overlap occurs
	  EVB_Complex* target;
	  
	  // Identify which complex this molecule should be deleted from
	  if(new_C>max_coeff[mol]) target = all_complex[complex_molecule[mol]-1];
	  else target = cplx;
	  
	  /*** Debug ***
	       if(comm->me==0 && screen)
	       {
	       for(int kkk1=0; kkk1<ncomplex; kkk1++)
	       {
	       fprintf(screen, "Found %d state(s) for center %d\n",all_complex[kkk1]->nstate, kkk1+1);
	       
	       fprintf(screen,"    state parent  shell  mol_A  mol_B  react   path    nextra\n");
	       for(int kkk2=0; kkk2<all_complex[kkk1]->nstate; kkk2++)
	       fprintf(screen,"   %6d %6d %6d %6d %6d %6d %6d %6d\n",kkk2,all_complex[kkk1]->parent_id[kkk2],
	       all_complex[kkk1]->shell[kkk2],all_complex[kkk1]->molecule_A[kkk2],all_complex[kkk1]->molecule_B[kkk2],
	       all_complex[kkk1]->reaction[kkk2],all_complex[kkk1]->path[kkk2],all_complex[kkk1]->extra_coupling[kkk2]);
	       }
	       }
	       /*************/ 
	  
	  // Delete states from target complex
	  for(int m=0; m<target->nstate; m++) if(target->molecule_B[m]==mol) {
	      target->delete_state(m--);
	      //fprintf(screen,"Delete overlap molecule %d (state %d) from complex %d \n",mol,m+1,target->id);
	    }
	  
	  // If delete it from original one, update its complex_index
	  if(target!=cplx) {
	    complex_molecule[mol] = i+1;
	    max_coeff[mol] = new_C;
	  }
	  else n--;
	} // if(complex_molecule[mol]==0)
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
  
  /*** Debug ***
       if(comm->me==0 && screen) {
       for(int kkk1=0; kkk1<ncomplex; kkk1++) {
       fprintf(screen, "Found %d state(s) for center %d\n",all_complex[kkk1]->nstate, kkk1+1);
       
       fprintf(screen,"    state parent  shell  mol_A  mol_B  react   path    nextra\n");
       for(int kkk2=0; kkk2<all_complex[kkk1]->nstate; kkk2++)
       fprintf(screen,"   %6d %6d %6d %6d %6d %6d %6d %6d\n",kkk2,all_complex[kkk1]->parent_id[kkk2],
       all_complex[kkk1]->shell[kkk2],all_complex[kkk1]->molecule_A[kkk2],all_complex[kkk1]->molecule_B[kkk2],
       all_complex[kkk1]->reaction[kkk2],all_complex[kkk1]->path[kkk2],all_complex[kkk1]->extra_coupling[kkk2]);
       }
       }
       /*************/
  
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
  
  // Compute qsum and qsqsum
  for(int i=0; i<ncomplex; i++) all_complex[i]->compute_qsqsum();
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::sci_initialize(int vflag)
{
  TIMER_STAMP(EVB_Engine,sci_initialize);

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
  evb_list->change_list(ENV_LIST);
  compute_diagonal(vflag);
  if(EFIELD_flag) compute_efield(0,0);
  
  if(evb_kspace) evb_kspace->sci_compute_env(vflag);
  
  Force_Reduce();
  
  evb_repulsive = NULL;
  full_matrix->save_ev_diag(MATRIX_ENV,vflag);
  MPI_Allreduce(full_matrix->e_env,full_matrix->sci_e_env,EDIAG_NITEM,MPI_DOUBLE,MPI_SUM,world);
  evb_list->change_list(SYS_LIST);
  
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

  // Loop all EVB_Complex
  for(int i=0; i<ncomplex; i++) {
    evb_complex = all_complex[i];
    evb_complex->cec->clear();
    all_matrix[i]->setup();
    all_matrix[i]->clear(true,vflag,true);
    
    full_matrix->clear(true,vflag,true,false); // Is this the cleanest way to keep full_matrix->f_env intact?
    
    if(evb_kspace) evb_kspace->sci_setup_init();
    
    int iextra_coupling = 0;
    
    // Build pair and bond list for complex i
    evb_list->multi_split();
    evb_list->change_list(EVB_LIST);
    evb_complex->update_list();
    
    // Compute the pivot state
    if(natom>0) atom->f = full_matrix->f_diagonal[MATRIX_PIVOT_STATE];
    compute_diagonal(vflag);
    if(EFIELD_flag) compute_efield(1,i+1);
    if(evb_kspace) evb_kspace->compute_cplx(vflag); 
    
    compute_repulsive(vflag);
    
    full_matrix->save_ev_diag(MATRIX_PIVOT_STATE,vflag);
    evb_complex->save_avec(0);
    evb_complex->cec->compute_coc();
    
    Force_Reduce();
    
    // Compute other states
    
    for(int j=1;j<evb_complex->nstate; j++) {
      // Diagonal-element
      
      evb_complex->build_state(j);
      if(natom>0) atom->f = full_matrix->f_diagonal[j];
      compute_diagonal(true);
      if(EFIELD_flag) compute_efield(1,i+1);
      if(evb_kspace) evb_kspace->compute_cplx(vflag);
      compute_repulsive(true);
      full_matrix->save_ev_diag(j,vflag);
      
      Force_Reduce();
      
      evb_offdiag = all_offdiag[evb_complex->reaction[j]-1];
      
      // Extra-coupling-elements
      
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
      
      if(natom>0) atom->f = full_matrix->f_off_diagonal[j-1];

      SETUP_OFFDIAG_EXCH(off,j-1);
      evb_offdiag->sci_setup(vflag);

      full_matrix->save_ev_offdiag(false,j-1,vflag);
      
      Force_Reduce();
      
      // Save energy and status
      evb_complex->save_avec(j);
      
      // Comute COC
      evb_complex->cec->compute_coc();
    }
    
    // Save background force and energy;
    full_matrix->total_energy();
    
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
  }
  
  /**************************************************************/
  ncircle = nstep = 0;

  TIMER_CLICK(EVB_Engine,sci_initialize);
}

/* ----------------------------------------------------------------------*/
// This procedure is used to calculated SCI energy and force, that is, the
// interaction between each complex pair. In each iteration loop, all complex
// will be looped, and when calculating a specific complex (id), it means
// calculating interaction between atoms of complex==id and (complex!=0 &&
// complex!=id). All procedures named ***_sci mean it.

void EVB_Engine::sci_iteration(int vflag)
{
  TIMER_STAMP(EVB_Engine,sci_iteration);

  engine_indicator = ENGINE_INDICATOR_ITERATION;
  
  evb_list->sci_split_inter();
  evb_list->change_list(CPL_LIST);
  evb_effpair->pre_compute();
  
  int nConverge = 0;
  int i = 0;
  
  double **lmp_f = atom->f;
  atom->f = full_matrix->f_env;
  
  int is_Vij_ex;
  
  while(true) {
    if(i==ncomplex) {
      i = 0;
      ncircle++;
      
      /*------------------------------------------*/
#ifdef __DEBUG
      fprintf(screen,"CIRCLE %d\n",ncircle);
      for(int ii=0; ii<ncomplex; ii++)
	fprintf(screen,"%3d: E=%lf\tdE=%lf\n",ii+1,all_matrix[ii]->E,all_matrix[ii]->dE);
      fprintf(screen,"\n");
#endif
      /*------------------------------------------*/
    }
    
    /*************************************************/
    /*************************************************/
    
    evb_complex = all_complex[i];
	  
    if(evb_kspace) evb_kspace->sci_setup_iteration();
    
    // Diagonal elements
    for(int j=0; j<evb_complex->nstate; j++) {
      evb_complex->load_avec(j);
      evb_effpair->compute_cplx(evb_complex->id);
      if(flag_EFFPAIR_SUPP) evb_effpair->compute_cplx_supp(evb_complex->id);
      if(evb_kspace) evb_kspace->sci_compute_cplx(vflag);
      all_matrix[i]->sci_save_ev_diag(j,vflag);
    }
    
    // Off-diagonal elements
    for(int j=0; j<evb_complex->nstate-1; j++) {
      is_Vij_ex = all_offdiag[evb_complex->reaction[j+1]-1]->is_Vij_ex;

      evb_effpair->ecoul = 0.0;
      if(evb_kspace) evb_kspace->off_diag_energy = 0.0;

      if(is_Vij_ex > 0) {
	evb_effpair->init_exch(false,j);
	evb_effpair->compute_exch(evb_complex->id);
	if(flag_EFFPAIR_SUPP) evb_effpair->compute_exch_supp(evb_complex->id);
	if(evb_kspace && is_Vij_ex == 1) evb_kspace->sci_compute_exch(vflag);
      }

      all_matrix[i]->sci_save_ev_offdiag(false,j,vflag);
    }
    
    // Extra-couplings
    for(int j=0; j<evb_complex->nextra_coupling; j++) {
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
    
    // total energy to full_matrix to get diagonalize
    
    all_matrix[i]->sci_total_energy();
    
    for(int j=0; j<evb_complex->nstate; j++) {  
      full_matrix->e_diagonal[j][EDIAG_POT] = all_matrix[i]->e_diagonal[j][EDIAG_POT]
	+all_matrix[i]->sci_e_diagonal[j][SCI_EDIAG_POT];
    }
    
    // e_offdiag     == VIJ_CONST * A_GEO
    // sci_e_offdiag == V_EX      * A_GEO
    for(int j=0; j<evb_complex->nstate-1; j++) { 
      full_matrix->e_offdiag[j][EOFF_ENE] =
	all_matrix[i]->e_offdiag[j][EOFF_ENE]+all_matrix[i]->sci_e_offdiag[j];
    }
    
    for(int j=0; j<evb_complex->nextra_coupling; j++) {
      full_matrix->e_extra[j][EOFF_ENE] = 
	all_matrix[i]->e_extra[j][EOFF_ENE]+all_matrix[i]->sci_e_extra[j];
    }
    
    full_matrix->diagonalize();
    
    all_matrix[i]->ground_state = full_matrix->ground_state;
    all_matrix[i]->pivot_state = full_matrix->pivot_state;
    all_matrix[i]->E = full_matrix->ground_state_energy;
    all_matrix[i]->dE = all_matrix[i]->E - all_matrix[i]->ground_state_energy;
    all_matrix[i]->ground_state_energy = all_matrix[i]->E;

    if(flag_DIAG_QEFF) evb_effpair->compute_para_qeff();
    else evb_effpair->compute_para();

    evb_complex->load_avec(0);
    
    /*************************************************/
    /*************************************************/
    
    nstep++;
    if(ncircle==MAX_CIRCLE) break;
    
    if(fabs(all_matrix[i]->dE)>CNVG_LIMIT) nConverge=0;
    else {
      nConverge++;
      if(nConverge==ncomplex && ncircle>0) {
	/*------------------------------------------*/
#ifdef __DEBUG
	fprintf(screen,"CIRCLE %d\n",ncircle+1);
	for(int ii=0; ii<ncomplex; ii++)
	  fprintf(screen,"%3d: E=%lf\tdE=%lf\n",ii+1,all_matrix[ii]->E,all_matrix[ii]->dE);
	fprintf(screen,"\n");
#endif
	/*------------------------------------------*/
        
	break;
      }
      
    }
    
    i++;
  }

  TIMER_CLICK(EVB_Engine,sci_iteration);
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::sci_finalize(int vflag)
{  
  TIMER_STAMP(EVB_Engine,sci_finalize);

  engine_indicator = ENGINE_INDICATOR_FINALIZE;

  atom->f = full_matrix->f_env;
  
  /************************************************************/
  /******** Energy from each complex **************************/
  /************************************************************/
  
  cplx_energy = 0.0;
  inter_energy = 0.0;
  env_energy = full_matrix->sci_e_env[EDIAG_POT];
  
  for(int i=0; i<ncomplex; i++) {
    cplx_energy += all_matrix[i]->E;
    
    // Interaction energy between each pair of complexes
    
    EVB_Complex* cplx = all_complex[i];
    EVB_MatrixSCI* mtx = all_matrix[i];
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
  
  energy  = cplx_energy - inter_energy + env_energy;

  /************************************************************/
  /******** Hellmann-feynman for each complex *****************/
  /************************************************************/
  
  for (int i=0; i<ncomplex; i++) all_matrix[i]->compute_hellmann_feynman();
  
  /************************************************************/
  /******** Geometry force for off-diagonals  *****************/
  /************************************************************/
  
  for (int i=0; i<ncomplex; i++) {
    evb_complex = all_complex[i];
    int iextra = 0;
    
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
   	  evb_offdiag->Vij = all_matrix[i]->e_extra[iextra][EOFF_VIJ];
   	  double arq = all_matrix[i]->e_extra[iextra][EOFF_ARQ];
   	  if(arq>0.0) evb_offdiag->Vij+= all_matrix[i]->sci_e_extra[iextra]/arq; // prevent division by zero
  	  
	  evb_offdiag->sci_compute(vflag);
   	  iextra++;
   	}
  	
   	evb_complex->molecule_A[j] = save_mol_A;
      }
      
      // Off-diagonal-element
      
      evb_offdiag->Vij = all_matrix[i]->e_offdiag[j-1][EOFF_VIJ];
      double arq = all_matrix[i]->e_offdiag[j-1][EOFF_ARQ];
      if(arq>0.0) evb_offdiag->Vij+= all_matrix[i]->sci_e_offdiag[j-1]/arq; // prevent division by zero

      evb_offdiag->sci_compute(vflag);
    }
    
    evb_complex->load_avec(0);
    evb_complex->update_mol_map();
  }
  
  /************************************************************/
  /******** Effective short-range force ***********************/
  /************************************************************/
  
  // cplx-cplx forces for interactions where only energy was computed during iterations
  evb_list->change_list(CPL_LIST);
  evb_effpair->compute_finter(vflag);
  if(flag_EFFPAIR_SUPP) evb_effpair->compute_finter_supp(vflag);
  
  // cplx-env forces on env atoms using converged eigenvectors 
  evb_list->change_list(SYS_LIST);
  evb_effpair->compute_fenv(vflag);
  if(flag_EFFPAIR_SUPP) evb_effpair->compute_fenv_supp(vflag);
  
  /************************************************************/
  /******** Effective kspace force ****************************/
  /************************************************************/
  
  double *q_save = atom->q;
  atom->q = evb_effpair->q;
  if(evb_kspace) {
    if(SCI_KSPACE_flag == KSPACE_DEFAULT)  evb_kspace->sci_compute_eff(vflag);        // HF forces for both Ewald
    else if(SCI_KSPACE_flag == PPPM_HF_FORCES) {
      evb_kspace->sci_compute_eff(vflag);                                             // HF forces for PPPM (ENV due to CPLX)
      evb_kspace->sci_compute_eff_cplx(vflag);                                        // HF forces for PPPM (CPLX due to CPLX)
    } else if(SCI_KSPACE_flag == PPPM_ACC_FORCES) evb_kspace->sci_compute_eff(vflag); // ACC forces for PPPM
    else if(SCI_KSPACE_flag == PPPM_POLAR_FORCES) sci_pppm_polar();                   // ACC + Polar forces for PPPM
  }
  Force_Reduce();
  atom->q = q_save;
  
  /************************************************************/
  /****************** End of Force ****************************/
  /************************************************************/
  
  nreact = 0;
  
  for(int i=0; i<ncomplex; i++) {
    evb_complex = all_complex[i];
    int pivot = all_matrix[i]->pivot_state; 
    evb_complex->load_avec(pivot);
    
    // Full calculation of effective charges if using real-space for off-diagonals.
    if(flag_DIAG_QEFF) evb_effpair->compute_q_eff(true,true);
    
    evb_complex->update_mol_map();
    if(pivot!=0) {
      nreact++;

      // Ensure order of complexes doesn't change on next step
      rc_molecule_prev[i] = evb_complex->molecule_B[pivot];
    }
  }
  
  // Reset force pointer.
  atom->f = lmp_f;
  memcpy(&(atom->f[0][0]), &(full_matrix->f_env[0][0]), sizeof(double)*3*natom);
  
  TIMER_CLICK(EVB_Engine,sci_finalize);
}

void EVB_Engine::sci_pppm_polar()
{
  error->all(FLERR,"Not yet coded");
}

