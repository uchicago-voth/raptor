/*
 * Functions for state decomposition 
 * 
 * AWGL 
 */

#ifdef STATE_DECOMP

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

#include "mp_verlet.h"
#include "pair_evb.h"
#include "EVB_text.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "timer.h"

using namespace LAMMPS_NS;


void EVB_Engine::Divvy_Out_Partitions(int* output)
{
  int jfrom = 0;
  int jto = evb_complex->nstate;
  int my_world = 0;
  if (universe->existflag && flag_mp_state) {

    if (ncomplex > 1) {
      // Crash for multiple complexes. Un-tested so far.
      error->all(FLERR,"Cannot use state partitioning with more than 1 complex for now.");
    }

    int jnum = evb_complex->nstate;
    int jid = universe->iworld;
    my_world = jid;
    if (evb_complex->nstate == 1 && (1 <= flag_mp_state && flag_mp_state <= 4)) {
      // Crash if only one state. Cannot partition.
      //error->all(FLERR, "Cannot run multi state partitioning for only one state!");
      // Make partition 0 do all the work
      if (jid == 0) {
        jfrom = 0;
        jto = 1;
      }
      else {
        jfrom = 1;
        jto = 1;
      } 
    }
    else if (flag_mp_state == 1 || flag_mp_state == 3) {
      int div = jnum / universe->nworlds;
      int rem = jnum % universe->nworlds;
      if (jid < rem) { jfrom = jid*div + jid; jto = jfrom + div + 1; }
      else {           jfrom = jid*div + rem; jto = jfrom + div; }
    }
    /*
    // This code is temporarily phased out because it is slightly off for some reason.
    else if (flag_mp_state == 2 || flag_mp_state == 4) {
      jid = jid - 1;
      int div = jnum / universe->nworlds;
      int rem = jnum % universe->nworlds;
      if (jid >= 0) {
        if (jid < rem) { jfrom = jid*div + jid; jto = jfrom + div + 1; }
        else {           jfrom = jid*div + rem; jto = jfrom + div; }
      } else {
        jfrom = jto = jnum;
      }
    }
    */
    else {
      error->all(FLERR,"Can only choose mp_state 1 or 3 options for now.");
    }
    if (jto   > jnum) jto   = jnum;
    if (jfrom > jnum) jfrom = jnum;
  } else if (flag_mp_state && !universe->existflag) {
    error->all(FLERR,"Cannot turn on multi state partitioning without running LAMMPS with multiple partitions.");
  } else if (flag_mp_state && mp_verlet) {
    error->all(FLERR,"Cannot run multi state partitioning with the multiprogram Verlet integrator yet.");
  }
  output[0] = jfrom; 
  output[1] = jto; 
  output[2] = my_world; 
}

/* -------------------------------------------------------------------------------- */

void EVB_Engine::Communicate_Between_Partitions(int nstate, int nextra_coupling, int vflag)
{
   // ** AWGL ** //
   // When running multiple state partitioning, this function handles the communication
   // of forces and energy between partitions. The result is that every partition has the
   // same energy and forces, even though they all worked on different states.

   // If not doing multiple state partitioning, this function does nothing.

   // ** NOTE: This is currently not working for virial calculations! ** //
      if(universe->existflag && flag_mp_state) {

         timer->stamp(); // track communication time

         EVB_Matrix* mtx = full_matrix; // alias
         int offset = 0;

         // Create a communication buffer, and put stuff in it
         int num = (9*nstate) + (4*(nstate-1)) + (nextra_coupling) + 8;
         double * sbuffer = new double[num];
         double * rbuffer = new double[num];

         offset = 0;
         for(int i=0; i<nstate; i++){ 
           sbuffer[offset++] = mtx->e_diagonal[i][EDIAG_POT];
           sbuffer[offset++] = mtx->e_diagonal[i][EDIAG_VDW];
           sbuffer[offset++] = mtx->e_diagonal[i][EDIAG_COUL];
           sbuffer[offset++] = mtx->e_diagonal[i][EDIAG_BOND];
           sbuffer[offset++] = mtx->e_diagonal[i][EDIAG_ANGLE];
           sbuffer[offset++] = mtx->e_diagonal[i][EDIAG_DIHEDRAL];
           sbuffer[offset++] = mtx->e_diagonal[i][EDIAG_IMPROPER];
           sbuffer[offset++] = mtx->e_diagonal[i][EDIAG_KSPACE];
           sbuffer[offset++] = mtx->e_repulsive[i];
         }
         for(int i=0; i<nstate-1; i++) {
            sbuffer[offset++] = mtx->e_offdiag[i][EOFF_ENE];
           sbuffer[offset++] = mtx->e_offdiag[i][EOFF_ARQ];
           sbuffer[offset++] = mtx->e_offdiag[i][EOFF_VIJ_CONST];
           sbuffer[offset++] = mtx->e_offdiag[i][EOFF_VIJ];
         }
         for(int i=0; i<nextra_coupling; i++)
           sbuffer[offset++] = mtx->e_extra[i][EOFF_ENE];
         // Environment
         sbuffer[offset++] = mtx->e_env[EDIAG_POT];
         sbuffer[offset++] = mtx->e_env[EDIAG_VDW];
         sbuffer[offset++] = mtx->e_env[EDIAG_COUL];
         sbuffer[offset++] = mtx->e_env[EDIAG_BOND];
         sbuffer[offset++] = mtx->e_env[EDIAG_ANGLE];
         sbuffer[offset++] = mtx->e_env[EDIAG_DIHEDRAL];
         sbuffer[offset++] = mtx->e_env[EDIAG_IMPROPER];
         // All partitions computed this, so only use one of them 
         if (universe->iworld == 0) {
           sbuffer[offset++] = mtx->e_env[EDIAG_KSPACE];
         } else {
           sbuffer[offset++] = 0.0;
         }

         MPI_Allreduce(sbuffer, rbuffer, num, MPI_DOUBLE, MPI_SUM, force_comm);

         // Unpack into the appropriate places
         offset = 0;
         for(int i=0; i<nstate; i++){ 
             mtx->e_diagonal[i][EDIAG_POT]      = rbuffer[offset++];
             mtx->e_diagonal[i][EDIAG_VDW]      = rbuffer[offset++];
             mtx->e_diagonal[i][EDIAG_COUL]     = rbuffer[offset++];
             mtx->e_diagonal[i][EDIAG_BOND]     = rbuffer[offset++];
             mtx->e_diagonal[i][EDIAG_ANGLE]    = rbuffer[offset++];
             mtx->e_diagonal[i][EDIAG_DIHEDRAL] = rbuffer[offset++];
             mtx->e_diagonal[i][EDIAG_IMPROPER] = rbuffer[offset++];
             mtx->e_diagonal[i][EDIAG_KSPACE]   = rbuffer[offset++];
             mtx->e_repulsive[i]                = rbuffer[offset++];
         }
         for(int i=0; i<nstate-1; i++) {
             mtx->e_offdiag[i][EOFF_ENE]       = rbuffer[offset++];
             mtx->e_offdiag[i][EOFF_ARQ]       = rbuffer[offset++];
             mtx->e_offdiag[i][EOFF_VIJ_CONST] = rbuffer[offset++];
             mtx->e_offdiag[i][EOFF_VIJ]       = rbuffer[offset++];
         }
         for(int i=0; i<nextra_coupling; i++)
             mtx->e_extra[i][EOFF_ENE] = rbuffer[offset++];

         mtx->e_env[EDIAG_POT]      = rbuffer[offset++];
         mtx->e_env[EDIAG_VDW]      = rbuffer[offset++];
         mtx->e_env[EDIAG_COUL]     = rbuffer[offset++];
         mtx->e_env[EDIAG_BOND]     = rbuffer[offset++];
         mtx->e_env[EDIAG_ANGLE]    = rbuffer[offset++];
         mtx->e_env[EDIAG_DIHEDRAL] = rbuffer[offset++];
         mtx->e_env[EDIAG_IMPROPER] = rbuffer[offset++];
         mtx->e_env[EDIAG_KSPACE]   = rbuffer[offset++];

         delete [] sbuffer;
         delete [] rbuffer;
         
         // **** Communicate forces here if mp_state 1 or 2, otherwise force is communicated later **** //
         if (flag_mp_state == 1 || flag_mp_state == 2) {

           // Buffer set up...
           int force_message_size = 6*natom*nstate + 3*natom*nextra_coupling;
           double* force_sbuff = new double[force_message_size];
           double* force_rbuff = new double[force_message_size];
           memset(force_sbuff, 0.0, sizeof(double)*force_message_size);
           memset(force_rbuff, 0.0, sizeof(double)*force_message_size);

           // ** First, broadcast the env force to my group ** //
           MPI_Bcast(&(mtx->f_env[0][0]), 3*natom, MPI_DOUBLE, group_root, force_comm);

           // ** Pack all rest of my forces into a send buffer ** //
           int i;
#if defined(_OPENMP)
           #pragma omp parallel for default(none)\
            shared(force_sbuff,nstate,nextra_coupling,mtx) private(i) 
#endif
           for(i=0; i<natom; ++i) {
	     for (int istate=0; istate<nstate; ++istate) {
	       // ** f_diagonal force ** //
	       force_sbuff[3*natom*istate + 3*i]   = mtx->f_diagonal[istate][i][0];
	       force_sbuff[3*natom*istate + 3*i+1] = mtx->f_diagonal[istate][i][1];
	       force_sbuff[3*natom*istate + 3*i+2] = mtx->f_diagonal[istate][i][2];
	       // ** f_off_diagonal force ** //
	       if(istate<nstate-1) {
		 force_sbuff[3*natom*nstate + 3*natom*istate + 3*i]   = mtx->f_off_diagonal[istate][i][0];
		 force_sbuff[3*natom*nstate + 3*natom*istate + 3*i+1] = mtx->f_off_diagonal[istate][i][1];
		 force_sbuff[3*natom*nstate + 3*natom*istate + 3*i+2] = mtx->f_off_diagonal[istate][i][2];
	       }
	     }
	     // ** f_extra_coupling force ** //
	     for (int icouple=0; icouple<nextra_coupling; ++icouple) {
	       force_sbuff[6*natom*nstate + 3*natom*icouple + 3*i]   = mtx->f_extra_coupling[icouple][i][0];
	       force_sbuff[6*natom*nstate + 3*natom*icouple + 3*i+1] = mtx->f_extra_coupling[icouple][i][1];
	       force_sbuff[6*natom*nstate + 3*natom*icouple + 3*i+2] = mtx->f_extra_coupling[icouple][i][2];
	     }
           }
	   
           // ** Reduce everybody in the group ** //
           MPI_Allreduce(force_sbuff, force_rbuff, force_message_size, MPI_DOUBLE, MPI_SUM, force_comm);

           // Unpack
           int j;
#if defined(_OPENMP)
           #pragma omp parallel for default(none)\
            shared(force_rbuff,nstate,nextra_coupling,mtx) private(j)
#endif
           for(j=0; j<natom; ++j) {
             for (int istate=0; istate<nstate; ++istate) {
               mtx->f_diagonal[istate][j][0] = force_rbuff[3*natom*istate + 3*j];
               mtx->f_diagonal[istate][j][1] = force_rbuff[3*natom*istate + 3*j+1];
               mtx->f_diagonal[istate][j][2] = force_rbuff[3*natom*istate + 3*j+2];
	       if(istate<nstate-1) {
		 mtx->f_off_diagonal[istate][j][0] = force_rbuff[3*natom*nstate + 3*natom*istate + 3*j];
		 mtx->f_off_diagonal[istate][j][1] = force_rbuff[3*natom*nstate + 3*natom*istate + 3*j+1];
		 mtx->f_off_diagonal[istate][j][2] = force_rbuff[3*natom*nstate + 3*natom*istate + 3*j+2];
	       }
             }
             for (int icouple=0; icouple<nextra_coupling; ++icouple) {
               mtx->f_extra_coupling[icouple][j][0] = force_rbuff[6*natom*nstate + 3*natom*icouple + 3*j];
               mtx->f_extra_coupling[icouple][j][1] = force_rbuff[6*natom*nstate + 3*natom*icouple + 3*j+1];
               mtx->f_extra_coupling[icouple][j][2] = force_rbuff[6*natom*nstate + 3*natom*icouple + 3*j+2];
             }
           }
         

           delete [] force_sbuff;
           delete [] force_rbuff;
    
         } // close if 1 or 2
         else {
           // *** Zero out the environment force on all except the root of the force comm *** //
           // This is so that f_env only gets added once in compute_hellman_feynman
           if (universe->iworld != 0) {
             memset(&(mtx->f_env[0][0]), 0.0, sizeof(double)*3*natom);
           }
         }


         // ** Finally, make sure all partitions have the same coordinates by making root broadcast ** //
         // This is a precaution to avoid partitions drifting apart with slightly different coordinates
         double **x = atom->x;
         MPI_Bcast(&x[0][0], 3*natom, MPI_DOUBLE, group_root, force_comm);
         double **v = atom->v;
         MPI_Bcast(&v[0][0], 3*natom, MPI_DOUBLE, group_root, force_comm);

         timer->stamp(TIME_COMM);
       }

}


/* -------------------------------------------------------------------------------- */

void EVB_Engine::Communicate_Force_Between_Partitions(double **ff)
{
   // ** AWGL ** //
   // When running multiple state partitioning, this function handles the communication
   // of forces only! We assume only the ground state forces are needed. So, other state forces
   // are discarded! The idea here is that compute_hellman_feynman has already summed the forces
   // my state partition knows about. So, all we have to do now is sum up the ground state over partitions. 

   // If not doing multiple state partitioning, this function does nothing.

   // ** NOTE: This is currently not working for virial calculations! ** //
   if(universe->existflag && flag_mp_state) {
     timer->stamp(); // track communication time

     // Buffer set up...
     int force_message_size = 3*natom;
     double* force_rbuff = new double[force_message_size];
     memset(force_rbuff, 0.0, sizeof(double)*force_message_size);

     // ** Reduce everybody in the group ** //
     MPI_Allreduce(&(ff[0][0]), force_rbuff, force_message_size, MPI_DOUBLE, MPI_SUM, force_comm);

     // ** Copy over receive buffer ** //
     memcpy(&(ff[0][0]), force_rbuff, sizeof(double)*force_message_size);

     delete [] force_rbuff;

     timer->stamp(TIME_COMM);
   }

}

#endif
