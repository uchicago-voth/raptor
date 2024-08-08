/* -------------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Authors: Chris Knight 
             Derived from USER-MULTIPRO package
------------------------------------------------------------------------- */

#ifdef RAPTOR_MPVERLET

#include "string.h"
#include "mp_verlet_sci.h"
#include "neighbor.h"
#include "domain.h"
#include "comm.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "output.h"
#include "update.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "timer.h"
#include "memory.h"
#include "error.h"
#include "universe.h"

#include "fix_evb.h"
#include "EVB_engine.h"

#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MP_Verlet_SCI::MP_Verlet_SCI(LAMMPS *lmp, int narg, char **arg) :
  Integrate(lmp, narg, arg) 
{

}

MP_Verlet_SCI::~MP_Verlet_SCI()
{

}

/* ----------------------------------------------------------------------
   initialization before run
------------------------------------------------------------------------- */

void MP_Verlet_SCI::init()
{
  // Partition 0 is the master partition
  if(universe->iworld==0) is_master = 1; else is_master = 0;

  // Partition 1 is the second master partition; default to partition 0 if running 1-partition calculation
  is_master2 = 0;
  if(universe->nworlds > 1 && universe->iworld == 1) is_master2 = 1;
  else if (universe->nworlds == 1) is_master2 = 1;

  // Partition 2 is the third master partition; default to partition 0 if running less than 3-partition calculation
  is_master3 = 0;
  if(universe->nworlds > 2 && universe->iworld == 2) is_master3 = 1;
  else if (universe->nworlds < 3 && universe->iworld == 0) is_master3 = 1;

  int nprocs = universe->procs_per_world[0];
  
  int iblock, key;
  if(is_master) key = 0;
  else key = 1;
  iblock = comm->me;

  MPI_Comm_split(universe->uworld, iblock, key, &block);
  
  // /* -------------------------------------------------------------
  //    Interface to fix_evb module
  // ------------------------------------------------------------- */
  
  fix_evb = NULL;
  
  for(int i=0; i<modify->nfix; i++)
    if(strcmp(modify->fix[i]->style, "evb")==0) {
      fix_evb = (FixEVB*)(modify->fix[i]);
      fix_evb->Engine->mp_verlet_sci = this;
      break;
    }
  if(!fix_evb) error->universe_one(FLERR,"fix_evb not found");

  // /* -------------------------------------------------------------
  //    The modified part ends here. The rest is copied from 
  //    Verlet::init() on Jan. 17, 2013.
  // ------------------------------------------------------------- */
    
  // warn if no fixes
  
  if (modify->nfix == 0 && comm->me == 0)
    error->warning(FLERR,"No fixes defined, atoms won't move");
  
  // virial_style:
  // 1 if computed explicitly by pair->compute via sum over pair interactions
  // 2 if computed implicitly by pair->virial_compute via sum over ghost atoms
  
  if (force->newton_pair) virial_style = 2;
  else virial_style = 1;
  
  // setup lists of computes for global and per-atom PE and pressure
  
  ev_setup();

  // detect if fix omp is present for clearing force arrays

  int ifix = modify->find_fix("package_omp");
  if (ifix >= 0) external_force_clear = 1;
  
  // set flags for what arrays to clear in force_clear()
  // need to clear torques if array exists
  
  torqueflag = extraflag = 0;
  if (atom->torque_flag) torqueflag = 1;
  if (atom->avec->forceclearflag) extraflag = 1;
  
  // orthogonal vs triclinic simulation box
  
  triclinic = domain->triclinic;
}

/* ----------------------------------------------------------------------
   communication
------------------------------------------------------------------------- */

void MP_Verlet_SCI::comm_forward()
{
  int n = atom->nlocal;
  MPI_Bcast(&(atom->x[0][0]), n*3, MPI_DOUBLE, 0, block);
}

/* ----------------------------------------------------------------------
   setup before run; mostly copied from Verlet::setup()
------------------------------------------------------------------------- */

void MP_Verlet_SCI::setup()
{
  if (comm->me == 0 && screen) fprintf(screen,"Setting up run ...\n");

#ifdef BGQ
  if(universe->me==0) {
    fprintf(stdout,"\nget_memory() at start of MP_Verlet_SCI::setup().\n");
    fix_evb->Engine->get_memory();
  }
#endif  

  update->setupflag = 1;
  
  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  atom->setup();
  modify->setup_pre_exchange();
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  comm->exchange();
  if (atom->sortfreq > 0) atom->sort();
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  domain->image_check();
  domain->box_too_small_check();
  modify->setup_pre_neighbor();
  neighbor->build();
  neighbor->ncalls = 0;
  
  // compute all forces

  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);

  if (pair_compute_flag) force->pair->compute(eflag,vflag);
  else if(force->pair) force->pair->compute_dummy(eflag,vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }
    
  if(force->kspace) {
    force->kspace->setup();
    if(kspace_compute_flag) force->kspace->compute(eflag,vflag);
    else force->kspace->compute_dummy(eflag,vflag);
  }
    
  if (force->newton) comm->reverse_comm();

  modify->setup(vflag); 
  if(is_master) output->setup(1);
  update->setupflag = 0;

#ifdef BGQ
  if(universe->me==0) {
    fprintf(stdout,"\nget_memory() at end of MP_Verlet_SCI::setup().\n");
    fix_evb->Engine->get_memory();
  }
#endif  
}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void MP_Verlet_SCI::setup_minimal(int flag)
{
  error->universe_all(FLERR,"MP_Verlet_SCI::setup_minimal() called");

  update->setupflag = 1;

  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  if (flag) {
    modify->setup_pre_exchange();
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    domain->image_check();
    domain->box_too_small_check();
    modify->setup_pre_neighbor();
    neighbor->build();
    neighbor->ncalls = 0;
  }

  // compute all forces

  ev_set(update->ntimestep);
  force_clear();
  modify->setup_pre_force(vflag);
  
  if(is_master) 
  {
    if (pair_compute_flag) force->pair->compute(eflag,vflag);
    else if (force->pair) force->pair->compute_dummy(eflag,vflag);

    if (atom->molecular) {
      if (force->bond) force->bond->compute(eflag,vflag);
      if (force->angle) force->angle->compute(eflag,vflag);
      if (force->dihedral) force->dihedral->compute(eflag,vflag);
      if (force->improper) force->improper->compute(eflag,vflag);
    }
  }
  /********************************************************************/  
  else
  {
    if (force->kspace) {
      force->kspace->setup();
      if(kspace_compute_flag) force->kspace->compute(eflag,vflag);
      else force->kspace->compute_dummy(eflag,vflag);
    }
  }
  
  if(is_master) {
    if (force->newton) comm->reverse_comm();
    modify->setup(vflag);
  } else if(fix_evb) fix_evb->setup(vflag);

  update->setupflag = 0;
}

/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */

void MP_Verlet_SCI::run(int n)
{
  bigint ntimestep;
  int nflag,sortflag;

  int n_post_integrate = modify->n_post_integrate;
  int n_pre_exchange = modify->n_pre_exchange;
  int n_pre_neighbor = modify->n_pre_neighbor;
  int n_pre_force = modify->n_pre_force;
  int n_post_force = modify->n_post_force;
  int n_end_of_step = modify->n_end_of_step;

/*****************************************************************/
/*****************************************************************/
/*****************************************************************/

  if (atom->sortfreq > 0) sortflag = 1;
  else sortflag = 0;

  for (int i = 0; i < n; i++) {
    
    ntimestep = ++update->ntimestep;
    ev_set(ntimestep);
    
    // initial time integration
    
    if(is_master) {
      modify->initial_integrate(vflag);
      if (n_post_integrate) modify->post_integrate();
    }

    // Update local (integrated) coordinates on slave partitions

    comm_forward();

    // regular communication vs neighbor list rebuild

    nflag = neighbor->decide();
    MPI_Bcast(&nflag, 1, MPI_INT, 0, block); // Master partition ensures re-neighboring is synced.

    if (nflag == 0) {
      timer->stamp();
      comm->forward_comm();
      timer->stamp(Timer::COMM);
    } else {
      if (n_pre_exchange) modify->pre_exchange();
      if (triclinic) domain->x2lamda(atom->nlocal);
      domain->pbc();
      if (domain->box_change) {
        domain->reset_box();
        comm->setup();
        if (neighbor->style) neighbor->setup_bins();
      }
      timer->stamp();
      comm->exchange();
      if (sortflag && ntimestep >= atom->nextsort) atom->sort();
      comm->borders();
      if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
      timer->stamp(Timer::COMM);
      if (n_pre_neighbor) modify->pre_neighbor();
      neighbor->build();
      timer->stamp(Timer::NEIGH);
    }

    // force computations
    
    force_clear();

    if (n_pre_force) modify->pre_force(vflag);

    /*****************************************/
    
    timer->stamp();
    
    /*****************************************/
    
    if(is_master) {    
      // reverse communication of forces
      
      if (force->newton) {
	comm->reverse_comm();
	timer->stamp(Timer::COMM);
      }
      
      // force modifications, final time integration, diagnostics
      
      if (n_post_force) modify->post_force(vflag);
    } else if(fix_evb) fix_evb->post_force(vflag);
    
    if(is_master) {
      modify->final_integrate();
      if (n_end_of_step) modify->end_of_step();
    }
    
    if (ntimestep == output->next) {
      timer->stamp();
      if(is_master) output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
    
  } // end of running-N-steps loop
  
}

/* ---------------------------------------------------------------------- */

void MP_Verlet_SCI::cleanup()
{
  modify->post_run();
  domain->box_too_small_check();
  update->update_time();
}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   setup and clear other arrays as needed
------------------------------------------------------------------------- */

void MP_Verlet_SCI::force_clear()
{
  int i;
  size_t nbytes;

  if (external_force_clear) return;

  // clear force on all particles
  // if either newton flag is set, also include ghosts
  // when using threads always clear all forces.

  int nlocal = atom->nlocal;

  if (neighbor->includegroup == 0) {
    nbytes = sizeof(double) * nlocal;
    if (force->newton) nbytes += sizeof(double) * atom->nghost;

    if (nbytes) {
      memset(&(atom->f[0][0]),0,3*nbytes);
      if (torqueflag)  memset(&(atom->torque[0][0]),0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  } else {
    nbytes = sizeof(double) * atom->nfirst;

    if (nbytes) {
      memset(&atom->f[0][0],0,3*nbytes);
      if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
      if (extraflag) atom->avec->force_clear(0,nbytes);
    }

    if (force->newton) {
      nbytes = sizeof(double) * atom->nghost;

      if (nbytes) {
        memset(&atom->f[nlocal][0],0,3*nbytes);
        if (torqueflag) memset(&atom->torque[nlocal][0],0,3*nbytes);
        if (extraflag) atom->avec->force_clear(nlocal,nbytes);
      }
    }
  }
}

#endif