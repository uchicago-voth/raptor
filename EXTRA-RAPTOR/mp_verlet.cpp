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
   Package: MULTIPRO
   Purpose: Improve the parallel effeciency of PPPM in MD simulations
   Authors: Yuxing Peng and Chris Knight 
            Voth Group, Department of Chemistry, University of Chicago
            
   Note: This file is Modified from verlet.cpp by Yuxing
------------------------------------------------------------------------- */

#ifdef RAPTOR_MPVERLET

#include "string.h"
#include "mp_verlet.h"
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

using namespace LAMMPS_NS;



/* ---------------------------------------------------------------------- */

MP_Verlet::MP_Verlet(LAMMPS *lmp, int narg, char **arg) :
  Integrate(lmp, narg, arg) 
{

}

MP_Verlet::~MP_Verlet()
{
  delete [] qsize;
  delete [] xsize;
  delete [] qdisp;
  delete [] xdisp;
  if(f_kspace) memory->destroy(f_kspace);
}

/* ----------------------------------------------------------------------
   initialization before run
------------------------------------------------------------------------- */

void MP_Verlet::init()
{
  /* -------------------------------------------------------------
    The following part is added by Yuxing to archive the MP apporach.
    Its basic function is doing the splitting the blocks.
    Here, the idea "block" means a communication group of serveral 
    real space ranks with one k-space rank
  ------------------------------------------------------------- */

  if(force->kspace==NULL) error->universe_all(FLERR,"[MPVERLET] No k-space calculation is in the simulation.");
  if(universe->iworld==0) is_master = 1; else is_master = 0;
  if(comm->me==0 and screen) fprintf(screen,"[MPVERLET] Is the master partition = %d\n",is_master);
  
  int nprocs_master = universe->procs_per_world[0];
  int nprocs_kspace = universe->procs_per_world[1];
  int ratio         = nprocs_master/nprocs_kspace;
  nprocs_per_block  = ratio+1;
  max_nlocal        = 0;
  f_kspace          = NULL;
  
  qsize             = new int[nprocs_per_block];
  xsize             = new int[nprocs_per_block];
  qdisp             = new int[nprocs_per_block];
  xdisp             = new int[nprocs_per_block];

  if(universe->uscreen && universe->me==0)
  {    
    fprintf(universe->uscreen,
      "[MPVERLET] Partition -> r-space:k-space = %d:%d = %d:1\n",
      nprocs_master,nprocs_kspace, ratio);
  }
  
  int kspace_procgrid[3],key;
  if(universe->me == universe->root_proc[1]) 
    memcpy(kspace_procgrid, comm->procgrid,sizeof(int)*3);
  MPI_Bcast(kspace_procgrid, 3 ,MPI_INT, universe->root_proc[1], universe->uworld);
  
  if(is_master)
  {
    iblock = comm->myloc[0]/(comm->procgrid[0]/kspace_procgrid[0]);
    iblock = iblock * kspace_procgrid[1] + comm->myloc[1]/(comm->procgrid[1]/kspace_procgrid[1]);
    iblock = iblock * kspace_procgrid[2] + comm->myloc[2]/(comm->procgrid[2]/kspace_procgrid[2]);
    key = 1;
  }
  else
  {
    iblock = comm->myloc[0];
    iblock = iblock * kspace_procgrid[1] + comm->myloc[1];
    iblock = iblock * kspace_procgrid[2] + comm->myloc[2];
    key = 0;
  }

  MPI_Comm_split(universe->uworld, iblock, key, &block);
  MPI_Comm_rank(block, &(rank_block));
  
  /* -------------------------------------------------------------
     This part is for outputing the MPI_rank map, which can be used
     to debugging and optimizing the code. It can be commented with
     no effect to the simulations
  ------------------------------------------------------------- */
  
  int *grid_map = new int [universe->nprocs*3];
  int *grid_map_all = new int [universe->nprocs*3];
  int *block_map = new int [universe->nprocs];
  int *block_map_all = new int [universe->nprocs];
  int *block_ranks = new int [universe->nprocs];
  int *block_ranks_all = new int [universe->nprocs];
  
  memset(grid_map,0,sizeof(int)*universe->nprocs*3);
  memset(grid_map_all,0,sizeof(int)*universe->nprocs*3);
  memset(block_map,0,sizeof(int)*universe->nprocs);
  memset(block_map_all,0,sizeof(int)*universe->nprocs);
  memset(block_ranks,0,sizeof(int)*universe->nprocs);
  memset(block_ranks_all,0,sizeof(int)*universe->nprocs);
  
  memcpy(&(grid_map[universe->me*3]), comm->myloc, sizeof(int)*3);
  MPI_Reduce(grid_map, grid_map_all, universe->nprocs*3, MPI_INT, MPI_SUM, 0, universe->uworld);
  block_map[universe->me] = iblock;
  MPI_Reduce(block_map, block_map_all, universe->nprocs, MPI_INT, MPI_SUM, 0, universe->uworld);
  block_ranks[universe->me]=rank_block;
  MPI_Reduce(block_ranks, block_ranks_all, universe->nprocs, MPI_INT, MPI_SUM, 0, universe->uworld);
  
  if(universe->me==0)
  {
    FILE* fp = fopen("rank.info","w");
    
    fprintf(fp,"============================================================\n");
    fprintf(fp,"MPI_rank   Partition   Coordinate(x,y,z)   Block:BRank\n");
    fprintf(fp,"------------------------------------------------------------\n");
    
    for(int i=0; i<nprocs_master; i++) 
      fprintf(fp,"    %4d    R-SPACE     (%3d,%3d,%3d)      %4d:%d\n",
        i, grid_map_all[i*3],grid_map_all[i*3+1],grid_map_all[i*3+2], block_map_all[i],block_ranks_all[i]);
    
    for(int i=nprocs_master; i<universe->nprocs; i++) 
      fprintf(fp,"    %4d    K-SPACE     (%3d,%3d,%3d)      %4d:%d\n",
        i, grid_map_all[i*3],grid_map_all[i*3+1],grid_map_all[i*3+2], block_map_all[i],block_ranks_all[i]);   
    
    fprintf(fp,"============================================================\n");
    fprintf(fp,"Block :    R-Space ranks   <--->   K-space rank\n");
    fprintf(fp,"------------------------------------------------------------\n");
    
    for(int i=0; i<nprocs_kspace; i++)
    {
      fprintf(fp, " %4d : ", i);
      
      int k=0;
      for(int j=0; j<universe->nprocs; j++) if(block_map_all[j]==i)
      {
        fprintf(fp, " %4d", j);
        k++;
        
        if(k==nprocs_per_block-1) fprintf(fp, "    <--->  ");
        else if(k==nprocs_per_block) { fprintf(fp, "\n"); break; }
      }
    }
    
    fprintf(fp,"============================================================\n");
    
    fclose(fp);
  }
  
  delete [] grid_map; delete [] grid_map_all;
  delete [] block_map; delete [] block_map_all;
  delete [] block_ranks; delete [] block_ranks_all;
  
  /* -------------------------------------------------------------
     Interface to fix_evb module
  ------------------------------------------------------------- */
  
  fix_evb = NULL;
  
  for(int i=0; i<modify->nfix; i++)
    if(strcmp(modify->fix[i]->style, "evb")==0)
    {
      fix_evb = (FixEVB*)(modify->fix[i]);
      fix_evb->Engine->mp_verlet = this;
      break;
    }
  
  /* -------------------------------------------------------------
     The modification part is ended here, all the following content
     is belonging to the original LAMMPS code
  ------------------------------------------------------------- */
    
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

  // set flags for what arrays to clear in force_clear()
  // need to clear torques if array exists

  torqueflag = 0;
  if (atom->torque_flag) torqueflag = 1;

  // orthogonal vs triclinic simulation box

  triclinic = domain->triclinic;
}

/* ----------------------------------------------------------------------
   communication
------------------------------------------------------------------------- */

void MP_Verlet::comm_init()
{
  nlocal = nlocal_block = xsize[0] = qsize[0] = qdisp[0] = xdisp[0] = 0;
  
  if(is_master) nlocal = atom->nlocal;
  MPI_Gather(&nlocal, 1, MPI_INT, qsize, 1, MPI_INT, 0, block);

  if(atom->nlocal>max_nlocal)
  {
    f_kspace = memory->grow(f_kspace, atom->nlocal, 3, "MP_Verlet:f_kspace");
    max_nlocal = atom->nlocal;
  }
  
  if(is_master==0)
  { 
    for(int i=1; i<nprocs_per_block; i++)
    {
      qdisp[i] = qdisp[i-1]+qsize[i-1];
      xsize[i] = qsize[i]*3;
      xdisp[i] = xdisp[i-1]+xsize[i-1];
    } 
    
    int nglobal = 0;
    for(int i=1; i<nprocs_per_block; i++) nlocal_block += qsize[i];
    MPI_Reduce(&nlocal_block, &nglobal, 1, MPI_INT, MPI_SUM, 0, world);

    if(comm->me==0 && nglobal!=atom->natoms)
    {
      char err_msg[100];
      sprintf(err_msg,
        "[MP_Verlet] Number of atoms is not consistent.(natom=%d but nglobal=%d)",
        (int)(atom->natoms),nglobal);
      error->one(FLERR,err_msg);
    }
    
    while(atom->nmax<=nlocal_block) atom->avec->grow(0);
    atom->nlocal = nlocal_block;
    atom->nghost = 0;
  }
  
  MPI_Gatherv(atom->q, nlocal, MPI_DOUBLE, atom->q, qsize, qdisp, MPI_DOUBLE, 0, block);
  
  if(fix_evb)
  {
    MPI_Gatherv(atom->tag,          nlocal, MPI_INT, atom->tag,          qsize, qdisp, MPI_INT, 0, block);
    MPI_Gatherv(atom->molecule,     nlocal, MPI_INT, atom->molecule,     qsize, qdisp, MPI_INT, 0, block);
    MPI_Gatherv(fix_evb->mol_type,  nlocal, MPI_INT, fix_evb->mol_type,  qsize, qdisp, MPI_INT, 0, block);
    MPI_Gatherv(fix_evb->mol_index, nlocal, MPI_INT, fix_evb->mol_index, qsize, qdisp, MPI_INT, 0, block);
  }
}

void MP_Verlet::comm_box()
{
  MPI_Status mpi_status;
  
  if(rank_block==1) 
  {
    MPI_Send(domain->boxlo, 3, MPI_DOUBLE, 0, 0, block);
    MPI_Send(domain->boxhi, 3, MPI_DOUBLE, 0, 0, block);
  }
  else if(!rank_block) 
  {
    MPI_Recv(domain->boxlo, 3, MPI_DOUBLE, 1, 0, block, &(mpi_status));
    MPI_Recv(domain->boxhi, 3, MPI_DOUBLE, 1, 0, block, &(mpi_status));
    
    domain->set_global_box();
    domain->set_local_box();
  }
}

void MP_Verlet::comm_forward()
{  
  MPI_Gatherv(atom->x[0], nlocal*3, MPI_DOUBLE, atom->x[0], xsize, xdisp, MPI_DOUBLE, 0, block);
  
  if(domain->box_change)
  {
    comm_box();
    if(is_master==0) force->kspace->setup();
  }
}

void MP_Verlet::comm_backward()
{
  MPI_Bcast(&(force->kspace->energy),1, MPI_DOUBLE, 0, block);
  MPI_Bcast(&(force->kspace->virial),6, MPI_DOUBLE, 0, block);
  MPI_Scatterv(atom->f[0], xsize, xdisp, MPI_DOUBLE, f_kspace[0], nlocal*3, MPI_DOUBLE, 0, block);

  //printf("energy %lf %d\n", force->kspace->energy, universe->me);
  
  if(is_master)
  { 
    for(int i=0; i<nlocal; i++)
    {
      atom->f[i][0] += f_kspace[i][0];
      atom->f[i][1] += f_kspace[i][1];
      atom->f[i][2] += f_kspace[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
   setup before run
------------------------------------------------------------------------- */

void MP_Verlet::setup()
{
  if (comm->me == 0 && screen) fprintf(screen,"Setting up run ...\n");
  
  if(is_master==0) for(int i=0; i<atom->nlocal; i++)
    atom->num_bond[i]=atom->num_angle[i]=atom->num_dihedral[i]=atom->num_improper[i]-=0;
    
  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  atom->setup();
  if (triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  if (neighbor->style) neighbor->setup_bins();
  comm->exchange();
  if (atom->sortfreq > 0) atom->sort();
  comm->borders();
  if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  
  if(is_master)
  {
    neighbor->build();
    neighbor->ncalls = 0;
  }
  
  // compute all forces

  ev_set(update->ntimestep);
  force_clear();

  /********************************************************************/
  comm_init();
  comm_forward();
  /********************************************************************/
  
  if(is_master) 
  {
    if (force->pair) force->pair->compute(eflag,vflag);

    if (atom->molecular) {
      if (force->bond) force->bond->compute(eflag,vflag);
      if (force->angle) force->angle->compute(eflag,vflag);
      if (force->dihedral) force->dihedral->compute(eflag,vflag);
      if (force->improper) force->improper->compute(eflag,vflag);
    }
    
    if (force->kspace) force->kspace->setup();
  }
  /********************************************************************/  
  else
  {
    if (force->kspace) {
      force->kspace->setup();
      force->kspace->compute(eflag,vflag);
    }
  }

  if(is_master)
  {
    if (force->newton) comm->reverse_comm();
    modify->setup(vflag);
  }
  else if(fix_evb) fix_evb->setup(vflag);
  
  /********************************************************************/
  comm_backward();
  /********************************************************************/
  
  if(is_master) output->setup(1);
}

/* ----------------------------------------------------------------------
   setup without output
   flag = 0 = just force calculation
   flag = 1 = reneighbor and force calculation
------------------------------------------------------------------------- */

void MP_Verlet::setup_minimal(int flag)
{
  // setup domain, communication and neighboring
  // acquire ghosts
  // build neighbor lists

  if (flag) {
    if (triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    if (triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    neighbor->build();
    neighbor->ncalls = 0;
  }

  // compute all forces

  ev_set(update->ntimestep);
  force_clear();
  
  /********************************************************************/
  comm_init();
  comm_forward();
  /********************************************************************/
  if(is_master) 
  {
    if (force->pair) force->pair->compute(eflag,vflag);

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
      force->kspace->compute(eflag,vflag);
    }
  }
  
  if(is_master) 
  {
    modify->setup(vflag);
    if (force->newton) comm->reverse_comm();
  }
  else if(fix_evb) fix_evb->setup(vflag);
  
  /********************************************************************/
  comm_backward();
  /********************************************************************/
}

/* ----------------------------------------------------------------------
   run for N steps
------------------------------------------------------------------------- */

void MP_Verlet::run(int n)
{
  int nflag,ntimestep,sortflag;

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

/*****************************************************************/
/*****************************************************************/
/*****************************************************************/

  // initial time integration
 
  if(is_master)
  {
    modify->initial_integrate(vflag);
    if (n_post_integrate) modify->post_integrate();
  }
  
  // regular communication vs neighbor list rebuild

  if(is_master) nflag = neighbor->decide(); else nflag = 0;
  MPI_Bcast(&nflag, 1, MPI_INT, 1, block);
  
  if(is_master)
  {
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
  }
  else
  {
    if(nflag)
    {
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
      neighbor->ago=0;
    }
    else neighbor->ago++;
  }
  
  if(nflag) comm_init();
  comm_forward();
  
/*****************************************************************/
/*****************************************************************/
/*****************************************************************/
    
  // force computations

  force_clear();

/*****************************************************************/
/*****************************************************************/
/*****************************************************************/

  if(is_master)
  {
    if (n_pre_force) modify->pre_force(vflag);
  }
  else if(fix_evb) fix_evb->pre_force(vflag);
  
  /*****************************************/

  timer->stamp();

  if(!fix_evb) {
  
    if(is_master)
    {
      if (force->pair) {
        force->pair->compute(eflag,vflag);
        timer->stamp(Timer::PAIR);
      }
   
      if (atom->molecular) {
        if (force->bond) force->bond->compute(eflag,vflag);
        if (force->angle) force->angle->compute(eflag,vflag);
        if (force->dihedral) force->dihedral->compute(eflag,vflag);
        if (force->improper) force->improper->compute(eflag,vflag);
        timer->stamp(Timer::BOND);
      }
    }
    /***********************************************************/
    else
    {
      if (force->kspace) {
        force->kspace->compute(eflag,vflag);
        timer->stamp(Timer::KSPACE);
      }
    }
  
  }
  
  /*****************************************/

  if(is_master)
  {    
    // reverse communication of forces
      
    if (force->newton) {
      comm->reverse_comm();
      timer->stamp(Timer::COMM);
    }

    // force modifications, final time integration, diagnostics

    if (n_post_force) modify->post_force(vflag);
  }
  else if(fix_evb) fix_evb->post_force(vflag);
  

  /************************************************************/
  comm_backward();
  /************************************************************/
  
  if(is_master)
  {
    modify->final_integrate();
    if (n_end_of_step) modify->end_of_step();
  }
  
/*****************************************************************/
/*****************************************************************/
/*****************************************************************/  
  
    if (ntimestep == output->next) {
      timer->stamp();
      if(is_master) output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }

/*****************************************************************/
/*****************************************************************/
/*****************************************************************/
    
  } // end of running-N-steps loop

/*****************************************************************/
/*****************************************************************/
/*****************************************************************/

}

/* ----------------------------------------------------------------------
   clear force on own & ghost atoms
   setup and clear other arrays as needed
------------------------------------------------------------------------- */

void MP_Verlet::force_clear()
{
  int i;

  // clear force on all particles
  // if either newton flag is set, also include ghosts

  if (neighbor->includegroup == 0) {
    int nall;
    if (force->newton) nall = atom->nlocal + atom->nghost;
    else nall = atom->nlocal;

    double **f = atom->f;
    for (i = 0; i < nall; i++) {
      f[i][0] = 0.0;
      f[i][1] = 0.0;
      f[i][2] = 0.0;
    }
    
    if (torqueflag) {
      double **torque = atom->torque;
      for (i = 0; i < nall; i++) {
	torque[i][0] = 0.0;
	torque[i][1] = 0.0;
	torque[i][2] = 0.0;
      }
    }

  // neighbor includegroup flag is set
  // clear force only on initial nfirst particles
  // if either newton flag is set, also include ghosts

  } else {
    int nall = atom->nfirst;

    double **f = atom->f;
    for (i = 0; i < nall; i++) {
      f[i][0] = 0.0;
      f[i][1] = 0.0;
      f[i][2] = 0.0;
    }
    
    if (torqueflag) {
      double **torque = atom->torque;
      for (i = 0; i < nall; i++) {
	torque[i][0] = 0.0;
	torque[i][1] = 0.0;
	torque[i][2] = 0.0;
      }
    }

    if (force->newton) {
      nall = atom->nlocal + atom->nghost;

      for (i = atom->nlocal; i < nall; i++) {
	f[i][0] = 0.0;
	f[i][1] = 0.0;
	f[i][2] = 0.0;
      }
    
      if (torqueflag) {
	double **torque = atom->torque;
	for (i = atom->nlocal; i < nall; i++) {
	  torque[i][0] = 0.0;
	  torque[i][1] = 0.0;
	  torque[i][2] = 0.0;
	}
      }
    }
  }
}

#endif