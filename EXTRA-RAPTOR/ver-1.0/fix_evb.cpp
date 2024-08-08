/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define _CRACKER_INTEGRATE
  #include "EVB_cracker.h"
#undef  _CRACKER_INTEGRATE

#include "update.h"

#define _CRACKER_NEIGHBOR
  #include "EVB_cracker.h"
#undef  _CRACKER_NEIGHBOR

#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"

#include "atom.h"
#include "domain.h"
#include "group.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "comm.h"

#include "fix_evb.h"
#include "EVB_engine.h"
#include "EVB_effpair.h"
#include "EVB_type.h"
#include "EVB_source.h"
#include "EVB_timer.h"
#include "EVB_version.h"

#include "math.h"

#ifdef STATE_DECOMP
#include "universe.h"
#endif

#ifdef BGQ
#include "universe.h"
#include <spi/include/kernel/memory.h>
#endif

#define KSPACE_DEFAULT    0 // Hellman-Feynman forces for Ewald
#define PPPM_HF_FORCES    1 // Hellman-Feynman forces for PPPM
#define PPPM_ACC_FORCES   2 // Approximate (acc) forces for PPPM. 
#define PPPM_POLAR_FORCES 3 // ACC forces plus an additional polarization force on complex atoms for PPPM.

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

FixEVB::FixEVB(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{  
  Engine = NULL;
  mol_type = mol_index = NULL;
  charge = NULL;
  array_atom = NULL;
  
  /*** Set up FIX parameters ***/
  virial_flag = 1;

  // Add EVB energy feedback
  thermo_energy = 1; 
  scalar_flag =1;   
  extscalar = 1;    
  global_freq = 1;
  size_array = 0;
  
  // Add EVB topology feedback
  comm_forward = 2;
  peratom_flag = 1;
  peratom_freq = 1;
  restart_peratom = 1;
  size_peratom_cols = 3;
  atom->add_callback(0);
  atom->add_callback(1);
 
  // Add CEC feedback
  // not achieved

  // Add molecule_type storage
  // not achieved
 
  // Add re-neighboring flag
  force_reneighbor=1;
  next_reneighbor=-1;

  // FDM
  FDM_flag=0;
  
  //////////////////////////////////////////////////////////////////////////

  if(narg<6) error->all(FLERR,"[EVB] Wrong number of parameters for [fix_evb].");

  /*** Get command parameters ***/
  if (comm->me == 0) 
  {
    if(screen) fprintf(screen,_EVB_LINE);
    if(screen) fprintf(screen,"[EVB] Construct EVB module ...\n");

    if(logfile) fprintf(logfile,_EVB_LINE);
    if(logfile) fprintf(logfile, "[EVB] Copyright %s   Version %s\n", _EVB_COPYRIGHT,_EVB_VERSION);
    if(logfile) fprintf(logfile,_EVB_LINE);
    if(logfile) fprintf(logfile, "EVB Command: \"fix");  
    if(logfile) for(int i=0; i<narg; i++) fprintf(logfile, " %s", arg[i]);
    if(logfile) fprintf(logfile,"\"\n\n");
  }
  
  /*** Create MS-EVB object ***/
  Engine = new EVB_Engine(lmp,arg[3],arg[4],arg[5]);
  Engine->mp_verlet=NULL;
  Engine->mp_verlet_sci=NULL;

#ifdef STATE_DECOMP
  // ** AWGL : check for multi state decomposition by partitions ** //
  if (narg >= 7) {
    int flag_mp_state = 0;
    if (strcmp(arg[6], "mp_state") == 0 ) {
      if (comm->me == 0) {
        if (screen)  fprintf(screen, "[EVB] Multi state decomposition flag detected.\n");
        if (logfile) fprintf(logfile,"[EVB] Multi state decomposition flag detected.\n");
      }
      flag_mp_state = atoi(arg[7]);
      if (!(flag_mp_state >= 1 && flag_mp_state <= 4)) 
        error->all(FLERR,"[EVB] Wrong state partition option. Valid is 1,2,3, or 4.");
      // If not running in multiple partition mode, just ignore
      if (!universe->existflag) {
        if (comm->me == 0) {
          if (screen)  fprintf(screen,  "[EVB] Not running with multiple partitions, so ignoring mp_state flag.\n");
          if (logfile) fprintf(logfile, "[EVB] Not running with multiple partitions, so ignoring mp_state flag.\n");
        }
        flag_mp_state = 0;
      }     
      Engine->flag_mp_state = flag_mp_state;
    }
  }
#endif

  /*** Set up command settings ***/ 
  if(narg>6)
  {
    for(int i=6; i<narg; i++)
    {
      char *cmd, *val;
      cmd = strtok(arg[i]," =\t\n");
      val = strtok(NULL," =\t\n");
      
      if(strcmp(cmd,"FDM")==0)
      {
        if(val==NULL) FDM_flag=0;
        else if(strcmp(val,"force")==0) FDM_flag=1;  
        else if(strcmp(val,"cec")==0) FDM_flag=2;       
      }
      else if(strcmp(cmd,"noreaction")==0) Engine->no_reaction=true;

      // This keyword is only currently used for pppm, but the variable is named for generality.
      else if(strcmp(cmd,"sci_pppm")==0) {
	if(val==NULL) Engine->SCI_KSPACE_flag=KSPACE_DEFAULT;
	else if(strcmp(val,"hf")==0) Engine->SCI_KSPACE_flag=PPPM_HF_FORCES;
	else if(strcmp(val,"acc")==0) Engine->SCI_KSPACE_flag=PPPM_ACC_FORCES;
	else if(strcmp(val,"polar")==0) Engine->SCI_KSPACE_flag=PPPM_POLAR_FORCES;
      }

      else if(strcmp(cmd,"sci_overlap") == 0) {
	if(val==NULL) Engine->sci_overlap_tol = 0.999;
	else Engine->sci_overlap_tol = atof(val);
      }

      else if(strcmp(cmd,"efieldz")==0) {
	Engine->EFIELD_flag = 1;
	Engine->efieldz = atof(val);
      }
    }
  }

  if(comm->me == 0) {
    if (screen && Engine->sci_overlap_tol > 0.0) fprintf(screen,"[EVB] SCI eigenvector overlap tolerance = %f\n",Engine->sci_overlap_tol);
    if (logfile && Engine->sci_overlap_tol > 0.0) fprintf(logfile,"[EVB] SCI eigenvector overlap tolerance = %f\n",Engine->sci_overlap_tol);
  }
  
  if(comm->me == 0 && Engine->EFIELD_flag) {
    if (screen) fprintf(screen,"[EVB] EFIELD active along z-axis: efieldz= %f\n",Engine->efieldz);
    if (logfile) fprintf(logfile,"[EVB] EFIELD active along z-axis: efieldz= %f\n",Engine->efieldz);
  }
  Engine->efieldz *= force->qe2f;

  /*** Construct other components ***/

#ifdef BGQ
  if(universe->me==0) {
    fprintf(stdout,"\nget_memory() in FixEVB constructor just before Engine->construct().\n");
    Engine->get_memory();
  }
#endif  

  grow_arrays(atom->nmax);
  Engine->construct();
  pack_dump();

#ifdef BGQ
  if(universe->me==0) {
    fprintf(stdout,"\nget_memory() at end of FixEVB constructor.\n");
    Engine->get_memory();
  }
#endif  
  
  /*** Message ***/
  if (comm->me==0)
  {
    if (screen) fprintf(screen,"[EVB] Construction finished.\n");
    if (screen) fprintf(screen,_EVB_LINE);
    if (screen) fprintf(screen, "[EVB] Copyright %s   Version %s\n", _EVB_COPYRIGHT,_EVB_VERSION);
    if (screen) fprintf(screen,_EVB_LINE);
  }
}

/* ---------------------------------------------------------------------- */

FixEVB::~FixEVB()
{
  int me;
  MPI_Comm_rank(world, &me);
  if(me==0) {
    if (screen) fprintf(screen,_EVB_LINE);
    if (screen) fprintf(screen,"[EVB] Destruct EVB module... \n");
  }
  
  if (Engine) delete Engine;
  memory->sfree(mol_type);
  memory->sfree(mol_index);
  memory->sfree(charge);
  memory->destroy(array_atom);
  
  atom->delete_callback(id, 0);
  atom->delete_callback(id, 1);
  
  if(me==0) {
    if (screen) fprintf(screen,"[EVB] Destruction finished.\n");
    if (screen) fprintf(screen,_EVB_LINE);
  }
}

/* ---------------------------------------------------------------------- */

int FixEVB::setmask()
{
    int mask = 0;
    mask |= PRE_FORCE;
    mask |= POST_FORCE;
    mask |= MIN_POST_FORCE;
    mask |= MIN_PRE_EXCHANGE;
    mask |= THERMO_ENERGY;
    return mask;
}

/* ---------------------------------------------------------------------- */

double FixEVB::compute_scalar()
{
    Engine->evb_timer->output();
    return Engine->energy;
}

/* ---------------------------------------------------------------------- */

void FixEVB::init()
{
  Engine->init();
}

/* ---------------------------------------------------------------------- */

void FixEVB::setup(int vflag)
{
  if(comm->me==0 && screen) fprintf(screen,"Setting up EVB ...\n");

  comm->forward_comm_fix(this);
  
  // If we just test a gas-phase system without KSPACE, still build FULL neighbor-list
  if(Engine->evb_kspace == NULL)
  {
    Neighbor *cracker = neighbor;
    cracker->special_flag[1] = 2;
    cracker->special_flag[2] = 2;
    cracker->special_flag[3] = 2;
        
    cracker->setup_bins();
    comm->exchange();
    comm->borders();
    cracker->build(0);
  }  
  
  switch (FDM_flag)
  {
    case 1: Engine->finite_difference_force(); break;
    case 2: Engine->finite_difference_cec(); break;
  }
   
  /*********************************************************************/
  /*********************************************************************/
  
  int nall = atom->nlocal + atom->nghost;
  double **f = atom->f;

  // remove the force calculation from all the other modules

#if defined (_OPENMP)
  int nthreads = comm->nthreads;
  for(int i=0; i<nall*nthreads; i++)
    f[i][0] = f[i][1] = f[i][2] = 0.0;
#else
  for(int i=0; i<nall; i++)
    f[i][0] = f[i][1] = f[i][2] = 0.0;
#endif

  // re-calculate force by MS-EVB
  Engine->execute(vflag);
  comm->reverse_comm();
  pack_dump();
    
  if(force->pair) force->pair->eng_vdwl = 0.0;
  if(force->pair) force->pair->eng_coul = 0.0;
  if(force->kspace) force->kspace->energy = 0.0;
  if(force->bond) force->bond->energy = 0.0;
  if(force->angle) force->angle->energy = 0.0;
  if(force->dihedral) force->dihedral->energy = 0.0;
  if(force->improper) force->improper->energy = 0.0;

  if (virial_flag)
    for (int i = 0; i < 6; i++)
      virial[i] =  Engine->virial[i];
  
  if(Engine->nreact) next_reneighbor=update->ntimestep+1;

  /*********************************************************************/
  /*********************************************************************/
}

void FixEVB::min_setup(int vflag)
{
  FixEVB::setup(vflag);
}

/* ---------------------------------------------------------------------- */

void FixEVB::set_force_compute(int flag)
{
  update->integrate->pair_compute_flag = flag;
  update->integrate->kspace_compute_flag = flag;
  atom->molecular = flag;
}

/* ---------------------------------------------------------------------- */

void FixEVB::pre_force(int vflag)
{    
  if(neighbor->ago==0) comm->forward_comm_fix(this);
  Engine->execute(vflag);
  
  if(Engine->nreact) next_reneighbor=update->ntimestep+1;
  set_force_compute(0);
}

/* ---------------------------------------------------------------------- */

void FixEVB::post_force(int vflag)
{
  /*********************************************************************/
  /*********************************************************************/
    
  set_force_compute(1);
   
  if(force->pair) force->pair->eng_vdwl = 0.0;
  if(force->pair) force->pair->eng_coul = 0.0;
  if(force->kspace) force->kspace->energy = 0.0;
  if(force->bond) force->bond->energy = 0.0;
  if(force->angle) force->angle->energy = 0.0;
  if(force->dihedral) force->dihedral->energy = 0.0;
  if(force->improper) force->improper->energy = 0.0;
    
  if (virial_flag)
    for (int i = 0; i < 6; i++)
      virial[i] =  Engine->virial[i];

  pack_dump();

  /*********************************************************************/
  /*********************************************************************/
}

/* ---------------------------------------------------------------------- */

void FixEVB::min_pre_exchange()
{
  set_force_compute(0);
}

void FixEVB::min_post_force(int vflag)
{
  Engine->execute(vflag); 
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixEVB::pack_dump()
{
    int nlocal = atom->nlocal;
    double *qeff = NULL;
    if(Engine->evb_effpair) Engine->evb_effpair->q;
    if(!qeff) qeff = atom->q;

    if(nlocal>size_array)
    {
        size_array = nlocal;
        memory->grow(array_atom,size_array,3,"fix_evb:array_atom");
    }

    for(int i=0; i<nlocal; i++)
    {
        array_atom[i][0] = mol_type[i];
        array_atom[i][1] = mol_index[i];
        array_atom[i][2] = qeff[i];
    }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixEVB::memory_usage()
{
  double bytes = 0;
  bytes = atom->nmax*2 * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array
------------------------------------------------------------------------- */

void FixEVB::grow_arrays(int nmax)
{ 
    Engine->mol_type = mol_type  = (int*)
      memory->srealloc(mol_type,sizeof(int)*nmax, "fix_evb:mol_type");
    Engine->mol_index = mol_index = (int*)
      memory->srealloc(mol_index,sizeof(int)*nmax, "fix_evb:mol_index");
    Engine->charge = charge = (double*)
      memory->srealloc(charge,sizeof(double)*nmax, "fix_evb:charge");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void FixEVB::copy_arrays(int i, int j, int delflag)
{
    mol_type[j] = mol_type[i];
    mol_index[j] = mol_index[i];    
    charge[j] = charge[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixEVB::pack_exchange(int i, double *buf)
{
    buf[0] = mol_type[i];
    buf[1] = mol_index[i];
    return 2;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixEVB::unpack_exchange(int nlocal, double *buf)
{
    mol_type[nlocal] = static_cast<int> (buf[0]);
    mol_index[nlocal] = static_cast<int> (buf[1]);
  
    return 2;
} 

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for File
------------------------------------------------------------------------- */

int FixEVB::pack_restart(int i, double *buf)
{
  buf[0] = 3;
  if(mol_type[i]) buf[1] = Engine->evb_type->id[mol_type[i]-1];
  else buf[1] = 0.0;
  buf[2] = mol_index[i];
  
  return 3;
}

/* ----------------------------------------------------------------------
   unpack values from atom->extra array to restart the fix
------------------------------------------------------------------------- */

void FixEVB::unpack_restart(int nlocal, int nth)
{
  double **extra = atom->extra;

  // skip to Nth set of extra values
  int m = 0;
  for (int i = 0; i < nth; i++) m += static_cast<int> (extra[nlocal][m]);
  m++;

  mol_type[nlocal] = static_cast<int> (extra[nlocal][m++]);
  mol_index[nlocal] = static_cast<int> (extra[nlocal][m++]);
  
  // type id
  int ntype = Engine->evb_type->type_count;
  int *tid = Engine->evb_type->id;

  if(mol_type[nlocal])
  {
    int i;
    for(i=0; i<ntype; i++)
      if(tid[i]==mol_type[nlocal]) 
      { mol_type[nlocal] = i+1; break; }
    if(i==ntype)
    {
      char errline[200];
      sprintf(errline,"[EVB] Error loading mol_type from restart file. Code: %d\n",mol_type[nlocal]);
      error->one(FLERR,errline);
    }
  }
}

/* ----------------------------------------------------------------------
   maxsize of any atom's restart data
------------------------------------------------------------------------- */

int FixEVB::maxsize_restart()
{
  return 3;
}

/* ----------------------------------------------------------------------
   size of atom nlocal's restart data
------------------------------------------------------------------------- */

int FixEVB::size_restart(int nlocal)
{
  return 3;
}

/* ---------------------------------------------------------------------- */

int FixEVB::pack_forward_comm(int n, int *list, double *buf,
			      int pbc_flag, int *pbc)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = mol_type[j];
    buf[m++] = mol_index[j];   
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixEVB::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    mol_type[i] = static_cast<int> (buf[m++]);
    mol_index[i] = static_cast<int> (buf[m++]);
  }
}

#ifdef RELAMBDA
// AWGL : Additional computes for lambda replica exchange
/* -------------------------------------------------------------------------------- */

double FixEVB::compute_array(int i, int j)
{
  // Returns data from arrays, or whatever
  double xx = 0.0;
  if (i == 0) xx = Engine->offdiag_lambda;
  if (i == 1) {
    // Recompute the EVB eneergy
    Engine->execute(false); // no virial calculation 
    xx = Engine->energy;
  }
  if (i == 2) xx = Engine->energy;
  return xx;
}

/* -------------------------------------------------------------------------------- */

void FixEVB::modify_fix(int which, double * values, char * notused)
{
  // Sets a specified variable to the input value(s)
  if (which == 0) {
    // Turn off lambda flag
    Engine->lambda_flag = 0;
  }
  else if (which == 1) {
    // Turn on lambda flag
    Engine->lambda_flag = 1;
  }
  else if (which == 2) {
    Engine->offdiag_lambda = values[0];
  }
  else if (which == 3) {
    // Change lambda flag to prevent file output
    Engine->lambda_flag = 2;
  }
}
#endif
