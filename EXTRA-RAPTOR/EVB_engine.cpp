/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
   Separate kspace for complex atoms added by Trung Nguyen (06Apr2023)
------------------------------------------------------------------------- */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define _CRACKER_NEIGHBOR
#include "EVB_cracker.h"
#undef _CRACKER_NEIGHBOR

#include "atom.h"
#include "atom_vec.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "update.h"
#define _CRACKER_PAIR  // cracked for GPU
#include "EVB_cracker.h"
#undef _CRACKER_PAIR
#include "angle.h"
#include "bond.h"
#include "dihedral.h"
#include "improper.h"

#define _CRACKER_KSPACE
#include "EVB_cracker.h"
#undef _CRACKER_KSPACE
#include "ewald.h"
#include "pppm.h"

#define _CRACKER_PAIR_HYBRID
#include "EVB_cracker.h"
#undef _CRACKER_PAIR_HYBRID

#include "comm.h"
#include "domain.h"
#include "timer.h"
#include "universe.h"

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

//#include "pair_electrode.h"
//#include "pair_electrode_omp.h"

#ifdef BGQ
#include <spi/include/kernel/memory.h>
#endif

#ifdef _RAPTOR_GPU
#include "timer.h"
#include "modify.h"
#include "fix_gpu.h"
#include "pair_lj_charmm_coul_long_gpu.h"
#include "pair_lj_cut_coul_long_gpu.h"
#endif

#define KSPACE_DEFAULT    0 // Hellman-Feynman forces for Ewald
#define PPPM_HF_FORCES    1 // Hellman-Feynman forces for PPPM
#define PPPM_ACC_FORCES   2 // Approximate (acc) forces for PPPM.
#define PPPM_POLAR_FORCES 3 // ACC forces plus an additional polarization force on complex atoms for PPPM.

//#define EVB_ENGINE_DEBUG

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------

   Member function: EVB_Engine::EVB_Engine()
   Description: Constructor
   Called by: Fix_EVB::Fix_EVB()

------------------------------------------------------------------------- */

EVB_Engine::EVB_Engine(LAMMPS *lmp, char* fin_name, char* fout_name, char* ftop_name) : Pointers(lmp)
{
  // Set global variables

  mol_type = mol_index = nullptr;
  charge = nullptr;

  evb_kspace   = nullptr;
  evb_kspace_cplx = nullptr;
  evb_type     = nullptr;
  evb_chain    = nullptr;
  evb_list     = nullptr;
  evb_reaction = nullptr;
  evb_output   = nullptr;
  evb_timer    = nullptr;

  evb_complex   = nullptr;
  evb_matrix    = nullptr;
  evb_repulsive = nullptr;
  evb_offdiag   = nullptr;

  all_complex   = nullptr;
  all_repulsive = nullptr;
  all_offdiag   = nullptr;

  no_reaction = false;

  flag_ACC          = 0;
  flag_DIAG_QEFF    = 0;
  flag_EFFPAIR_SUPP = 0;

  #ifdef STATE_DECOMP
  flag_mp_state     = 0;
  force_comm        = 0;
  group_root        = 0;
  group_rank        = 0;
  #endif

  #ifdef RELAMBDA
  // AWGL : for lambda replica exchange
  offdiag_lambda = -999999.0;
  lambda_flag = 0;
  #endif

  strcpy(cfg_name,fin_name);
  strcpy(out_name,fout_name);
  strcpy(top_name,ftop_name);

  // load-balancer for sci_mp simulations
  lb_tasklist    = nullptr;
  lb_cplx_master = nullptr;
  lb_cplx_split  = nullptr;
  lb_cplx_owned  = nullptr;
  lb_cplx_block  = nullptr;

  lb_num_owners      = nullptr;
  lb_cplx_owner_list = nullptr;

  num_tasks_per_part = nullptr;
  cplx_owner_list = nullptr;
  num_part_per_complex = nullptr;
  num_states_per_complex = nullptr;

  max_comm_ek = 0;
  comm_ek = nullptr;
  comm_ek2 = nullptr;

  start_indx_int = nullptr;
  start_indx_double = nullptr;
  size_total_int = nullptr;
  size_total_double = nullptr;

  buf_size_all = nullptr;
  buf_nextra_all = nullptr;
  buf_shell_all = nullptr;
  start_indx = nullptr;
  max_tmp_state_buf = 0;
  tmp_state_buf = nullptr;

  // buffers to sync avec
  old_size_gtotal_int = 0;
  old_size_gtotal_double = 0;
  comm_avec_buf_int = nullptr;
  comm_avec_buf_double = nullptr;

  debug_sci_mp = false; // DEBUG

  count_comm_create = 0; // DEBUG
  count_comm_total = 0; // DEBUG

  lb_comm_update = 1; // Default is to create new comms on first step.  Not all strategies use this.

  sci_cplx_count = nullptr;

  max_list_Vij = 0;
  list_Vij = nullptr;

  max_cc2 = 0;
  cc2 = nullptr;

  #ifdef _RAPTOR_GPU
  fix_gpu = nullptr;
  #endif

  evb_full_neigh = 0; // Default is half neighbor list
                      // GPU-enabled pair_styles use full list
                      // table/sw/lj/cut/coul/long uses full list

  // Hamiltonian screening
  bscreen_hamiltonian = false;
  screen_minP  = 1e-6;
  screen_minP2 = 1e-9;
  screen_max_list = 0;
  screen_del_list = nullptr;
  screen_delsum_list = nullptr;
  screen_important_list = nullptr;

  // Multipro_sci support for Hamiltonian screening
  screen_del_list_count = nullptr;
  screen_del_list_global = nullptr;

  // Method to compute KSPACE forces in SCI simulation
  // HF for PPPM. If Ewald used, then Ewald constructor will set this to zero for HF.
  SCI_KSPACE_flag = PPPM_HF_FORCES;

  // Indicates what work function in EVB_Engine is currently running
  engine_indicator = ENGINE_INDICATOR_COMPUTE;

  // Eigenvector overlap used to test convergence if positive value
  sci_overlap_tol = -1.0;

  // Electric field along z-axis
  EFIELD_flag = 0;
  efieldz = 0.0;
  efield_energy_env = 0.0;

  use_kspace_cplx = false;     // true if using a separate KSpace solver for non-env terms
  partial_gpu_offload = false;  // true if only the env diag is ported to the GPU

  first = true;
}

/* ---------------------------------------------------------------------- */
/*
   Member function: EVB_Engine::~EVB_Engine()
   Description: Destructor
   Called by: Fix_EVB::~Fix_EVB()

/* ---------------------------------------------------------------------- */

EVB_Engine::~EVB_Engine()
{
  // EVB_Complex

  if (all_complex) {
    for (int i = 0; i < ncomplex; i++) delete all_complex[i];
    delete [] all_complex;
  }

  // EVB_Matrix

  if (all_matrix) {
    for(int i = 0; i < ncomplex; i++) delete all_matrix[i];
    delete [] all_matrix;
  }
  delete full_matrix;

  // EVB_Repulsive

  for (int i = 0; i < nrepulsive; i++) delete all_repulsive[i];
  if (nrepulsive > 0) delete [] all_repulsive;

  // EVB_OffDiag

  for (int i = 0; i < evb_reaction->nPair; i++) delete all_offdiag[i];
  delete [] all_offdiag;

  // Other EVB objects

  if (evb_type)     delete evb_type;
  if (evb_chain)    delete evb_chain;
  if (evb_reaction) delete evb_reaction;
  if (evb_list)     delete evb_list;
  if (evb_effpair)  delete evb_effpair;
  if (evb_output)   delete evb_output;
  if (evb_timer)    delete evb_timer;

  if (evb_kspace_cplx) delete evb_kspace_cplx;

  // For complex

  memory->destroy(molecule_map);
  memory->sfree(complex_molecule);
  memory->sfree(complex_atom);
  memory->sfree(kernel_atom);
  memory->sfree(rc_molecule);
  memory->sfree(rc_molecule_prev);
  memory->sfree(rc_rank);

  // For SCI

  memory->sfree(max_coeff);
  memory->sfree(complex_pos);

  // For state-search

  delete [] is_center_node;
  delete [] is_center;

  #ifdef STATE_DECOMP
  if (flag_mp_state) {
    MPI_Comm_free(&force_comm);
  }
  #endif

  // load-balancer for sci_mp simulations
  delete [] lb_tasklist;
  delete [] lb_cplx_master;
  delete [] lb_cplx_split;
  delete [] lb_cplx_owned;
  delete [] lb_cplx_block;

  delete [] lb_num_owners;
  delete [] lb_cplx_owner_list;

  delete [] num_tasks_per_part;
  delete [] cplx_owner_list;
  delete [] num_part_per_complex;
  delete [] num_states_per_complex;

  memory->destroy(comm_ek);
  memory->destroy(comm_ek2);

  delete [] start_indx_int;
  delete [] start_indx_double;
  delete [] size_total_int;
  delete [] size_total_double;

  delete [] buf_size_all;
  delete [] buf_nextra_all;
  delete [] buf_shell_all;
  delete [] start_indx;
  delete [] tmp_state_buf;

  // buffers to sync avec
  memory->destroy(comm_avec_buf_int);
  memory->destroy(comm_avec_buf_double);

  delete [] sci_cplx_count;

  memory->destroy(list_Vij);
  memory->destroy(cc2);

  if (bscreen_hamiltonian) {
    memory->destroy(screen_del_list);
    memory->destroy(screen_delsum_list);
    memory->destroy(screen_important_list);

    memory->destroy(screen_del_list_count);
    memory->destroy(screen_del_list_global);
  }
}

/* ---------------------------------------------------------------------- */

int EVB_Engine::data_offdiag(char* buf, int* offset, int start, int end)
{
  int t = start;
  char name_line[1000];

  if (universe->me == 0) fprintf(fp_cfg_out,"\n\nStarting to process off-diagonal couplings.\n");

  if (t == end || strstr(buf+offset[t],"segment.off_diagonal")==0)
    error->all(FLERR,"[EVB] Expecting key word [segment.off_diagonal]");
  t++;

  for (int i = 0; i < evb_reaction->nPair; i++) {
    char* pp = strstr(buf+offset[t],"off_diagonal.start");
    if (t == end || pp == 0) error->all(FLERR,"[EVB] Expecting key word [off_diagonal.start]");
    t++;

    pp += 19;
    char *ppp=name_line;
    while (*pp && *pp!=']') {
      *ppp=*pp;
      pp++;
      ppp++;
    }
    *ppp = 0;

    int rtype = evb_reaction->get_reaction(name_line);
    if (rtype==-1) {
      char errline[255];
      sprintf(errline,"[EVB] Undefined reaction_type [%s].",name_line);
      error->all(FLERR,errline);
    }

    char* type_name=buf+offset[t++];

    if (universe->me == 0) {
      fprintf(fp_cfg_out,"\n   Off-diagonal coupling for reaction template: %s.\n",name_line);
      fprintf(fp_cfg_out,"   +++++++++++++++++++++++++++++++\n");
      fprintf(fp_cfg_out,"   Type of off-diagonal coupling: %s.\n",type_name);
      fprintf(fp_cfg_out,"   Starting to process off-diagonal.\n");
    }

    //////////////////////////////////

    if(0);

    #define EVB_MODULE_OFFDIAG
    #define MODULE_OFFDIAG(key,Class)                           \
    else if (strcmp(type_name,#key) == 0) {                     \
      all_offdiag[rtype-1] = new Class(lmp,this);               \
      t = all_offdiag[rtype-1]->data_offdiag(buf,offset,t,end); \
    }
    #include "EVB_module_offdiag.h"
    #undef EVB_MODULE_OFFDIAG

    else error->one(FLERR,"Unknown name of off-diagonals.");

    if (universe->me == 0) fprintf(fp_cfg_out,"   Finished processing off-diagonal.\n");

    //////////////////////////////////

    strcpy(all_offdiag[rtype-1]->name, name_line);

    if (t == end || strstr(buf+offset[t],"off_diagonal.end")==0)
      error->all(FLERR,"[EVB] Expecting key word [off_diagonal.end]");
    t++;
  } // end for i

  if (t == end || strstr(buf+offset[t],"segment.end")==0)
    error->all(FLERR,"[EVB] Expecting key word [segment.end]");
  t++;

  if (universe->me == 0) {
    fprintf(fp_cfg_out,"\nProcessing ALL off-diagonal couplings complete.\n");
    fprintf(fp_cfg_out,"\n==================================\n");
  }

  return t;
}

/* ----------------------------------------------------------------------*/

int EVB_Engine::data_repulsive(char *buf, int *offset, int start, int end)
{
  int t = start;

  if (universe->me == 0) fprintf(fp_cfg_out,"\n\nStarting to process diabatic shifts (a.k.a. repulsions).\n");

  if (t == end || strstr(buf+offset[t],"segment.repulsive")==0)
    error->all(FLERR,"[EVB] Expecting key word [segment.repulsive]");

  int count1=0, count2=0;
  for (t = start + 1; t < end; t++) {
    if (strstr(buf+offset[t],"repulsive.start")) {
      if (count1 > count2) error->all(FLERR,"[EVB] Expecting key word [repulsive.end].");
      count1++;
    } else if(strstr(buf+offset[t],"repulsive.end")) {
      count2++;
      if (count2 > count1) error->all(FLERR,"[EVB] Unexpected key word [repulsive.end].");
    }
  }
  if (count1 > count2) error->all(FLERR,"[EVB] Expecting key word [repulsive.end].");

  t = start+1;
  nrepulsive = count1;

  if (universe->me == 0) fprintf(fp_cfg_out,"Allocating memory for %i repulsions.\n",count1);

  if (nrepulsive > 0) {
    all_repulsive = new EVB_Repulsive*[nrepulsive];

    for (int i = 0; i < nrepulsive; i++) {
      char name_line[50];
      char* pp = strstr(buf+offset[t],"repulsive.start");
      if (t == end || pp ==0 ) error->all(FLERR,"[EVB] Expecting key word [repulsive.start]");
      t++;

      int nch = 0;
      while (pp[16+nch]!=']') nch++;
      pp[16+nch]=0;

      char* type_name = buf+offset[t++];

      if(universe->me == 0) {
        fprintf(fp_cfg_out,"\n   Starting to process repulsion: %s.\n",type_name);
        fprintf(fp_cfg_out,"   +++++++++++++++++++++++++++++++\n");
      }

      //////////////////////////////////

      if(0);

      #define EVB_MODULE_REPULSIVE
      #define MODULE_REPULSIVE(key,Class)                 \
      else if (strcmp(type_name,#key) == 0) {             \
        all_repulsive[i] = new Class(lmp,this);                 \
        t = all_repulsive[i]->data_rep(buf,offset,t,end);       \
      }
      #include "EVB_module_rep.h"
      #undef EVB_MODULE_REPULSIVE

      else {
        if (comm->me==0 && screen) fprintf(screen,"Parameter: %s >>> %s <<< %s\n",
                                          buf+offset[t-1], buf+offset[t], buf+offset[t+1]);
        error->one(FLERR,"Unknown name of repulsive term.");
      }

      if (universe->me == 0) fprintf(fp_cfg_out,"   Finished processing repulsion.\n");

      //////////////////////////////////

      strcpy(all_repulsive[i]->name, pp+16);

      if (t == end || strstr(buf+offset[t],"repulsive.end") == 0)
        error->all(FLERR,"[EVB] Expecting key word [repulsive.end]");
      t++;
    }
  } else
    evb_repulsive = nullptr;

  if (t == end || strstr(buf+offset[t],"segment.end") == 0)
    error->all(FLERR,"[EVB] Expecting key word [segment.end]");
  t++;

  if (universe->me == 0) {
    fprintf(fp_cfg_out,"\nProcessing ALL diabatic shifts complete.\n");
    fprintf(fp_cfg_out,"\n==================================\n");
  }

  return t;
}

/* ---------------------------------------------------------------------- */
/*
   Member function: EVB_Engine::construct()
   Description: Create all EVB module when EVB_Engine is constructed.
   Called by: Fix_EVB::Fix_EVB()

/* ---------------------------------------------------------------------- */

void EVB_Engine::construct()
{
  int me = comm->me;
  natom = atom->nlocal + atom->nghost;

  // Read input file

  Text* Cfg=nullptr;
  char *buf=nullptr;
  int nch, nword;
  int *word;

  if (universe->me==0) {

    char fname[255];
    strcpy(fname, cfg_name);
    strcat(fname, "-raptor.out");
    fp_cfg_out = fopen(fname,"w");

    Cfg = new Text(cfg_name,fp_cfg_out);
    nword = Cfg->nword;
    nch = Cfg->nch;
  }

  MPI_Bcast(&nword, 1, MPI_INT, 0, universe->uworld);
  MPI_Bcast(&nch, 1, MPI_INT, 0, universe->uworld);
  word = new int [nword];
  buf = new char[nch];

  if (universe->me==0) {
    memcpy(word,Cfg->word,sizeof(int)*nword);
    memcpy(buf,Cfg->buf,sizeof(char)*nch);
  }

  MPI_Bcast(word,nword,MPI_INT,0,universe->uworld);
  MPI_Bcast(buf,nch,MPI_CHAR,0,universe->uworld);

  // Build and store the EVB parameters
  int m;

  // Build output module
  if (comm->me==0) {
    evb_output = new EVB_Output(lmp, this, out_name);
    m = evb_output->data_output(buf,word,0,nword);
    evb_output->print("[EVB] Build and store the EVB parameters\n");
  }
  MPI_Bcast(&m,1,MPI_INT,0,universe->uworld);

  // EVB kernel types
  if (comm->me==0) evb_output->print("[EVB]   Build EVB molecule types\n");

  evb_type = new EVB_Type(lmp, this);
  m = evb_type->data_type(buf,word,m,nword);
  data_top();

  // EVB reactions and paths
  if (comm->me==0) evb_output->print("[EVB]   Build EVB reactions and paths\n");

  evb_reaction = new EVB_Reaction(lmp, this);
  m = evb_reaction->data_reaction(buf,word,m,nword);

  // EVB chains
  if (comm->me==0) evb_output->print("[EVB]   Build EVB kernel chains\n");

  evb_chain = new EVB_Chain(lmp, this);
  m = evb_chain->data_chain(buf,word,m,nword);

  // EVB off-diagonal terms
  if (comm->me==0) evb_output->print("[EVB]   Build EVB off-diagonals\n");

  all_offdiag = new EVB_OffDiag*[evb_reaction->nPair];
  m = data_offdiag(buf,word,m,nword);

  // EVB repulsive terms
  if (comm->me==0) evb_output->print("[EVB]   Build EVB repulsive terms\n");
  m = data_repulsive(buf,word,m,nword);

  // Set EVB complex variables
  complex_molecule = nullptr;
  max_coeff = nullptr;
  complex_atom = nullptr;
  complex_pos = nullptr;
  kernel_atom = nullptr;
  complex_atom_size = 0;

  // build molecule map
  molecule_map = nullptr;
  build_molecule_map();

  // build EVB object arrays based on # of RC
  rc_rank = rc_molecule = rc_molecule_prev = nullptr;
  count_rc();
  ncomplex = ncenter; // when single-rc or SCI-multi-rc

  full_matrix = new EVB_MatrixFull(lmp,this);
  all_complex = new EVB_Complex*[ncomplex];
  all_matrix = nullptr;

  for (int i = 0; i < ncomplex; i++) {
    all_complex[i] = new EVB_Complex(lmp,this);
    all_complex[i]->id = i + 1;
  }

  all_matrix = new EVB_MatrixSCI*[ncomplex];

  for (int i = 0; i < ncomplex; i++) {
    all_matrix[i] = new EVB_MatrixSCI(lmp,this);
    all_matrix[i]->cplx=all_complex[i];
  }

  evb_list  = new EVB_List(lmp,this);
  evb_timer = new EVB_Timer(lmp,this);

  // Check KSPACE object
  init_kspace();

  // EffPair
  evb_effpair = nullptr;
  evb_effpair = new EVB_EffPair(lmp,this);

  #ifdef DLEVB_MODEL_SUPPORT
  if (ncomplex > 1) {
    EVB14 = false;
    if (me == 0 && screen) fprintf(screen,"DLEVB 1-4 interactions not yet coded with SCI.\n");
  } else {
    EVB14 = true;
    if (me == 0 && screen) fprintf(screen,"[WARNING] Use DLEVB 1-4 scaling rules for dihedrals\n");
  }
  #endif

  // EVB extensions

  if (comm->me==0) evb_output->print("[EVB]   Read extensible parameters\n");
  m = data_extension(buf,word,m,nword);

  if (Cfg) delete Cfg;
  if (word) delete [] word;
  if (buf) delete [] buf;

  if (universe->me == 0) {
    fprintf(fp_cfg_out,"\n\nProcessing cfg file complete.\n");
    fclose(fp_cfg_out);
  }

  // Flush debugging message
  if (comm->me == 0 && logfile) {
    char tmpstr[255];
    sprintf(tmpstr,"[EVB] %d/%d parameters were read\n",m,nword);
    evb_output->print(tmpstr);
    evb_output->print("\n");
    evb_output->print(_EVB_LINE);
    fflush(logfile);
  }

  #ifdef STATE_DECOMP
  if (flag_mp_state) {
    // ** AWGL : Set up communication list for multi state partitioning once and for all ** //
    int * map_u2myloc = new int[3*universe->nprocs];
    int * map_buffer  = new int[3*universe->nprocs];

    for(int i=0; i<3*universe->nprocs; ++i) map_buffer[i] = map_u2myloc[i] = 0;
    map_buffer[3*universe->me + 0] = comm->myloc[0];
    map_buffer[3*universe->me + 1] = comm->myloc[1];
    map_buffer[3*universe->me + 2] = comm->myloc[2];
    MPI_Allreduce(map_buffer, map_u2myloc, 3*universe->nprocs, MPI_INT, MPI_SUM, universe->uworld);

    // Allocate the list
    comm_list = new int[universe->nworlds];

    // save index if proc has same location as me
    int n = 0;
    for (int i = 0; i < universe->nprocs; ++i) {
      if (map_u2myloc[3*i + 0] == comm->myloc[0])
        if (map_u2myloc[3*i + 1] == comm->myloc[1])
          if (map_u2myloc[3*i + 2] == comm->myloc[2]) comm_list[n++] = i; // <-- universe proc rank of each proc with same coordinate
    }

    // ** Special: For reordering, print out the MPI ranks of group ** //
    /*
    if (comm->me == 0 && logfile) {
      fprintf(logfile, "***** Force group ranks *****\n");
      for (int xx=0; xx<comm->procgrid[0]; ++xx) {
        for (int yy=0; yy<comm->procgrid[1]; ++yy) {
          for (int zz=0; zz<comm->procgrid[2]; ++zz) {
            fprintf(logfile, "Group %3d %3d %3d : ", xx, yy, zz);
            for(int i=0; i<universe->nprocs; ++i) {
              if (map_u2myloc[3*i + 0] == xx)
              if (map_u2myloc[3*i + 1] == yy)
              if (map_u2myloc[3*i + 2] == zz)
                fprintf(logfile, " %d ", i);
            }
            fprintf(logfile, "\n");
          }
        }
      }
      fprintf(logfile, "*****************************\n");
    }
    MPI_Barrier(universe->uworld);
    */

    delete [] map_u2myloc;
    delete [] map_buffer;

    // ** Collective communicator for forces ** //
    MPI_Group universe_group, force_group;
    MPI_Comm_group(universe->uworld, &universe_group);
    MPI_Group_incl(universe_group, universe->nworlds, comm_list, &force_group);
    MPI_Comm_create(universe->uworld, force_group, &force_comm);

    // ** Determine root (i.e., universe->iworld==0) for the group ** //
    MPI_Comm_rank(force_comm, &group_rank);
    int* group_info_r = new int [universe->nworlds];
    int* group_info_s = new int [universe->nworlds];
    memset(group_info_s, 0, sizeof(int)*universe->nworlds);
    group_info_s[universe->iworld] = group_rank;
    MPI_Allreduce(group_info_s, group_info_r, universe->nworlds, MPI_INT, MPI_SUM, force_comm);
    group_root = group_info_r[0];

    delete [] group_info_r;
    delete [] group_info_s;
    delete [] comm_list;
  }
  #endif

  sci_cplx_count = new int [3 * ncomplex];

  #ifdef _RAPTOR_GPU
  int ifix = modify->find_fix("package_gpu");
  if (ifix < 0) fix_gpu = nullptr;
  else {
    fix_gpu = modify->fix[ifix];

    // Remove GPU fix from list of post_force and min_post_force
    //  This works because Modify::init() hasn't been called yet.
    //  FixGPU::post_force is now only called during FixGPU::setup
    modify->fmask[ifix] = 0;
  }
        
  if (fix_gpu) {
    if (comm->me == 0 && screen) {
      fprintf(screen,"[EVB]   fix_gpu found!\n");
      fprintf(screen,"[EVB]   *** IMPORTANT ***\n");
      fprintf(screen,"[EVB]   1. Neighor builds should be done on the host, must use 'neigh no'.\n");
      fprintf(screen,"[EVB]   2. 'split' must be 1.0 (by default).\n");
      fprintf(screen,"[EVB]   3. When kspace_cplx is no (by default), pppm is done on the CPU (equivalent to using 'pair/only yes').\n");
      if (partial_gpu_offload)
        fprintf(screen,"[EVB]   Offload only the env real-space diagonal term to the GPU (partial_offload yes)\n");
      else
        fprintf(screen,"[EVB]   Offload all the env real-space diagonal terms to the GPU (partial_offload no)\n");
    }
    evb_full_neigh = 1; // Full neighbor list calculated.

    if (ncomplex > 1) error->universe_all(FLERR,"ncomplex > 1 does not yet support GPUs");

    // enforce "neigh no" and "split 1.0"

    FixGPU* fix = dynamic_cast<FixGPU *>(fix_gpu);
    if (fix->gpu_mode() != 0) 
      error->universe_all(FLERR,"RAPTOR requires -pk gpu with neigh no");
    if (fix->particle_split() < 1) 
      error->universe_all(FLERR,"RAPTOR requires -pk gpu with split 1.0");

  } else {
    if(comm->me == 0 && screen) fprintf(screen,"[EVB]   fix_gpu not found\n");
  }
  #endif

  if (use_kspace_cplx) {
    if (comm->me == 0 && screen) {
      fprintf(screen,"[EVB] Using a separate KSpace solver for complex charges\n");
    }
  }

  if (strcmp(force->pair_style,"table/sw/lj/cut/coul/long") == 0)
    evb_full_neigh = 1; // Full neighbor list calculated
}

/* ---------------------------------------------------------------------- */
/*
/* ---------------------------------------------------------------------- */

void EVB_Engine::init_kspace()
{
  bEffKSpace = false;
  bDelayEff = false;

  if (force->kspace) {
    int _narg=2;
    char _buf[40];
    char *_arg[2];

    _arg[0] = _buf; _arg[1] = _buf+25;
    sprintf(_arg[1],"%lf", force->kspace->accuracy_relative);

    if (strstr(force->kspace_style,"evb_")==force->kspace_style) {
      sprintf(_arg[0],"%s", force->kspace_style);
    }
    /*
    else if(strcmp(force->kspace_style,"ewald")==0) {
      sprintf(_arg[0],"%s", "evb_ewald");
      SCI_KSPACE_flag=0;
    }
    */
    else if (strcmp(force->kspace_style,"pppm")==0)
      sprintf(_arg[0],"%s", "evb_pppm");
    else if(strcmp(force->kspace_style,"pppm/gpu")==0) {
      if (use_kspace_cplx)
        sprintf(_arg[0],"%s", "evb_pppm/gpu");
      else
        sprintf(_arg[0],"%s", "evb_pppm");
    } else if(strcmp(force->kspace_style,"pppm/omp")==0) {
      sprintf(_arg[0],"%s", "evb_pppm");
    }
    /*
    else if(strcmp(force->kspace_style,"ewald/acc")==0) {
      sprintf(_arg[0],"%s", "evb_ewald/acc");
      SCI_KSPACE_flag=0;
    } else if(strcmp(force->kspace_style,"pppm/acc")==0)
      sprintf(_arg[0],"%s", "evb_pppm/acc");

    else if(strcmp(force->kspace_style,"pppm/acc/omp")==0)
      sprintf(_arg[0],"%s", "evb_pppm/acc/omp");

    else if(strcmp(force->kspace_style,"pppm/crhydroxide")==0)
      sprintf(_arg[0],"%s", "evb_pppm/crhydroxide");

    // ** AWGL ** //
    else if(strcmp(force->kspace_style,"pppm/omp")==0)
      sprintf(_arg[0],"%s", "evb_pppm/omp");

    else if(strcmp(force->kspace_style,"pppm/gpu")==0)
      sprintf(_arg[0],"%s", "evb_pppm/gpu");

    else if(strcmp(force->kspace_style,"pppm/electrode")==0);

    else if(strcmp(force->kspace_style,"pppm/electrode/omp") == 0);
    */
    else {
      char tmpstr[255];
      sprintf(tmpstr,"[EVB] K-Space %s cannot be used in EVB yet.\n",force->kspace_style);
      error->all(FLERR,tmpstr);
    }

    // if(strcmp(force->kspace_style,"pppm/omp")==0) force->create_kspace(_narg,_arg,1);
    // else if(strcmp(force->kspace_style,"pppm/acc/omp")==0) force->create_kspace(_narg,_arg,1);
    // else if(strcmp(force->kspace_style,"pppm/gpu")==0) force->create_kspace(_narg,_arg,1);
    // else if(strcmp(force->kspace_style,"pppm/electrode")==0);
    // else if(strcmp(force->kspace_style,"pppm/electrode/omp")==0);
    // else force->create_kspace(_narg,_arg,1);

    if (strcmp(force->kspace_style,"pppm/electrode")==0);
    else if (strcmp(force->kspace_style,"pppm/electrode/omp") == 0);
    else {
      int _differentiation_flag = force->kspace->differentiation_flag;

      force->create_kspace(_arg[0], 0);
      force->kspace->settings(_narg-1, _arg+1);
      force->kspace->differentiation_flag = _differentiation_flag;
    }

    // Check for SCI support

    if (ncomplex > 1) {
      char tmpstr[255];
      if (SCI_KSPACE_flag == KSPACE_DEFAULT) sprintf(tmpstr,"[EVB] SCI + EWALD using Hellman-Feynman KSpace forces.\n");
      if (SCI_KSPACE_flag == PPPM_HF_FORCES) sprintf(tmpstr,"[EVB] SCI + PPPM using Hellman-Feynman KSpace forces.\n");
      if (SCI_KSPACE_flag == PPPM_ACC_FORCES) sprintf(tmpstr,"[EVB] SCI + PPPM using approximate (acc) KSpace forces.\n");
      if (SCI_KSPACE_flag == PPPM_POLAR_FORCES) sprintf(tmpstr,"[EVB] SCI + PPPM using acc + polarization KSpace forces.\n");

      if (universe->me==0 && screen) fprintf(screen,tmpstr);
      if (universe->me==0 && logfile) fprintf(logfile,tmpstr);
    }

    if (ncomplex > 1 && strstr(force->kspace_style,"pppm")) {
      flag_DIAG_QEFF = 1;
    }

    if (ncomplex > 1 && strstr(force->kspace_style,"ewald/acc")) {
      char tmpstr[255];
      sprintf(tmpstr,"[EVB] ewald/acc cannot be used in SCI simulations yet.\n");
      error->all(FLERR,tmpstr);
    }

    if (ncomplex > 1 && strstr(force->kspace_style,"pppm/acc")) {
      char tmpstr[255];
      sprintf(tmpstr,"[EVB] pppm/acc cannot be used in SCI simulations yet.\n");
      error->all(FLERR,tmpstr);
    }

    if (ncomplex > 1 && strstr(force->kspace_style,"pppm/crhydroxide")) {
      char tmpstr[255];
      sprintf(tmpstr,"[EVB] pppm/acc cannot be used in SCI simulations yet.\n");
      error->all(FLERR,tmpstr);
    }

    if (ncomplex > 1 && strstr(force->kspace_style,"pppm/gpu")) {
      char tmpstr[255];
      sprintf(tmpstr,"[EVB] pppm/gpu cannot be used in SCI simulations yet.\n");
      error->all(FLERR,tmpstr);
    }

    if (ncomplex > 1 && strcmp(force->kspace_style,"pppm/electrode") == 0) {
      char tmpstr[255];
      sprintf(tmpstr,"[EVB] pppm/electrode cannot be used in SCI simulations; switch to evb_pppm/electrode.\n");
      error->all(FLERR,tmpstr);
    }

    if (ncomplex > 1 && strcmp(force->kspace_style,"pppm/electrode/omp") == 0) {
      char tmpstr[255];
      sprintf(tmpstr,"[EVB] pppm/electrode/omp cannot be used in SCI simulations; switch to evb_pppm/electrode.\n");
      error->all(FLERR,tmpstr);
    }

    evb_kspace = (EVB_KSpace*)(force->kspace);
    evb_kspace->evb_engine = this;
    evb_kspace->evb_timer = evb_timer;

    // create a separate kspace for complex: experimenting with compute_cplx() on the GPU
    //   cannot have both compute_env and compute_cplx on the GPU because there is a single PPPMGPU instance on the GPU lib side
    //#ifdef USE_KSPACE_CPLX
    if (use_kspace_cplx) {
      int sflag;
      evb_kspace_cplx = (EVB_KSpace*)force->new_kspace((char*)"evb_pppm", 0, sflag);
      evb_kspace_cplx->evb_engine = this;
      evb_kspace_cplx->evb_timer = evb_timer;

      // TDN: to see the effect of fewer mesh points with the separate kspace solver for cplx atoms
      // For the ClC system: ncplx_atoms = 28, q2 = 6.4242 in a box of 106 by 106 by 80 Angtroms
      //    the accuracy of 1e-5 requires a 3d grid of (20 20 15),
      //     and for the accuracy of 1e-4 requires a 3d grid of (18 18 15)
      // Here I hard-coded 
      //   accuracy_relative = 0.0015 for the evb_kspace_cplx so its init() yields the same grid (20 20 15) for accuracy = 1e-5; g_ewald = 0.155, 
      //   accuracy_relative = 0.0025 for the evb_kspace_cplx so its init() yields the same grid (18 18 15) for accuracy = 1e-4; g_ewald = 0.149
      evb_kspace_cplx->accuracy_relative = 0.0015; //0.0015; //0.0025;
      //    also needs to match off diagonal kspace energies with the original evb_kspace (offdiag energy 36.126):
      evb_kspace_cplx->gewaldflag = 1;
      evb_kspace_cplx->g_ewald = 0.2418; // for acc 1e-5: leading to a (32 32 25) grid
      //evb_kspace_cplx->g_ewald = 0.2049; // for acc 1e-4: leading to a (25 25 20) grid

      // experimenting with specifying mesh points
      //evb_kspace_cplx->gridflag = 1;
      //evb_kspace_cplx->nx_pppm = 24;
      //evb_kspace_cplx->ny_pppm = 24;
      //evb_kspace_cplx->nz_pppm = 20;
      // experimenting with analytical differentiation with kspace_cplx
      //evb_kspace_cplx->differentiation_flag = 1;
    }
    
    if (evb_kspace->bEff) bEffKSpace = true;
    evb_type->init_kspace();

  } else {
    error->warning(FLERR,"[EVB] Gas-phase calculation has not been fully tested yet!");
    if(ncomplex>1) error->warning(FLERR,"[EVB] SCI without k-space is not correct!!!");
  }
}

/* ---------------------------------------------------------------------- */
/*
   Member function: EVB_Engine::data_top()
   Description: Read TOP file, get and build [mol_type] and [mol_index].
   Called by: EVB_Engine::construct()

/* ---------------------------------------------------------------------- */

#define BLOCK 500000 // max lines read per time

void EVB_Engine::data_top()
{
  if(universe->me == 0) fprintf(fp_cfg_out,"\n\nStarting to process top file.\n");

  int *tag = atom->tag;
  FILE *fp;
  int data[BLOCK*3]; // three columes data: atom_tag, mol_type, mol_index
  int i,j,k;
  char line[256];

  // Open the file handle for reading
  if (universe->me==0) {
    if(screen) fprintf(screen,"[EVB] Reading TOPOLOGY from file %s\n",top_name);
    fp = fopen(top_name,"r");
    if (!fp) {
      char errline[255];
      sprintf(errline,"[EVB] Cannot open input file: %s", top_name);
      error->one(FLERR,errline);
    }
  }

  // Initialize the arrays

  memset(mol_type,0, sizeof(int)*atom->nmax);
  memset(mol_index,0, sizeof(int)*atom->nmax);

  // input loop

  while (true) {

    // Read text from rank 0 and format
    if (universe->me==0) for(i=0; i<BLOCK; i++) {
        if(!fgets(line,255,fp)) break;
        sscanf(line,"%d %d %d", data+i*3, data+i*3+1, data+i*3+2);
      }

    // Broadcast data

    MPI_Bcast(&i,1,MPI_INT,0,universe->uworld);
    MPI_Bcast(data,i*3,MPI_INT,0,universe->uworld);

    // allocate data
    int ntype = evb_type->type_count;
    int *tid = evb_type->id;

    for(int j = 0; j < i; j++) {
      int id = data[j*3];
      int itype = data[j*3+1];
      if (itype != 0)
        for (int k = 0; k < ntype; k++)
          if (tid[k] == itype) {
            itype = k + 1;
            break;
          }

      k = atom->map(id);
      if (k >= 0) {
        mol_type[k] = itype;
        mol_index[k] = data[j*3+2];
      }
    }

    if (i < BLOCK) break;
  }

  // Close file

  if (universe->me == 0) {
    fclose(fp);
    fprintf(fp_cfg_out,"\nProcessing top file complete\n");
    fprintf(fp_cfg_out,"\n==================================\n");
  }
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::init()
{
  int me = comm->me;

  bHybridPair = bHybridBond = bHybridAngle = bHybridDihedral = bHybridImproper = false;

  if(strcmp(force->pair_style,"hybrid")==0 || strcmp(force->pair_style,"hybrid/overlay")==0) bHybridPair = true;
  if(strcmp(force->bond_style,"hybrid")==0) bHybridBond = true;
  if(strcmp(force->angle_style,"hybrid")==0) bHybridAngle = true;
  if(strcmp(force->dihedral_style,"hybrid")==0) bHybridDihedral = true;
  if(strcmp(force->improper_style,"hybrid")==0) bHybridImproper = true;

  //if(bHybridPair)
  //error->all(FLERR,"Warning: \"hybrid\" pair_style has not been considered in EVB module yet!");

  if (strstr(force->pair_style,"lj/cut/coul/")!=force->pair_style &&
      strstr(force->pair_style,"lj/charmm/coul/")!=force->pair_style && me==0) {
    error->warning(FLERR,"[EVB] Not a confirmed pair_style in the mission.");
    error->warning(FLERR,force->pair_style);
  }

  if (evb_kspace) {
    double _qsqsum;
    _qsqsum = qsqsum_sys = 0.0;
    double *q = atom->q;

    for (int i = 0; i < atom->nlocal; i++) _qsqsum += (q[i]*q[i]);

    MPI_Allreduce(&_qsqsum, &qsqsum_sys, 1, MPI_DOUBLE, MPI_SUM, world);
  }

  // Increase per atom bond lists

  increase_atomic_space();

  // Increase neigh_bond lists

/* YP: Temporarily remove this part as it may not be needed for non- blue-gene systems

  int nprocs = neighbor->nprocs;
  #define LB_FACTOR 1.5

  if (atom->molecular && atom->nbonds && neighbor->maxbond == 0) {
    if (nprocs == 1) neighbor->maxbond = atom->nbonds+10;
    else neighbor->maxbond = static_cast<int> (LB_FACTOR * atom->nbonds / nprocs) +10;
    memory->create(neighbor->bondlist,neighbor->maxbond,3,"neigh:bondlist");
  }

  if (atom->molecular && atom->nangles && neighbor->maxangle == 0) {

    if (nprocs == 1) neighbor->maxangle = atom->nangles+10;
    else neighbor->maxangle = static_cast<int> (LB_FACTOR * atom->nangles / nprocs)+10;
    neighbor->anglelist =  memory->create(neighbor->anglelist,neighbor->maxangle,4,"neigh:anglelist");
  }

  if (atom->molecular && atom->ndihedrals && neighbor->maxdihedral == 0) {
    if (nprocs == 1) neighbor->maxdihedral = atom->ndihedrals+10;
    else neighbor->maxdihedral = static_cast<int>
           (LB_FACTOR * atom->ndihedrals / nprocs)+10;
    memory->create(neighbor->dihedrallist,neighbor->maxdihedral,5,"neigh:dihedrallist");
  }

  if (atom->molecular && atom->nimpropers && neighbor->maximproper == 0) {
    if (nprocs == 1) neighbor->maximproper = atom->nimpropers+10;
    else neighbor->maximproper = static_cast<int>
           (LB_FACTOR * atom->nimpropers / nprocs)+10;
    memory->create(neighbor->improperlist,neighbor->maximproper,5,"neigh:improperlist");
  }

*/

  // effective potential
  if (evb_effpair) evb_effpair->init();

  // AWGL : for special atom flags
  has_complex_atom = 0;
  has_exch_chg = 0;
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::increase_atomic_space()
{
  #define __TEST_MAX(type)                                \
  int grow_##type=0, save_##type=0;                       \
  if (evb_type->type##_per_atom>atom->type##_per_atom) {  \
    save_##type = atom->type##_per_atom; grow_##type=1;   \
    atom->type##_per_atom=evb_type->type##_per_atom;      \
  }

  __TEST_MAX(bond);
  __TEST_MAX(angle);
  __TEST_MAX(dihedral);
  __TEST_MAX(improper);

  if (grow_bond || grow_angle || grow_dihedral || grow_improper) {
    char warn_str[255];
    sprintf(warn_str,
           "[EVB] Not fully tested: Increase atomic space [bond|angle|dihedral|improper]= %d %d %d %d",
           grow_bond, grow_angle, grow_dihedral, grow_improper);
    if (comm->me == 0) error->warning(FLERR,warn_str);

    atom->avec->grow(atom->nmax);
    int nlocal = atom->nlocal;

    #define  __REMAP_ARRAY(array, type) {                 \
      int *tt = &(atom->array[0][0]);                     \
      for (int i = nlocal-1; i >= 0; i--)                 \
        for (int j = atom->num_##type[i]-1; j >= 0; j--)        \
          atom->array[i][j] = tt[i*save_##type+j]; }

    if (grow_bond) {
      __REMAP_ARRAY(bond_type,bond);
      __REMAP_ARRAY(bond_atom,bond);
    }

    if (grow_angle) {
      __REMAP_ARRAY(angle_type,angle);
      __REMAP_ARRAY(angle_atom1,angle);
      __REMAP_ARRAY(angle_atom2,angle);
      __REMAP_ARRAY(angle_atom3,angle);
    }

    if (grow_dihedral) {
      __REMAP_ARRAY(dihedral_type,dihedral);
      __REMAP_ARRAY(dihedral_atom1,dihedral);
      __REMAP_ARRAY(dihedral_atom2,dihedral);
      __REMAP_ARRAY(dihedral_atom3,dihedral);
      __REMAP_ARRAY(dihedral_atom4,dihedral);
    }

    if( grow_improper) {
      __REMAP_ARRAY(improper_type,improper);
      __REMAP_ARRAY(improper_atom1,improper);
      __REMAP_ARRAY(improper_atom2,improper);
      __REMAP_ARRAY(improper_atom3,improper);
      __REMAP_ARRAY(improper_atom4,improper);
    }
  }
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::execute(int vflag)
{
  TIMER_STAMP(EVB_Engine, execute);

  pre_process(vflag);

  if (first) {
    if (evb_kspace_cplx) {
      
      // NOTE: compute the q2 value of the complex atoms after build_cplx_map()
      //   need a way to initialize the global grid of evb_kspace_cplx
      //   with q2 and number of complex atoms (not q2 and natoms of the whole system)

      double tmp, q2sum = 0.0;
      double* q = atom->q;
      int nlocal = atom->nlocal;
      int num, ncomplex_atoms = 0;
      for (int i = 0; i < nlocal; ++i) {
        if (complex_atom[i]) {
          q2sum += q[i]*q[i];
          ncomplex_atoms++;
        }
      }

      MPI_Allreduce(&q2sum,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
      q2sum = tmp;
      MPI_Allreduce(&ncomplex_atoms,&num,1,MPI_INT,MPI_SUM,world);
      ncomplex_atoms = num;

      //if (comm->me == 0) printf("Total number of complex atoms = %d q2 = %f\n", ncomplex_atoms, q2sum);

      // Need to pass ncomplex and q2sum in place of natoms and q2 in EVB_PPPM::init()

      evb_kspace_cplx->init();
      evb_kspace_cplx->setup();
    }
    first = false;
  }

  // Hamiltonian screening
  if (bscreen_hamiltonian) {
    if (mp_verlet_sci) screen_states_mp(vflag);
    else screen_states(vflag);
    screen_finalize();
  }

  if (mp_verlet_sci) {
    compute_sci_mp(vflag);
    post_process_sci_mp(vflag);
  } else {
    compute(vflag);
    post_process(vflag);
  }

  TIMER_CLICK(EVB_Engine, execute);
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::pre_process(int vflag)
{
  TIMER_STAMP(EVB_Engine, pre_process);

  // Since run_style is now setup, check for SCI_MP + Ewald

  if (force->kspace && mp_verlet_sci) {
    if (strcmp(force->kspace_style,"evb_ewald")     == 0) error->universe_all(FLERR,"Support for SCI_MP + Ewald is not ready. Switch to PPPM.");
    if (strcmp(force->kspace_style,"evb_ewald/acc") == 0) error->universe_all(FLERR,"Support for SCI_MP + Ewald is not ready. Switch to PPPM.");
  }

  /****************************************************************/
  /***   Get necessary variables and pointers *********************/
  /****************************************************************/

  int nlocal = atom->nlocal;
  natom      = nlocal + atom->nghost;
  nreact     = 0;
  evb_matrix = full_matrix;

  #ifdef STATE_DECOMP
  if (flag_mp_state) {
    // Check for same number of atoms
    int num = natom;
    int all_num = 0;
    MPI_Allreduce(&num, &all_num, 1, MPI_INT, MPI_SUM, force_comm);
    int force_comm_size;
    MPI_Comm_size(force_comm, &force_comm_size);
    if (all_num / force_comm_size != natom) {
      printf("Partition %d does not have the same number of atoms in all force comms!\n", universe->me);
      error->one(FLERR,"Number of atoms not the same on all state partition force comms!\n");
    }
    // Flush output b/c I don't like to wait till the end
    if(comm->me==0 && logfile) fflush(logfile);
    double **x = atom->x;
    MPI_Bcast(&x[0][0], 3*natom, MPI_DOUBLE, group_root, force_comm);
    double **v = atom->v;
    MPI_Bcast(&v[0][0], 3*natom, MPI_DOUBLE, group_root, force_comm);
  }
  #endif

  #ifdef BGQ
  if (universe->me==0 && screen) {
    fprintf(screen,"\nget_memory() start of pre_process().\n");
    get_memory();
  }
  #endif

  /****************************************************************/
  /***   Increase the complex_map if needed  **********************/
  /****************************************************************/

  if (natom > complex_atom_size) {
    complex_atom_size = natom;
    complex_atom = (int*) memory->srealloc(complex_atom,sizeof(int)*complex_atom_size, "EVB_Engine:complex_atom");
    kernel_atom = (int*) memory->srealloc(kernel_atom,sizeof(int)*complex_atom_size, "EVB_Engine:kernel_atom");
    complex_pos = (int*) memory->srealloc(complex_pos,sizeof(int)*complex_atom_size, "EVB_Engine:complex_pos");
  }

  /****************************************************************/
  /***   Update molecule_map if comm->exchange happened  **********/
  /****************************************************************/

  if (neighbor->ago==0) update_molecule_map();

  /****************************************************************/
  /***   State_search algorithm ***********************************/
  /****************************************************************/

  #ifdef BGQ
  if (universe->me==0) {
    fprintf(stdout,"\nget_memory() just before state_search().\n");
    get_memory();
  }
  #endif

  locate_rc();
  if (mp_verlet_sci)
    state_search_sci_mp();
  else
    state_search();

  #ifdef BGQ
  if(universe->me==0) {
    fprintf(stdout,"\nget_memory() just after state_search().\n");
    get_memory();
  }
  #endif

  // Setup load-balancer for sci_mp simulations
  //if (mp_verlet_sci) setup_lb_mp();

  /****************************************************************/
  /***   Setup all EVB objects   **********************************/
  /****************************************************************/

  if (evb_effpair)
    evb_effpair->setup();
  evb_list->setup();

  #ifdef BGQ
  if (universe->me==0) {
    fprintf(stdout,"\nget_memory() just before full_matrix->setup().\n");
    get_memory();
  }
  #endif

  full_matrix->setup();

  #ifdef BGQ
  if (universe->me==0) {
    fprintf(stdout,"\nget_memory() just after full_matrix->setup().\n");
    get_memory();
  }
  #endif

  evb_reaction->setup();
  for (int i = 0; i < ncomplex; i++)
    all_complex[i]->setup();

  //if (mp_verlet_sci) sci_comm_cplx_map();

  cplx_map_changed = 0;
  if (!mp_verlet_sci) {
    for (int i = 0; i < ncomplex; i++) {
      all_complex[i]->build_cplx_map();
      if (all_complex[i]->complex_map_changed) {
        cplx_map_changed = 1;
        //printf("complex %d changed their map on proc %d at t = %d\n", i, comm->me, update->ntimestep);
      }
        
    }
  }
  #ifdef EVB_ENGINE_DEBUG
  int ntmp = 0;
  MPI_Allreduce(&cplx_map_changed, &ntmp, 1, MPI_INT, MPI_MAX, world);
  cplx_map_changed = ntmp;
  #endif

  if (evb_kspace)
    evb_kspace->evb_setup();
  if (evb_kspace_cplx)
    evb_kspace_cplx->evb_setup();

  #ifdef BGQ
  if (universe->me==0) {
    fprintf(stdout,"\nget_memory() at end of EVB_engine::pre_process().\n");
    get_memory();
  }
  #endif

  TIMER_CLICK(EVB_Engine, pre_process);
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::compute(int vflag)
{
  TIMER_STAMP(EVB_Engine, compute);

  engine_indicator = ENGINE_INDICATOR_COMPUTE;

  // store the atom force array into lmp_f

  lmp_f      = atom->f;
  energy     = 0.0;

  for (int i = 0; i < 6; i++) virial[i] = 0.0;
  clear_virial();

  /****************************************************************/
  /***    Single-complex subroutine   *****************************/
  /****************************************************************/

  for (int i = 0; i < ncomplex; i++) {

    evb_complex = all_complex[i];

    if (mp_verlet_sci)
      evb_complex->build_cplx_map();   // this is already done in pre_process() for other styles

    if (neighbor->ago == 0)
      check_for_complex_atoms();       // AWGL : Identify if my rank contains complex atoms

    evb_complex->setup_avec();
    evb_complex->setup_offdiag();

    if (neighbor->ago == 0)
      evb_complex->update_pair_list(); // This is VERY important after pair-list got rebuilt

    #ifdef EVB_ENGINE_DEBUG
    if (cplx_map_changed)
      printf("cplx map changed at t = %d on proc %d ago = %d\n", update->ntimestep, comm->me, neighbor->ago);
    #endif
    evb_list->single_split();
  
    full_matrix->clear(true,vflag,true);

    if (ncomplex == 1) {
      evb_complex->cec->clear();
      evb_complex->cec_v2->clear();
    }
    int iextra_coupling = 0;

    #ifdef STATE_DECOMP
    int tmp[3];
    Divvy_Out_Partitions(tmp);
    int jfrom    = tmp[0];
    int jto      = tmp[1];
    int my_world = tmp[2];
    #endif

    if (natom > 0) atom->f = full_matrix->f_env; // point atom->f to full_matrix->f_env
    evb_list->change_list(ENV_LIST);

    #ifdef _RAPTOR_GPU
    // ensure that the force compute is done on the GPUs for the env compute part
    set_gpu_pair_offload(true);
    #endif

    #ifdef STATE_DECOMP
    if (my_world == 0) {
      TIMER_STAMP(EVB_Engine, compute__Env_compute_rspace_diagonal);
      if(!mp_verlet || mp_verlet->is_master==1) {
        compute_diagonal(vflag);
        if(EFIELD_flag) compute_efield(0,i+1);
      }
      TIMER_CLICK(EVB_Engine, compute__Env_compute_rspace_diagonal);
    } else {
      // Setup image charges if electrode pair_style
      /*
      if (strcmp(force->pair_style,"electrode") == 0) {
        PairElectrode * ptrPair = (PairElectrode*)(force->pair);
        ptrPair->et_compute_setup();
      } else if (strcmp(force->pair_style,"electrode/omp") == 0) {
        PairElectrodeOMP * ptrPair = (PairElectrodeOMP*)(force->pair);
        ptrPair->et_compute_setup();
      }
      */
    }

    TIMER_STAMP(EVB_Engine, compute__Env_compute_kspace);
    if(!mp_verlet || mp_verlet->is_master==0) {
      if (evb_kspace_cplx) evb_kspace_cplx->compute_env(vflag);
      if (evb_kspace) evb_kspace->compute_env(vflag);
    }
    TIMER_CLICK(EVB_Engine, compute__Env_compute_kspace);

    #ifdef _RAPTOR_GPU
    get_gpu_data();
    #endif

    if (my_world == 0) Force_Reduce();
    else               Force_Clear(1);
    evb_repulsive = nullptr;
    if (my_world == 0) {
      if (evb_kspace_cplx) full_matrix->save_ev_diag(MATRIX_ENV,vflag,1);
      else full_matrix->save_ev_diag(MATRIX_ENV,vflag);
    }
    #else // not defined STATE_DECOMP

    TIMER_STAMP(EVB_Engine, compute__Env_compute_rspace_diagonal);
    if(!mp_verlet || mp_verlet->is_master==1) {
      compute_diagonal(vflag);
      if (EFIELD_flag) compute_efield(0,i+1);
    }
    TIMER_CLICK(EVB_Engine, compute__Env_compute_rspace_diagonal);

    TIMER_STAMP(EVB_Engine, compute__Env_compute_kspace);
    if(!mp_verlet || mp_verlet->is_master==0) {
      if (evb_kspace) evb_kspace->compute_env(vflag);
      // for now also ensure that evb_kspace_cplx also has env_density_brick
      // To remove this step, need to modify EVB_PPPM::compute_cplx() and compute_exch()
      //   to handle only complex atoms
      if (evb_kspace_cplx) evb_kspace_cplx->compute_env(vflag);
    }
    TIMER_CLICK(EVB_Engine, compute__Env_compute_kspace);

    #ifdef _RAPTOR_GPU
    get_gpu_data();
    #endif
    Force_Reduce(); // ** AWGL ** //

    evb_repulsive = nullptr;
    full_matrix->save_ev_diag(MATRIX_ENV,vflag);

    #endif // STATE_DECOMP

    // Compute the pivot state
    if (evb_kspace) evb_kspace->energy = 0.0;
    if (evb_kspace_cplx) evb_kspace_cplx->energy = 0.0;
    if (natom > 0) atom->f = full_matrix->f_diagonal[MATRIX_PIVOT_STATE]; // point atom->f to full_matrix->f_diagonal[MATRIX_PIVOT_STATE]
    evb_list->change_list(EVB_LIST);

    #ifdef _RAPTOR_GPU
    // if set to be false, leave the force compute on the CPU for the remaining diagonal terms (pivot+states)
    if (partial_gpu_offload)
      set_gpu_pair_offload(false);
    else
      set_gpu_pair_offload(true);
    #endif

    #ifdef STATE_DECOMP
    if (jfrom == 0) {
    #endif

      TIMER_STAMP(EVB_Engine, compute__Eii_compute_diagonal__pivot);
      if (!mp_verlet || mp_verlet->is_master==1) {
        compute_diagonal(vflag);
        if (EFIELD_flag) compute_efield(1,i+1);
      }
      TIMER_CLICK(EVB_Engine, compute__Eii_compute_diagonal__pivot);

      #ifdef DLEVB_MODEL_SUPPORT
      if(!mp_verlet || mp_verlet->is_master==1) if(EVB14) compute_LJ14(vflag);
      #endif
      if (!mp_verlet || mp_verlet->is_master==1) compute_repulsive(vflag);

      TIMER_STAMP(EVB_Engine, compute__Eii_evb_kspace_compute_cplx__pivot);
      if (!mp_verlet || mp_verlet->is_master==0) {
        if (evb_kspace_cplx) {
          evb_kspace_cplx->pivot_state = 1;
          evb_kspace_cplx->compute_cplx(vflag);
        } else if (evb_kspace) evb_kspace->compute_cplx(vflag);
      }
      TIMER_CLICK(EVB_Engine, compute__Eii_evb_kspace_compute_cplx__pivot);

      #ifdef _RAPTOR_GPU
      get_gpu_data();
      #endif

      Force_Reduce(); // ** AWGL ** //
      if (evb_kspace_cplx) evb_matrix->save_ev_diag(MATRIX_PIVOT_STATE,vflag,1);
      else evb_matrix->save_ev_diag(MATRIX_PIVOT_STATE,vflag);

    #ifdef STATE_DECOMP
    }
    #endif

    evb_complex->save_avec(MATRIX_PIVOT_STATE);

    if (ncomplex == 1) {
      if (!mp_verlet || mp_verlet->is_master) {
        evb_complex->cec->compute_coc();
      }
    }

    // Compute other states
    //if (comm->me == 0) printf("t = %d: nstates %d\n", update->ntimestep, evb_complex->nstate);

    // real-space + repuslive + coupling + coc
    for (int j = 1; j < evb_complex->nstate; j++) { 

      int in_my_range = 1;
      #ifdef STATE_DECOMP
      if (flag_mp_state) {
        in_my_range = 0;
        if (j >= jfrom && j < jto) in_my_range = 1;
      }
      #endif

      // Build new state
      evb_complex->build_state(j);

      // Setup diagonal (force and repulsion) and calculate
      //   point atom->f to full_matrix->f_diagonal[j]
      if (natom > 0) atom->f = full_matrix->f_diagonal[j];

      if (in_my_range) {

        TIMER_STAMP(EVB_Engine, compute__Eii_compute_diagonal);
        if (!mp_verlet || mp_verlet->is_master == 1) {
          compute_diagonal(vflag);
          if (EFIELD_flag) compute_efield(1,i+1);
        }
        TIMER_CLICK(EVB_Engine, compute__Eii_compute_diagonal);

        #ifdef DLEVB_MODEL_SUPPORT
        if(!mp_verlet || mp_verlet->is_master==1) if(EVB14) compute_LJ14(vflag);
        #endif
        if(!mp_verlet || mp_verlet->is_master==1) compute_repulsive(vflag);

        #ifdef _RAPTOR_GPU
        get_gpu_data();
        #endif
        Force_Reduce(); // ** AWGL ** //

        // can move this kspace part out (see the commented section below)

        TIMER_STAMP(EVB_Engine, compute__Eii_evb_kspace_compute_cplx);
        if(!mp_verlet || mp_verlet->is_master==0) {
          if (evb_kspace_cplx) {
            // switch to non-pivot state
            evb_kspace_cplx->pivot_state = 0;
            evb_kspace_cplx->compute_cplx(vflag);
          } else if (evb_kspace) evb_kspace->compute_cplx(vflag);
        }
        TIMER_CLICK(EVB_Engine, compute__Eii_evb_kspace_compute_cplx);

        if (evb_kspace_cplx) full_matrix->save_ev_diag(j,vflag,1);
        else full_matrix->save_ev_diag(j,vflag);

      } // endif in_my_range

      #ifdef STATE_DECOMP
      full_matrix->e_extra[j][EOFF_ENE] = 0.0;
      #endif

      // Setup extra couplings and calculate

      if (evb_complex->extra_coupling[j] > 0) {

        // Save original Zundel
        int save_mol_A = evb_complex->molecule_A[j];

        // Build Zundels of extra couplings and compute
        for (int k = 0; k < evb_complex->extra_coupling[j]; k++) {
          evb_complex->extra_i[iextra_coupling] = j-k-1;
          evb_complex->extra_j[iextra_coupling] = j;
          evb_complex->molecule_A[j] = evb_complex->molecule_B[j-k-1];

          int type_A=evb_reaction->reactant_B[evb_complex->reaction[j-k-1]-1];
          int type_B=evb_reaction->product_B[evb_complex->reaction[j]-1];

          for (int l = 0; l < evb_reaction->nPair; l++) {
            if (evb_reaction->product_A[l]==type_A && evb_reaction->product_B[l]==type_B) {
              evb_offdiag = all_offdiag[l];
              break;
            }
          }

          // point atom->f to full_matrix->f_extra_coupling[iextra_coupling]
          if (natom > 0) atom->f = full_matrix->f_extra_coupling[iextra_coupling];
            SETUP_OFFDIAG_EXCH(extra,iextra_coupling);

          evb_offdiag->index = evb_matrix->ndx_extra+iextra_coupling*10;

          if (in_my_range) {
            TIMER_STAMP(EVB_Engine, compute__evb_offdiag__compute__extra);
            evb_offdiag->compute(vflag);
            TIMER_CLICK(EVB_Engine, compute__evb_offdiag__compute__extra);

            full_matrix->save_ev_offdiag(true,iextra_coupling,vflag);

            Force_Reduce(); // ** AWGL ** //
          }

          iextra_coupling++;
        }

        // Restore original Zundel
        evb_complex->molecule_A[j] = save_mol_A;

      } // endif (evb_complex->extra_coupling)

      // Setup off-diagonal and calculate
      evb_offdiag = all_offdiag[evb_complex->reaction[j]-1];

      // point atom->f to evb_matrix->f_off_diagonal[j]
      if (natom > 0) atom->f = evb_matrix->f_off_diagonal[j-1];
      SETUP_OFFDIAG_EXCH(off,j-1);

      if (in_my_range) {
        //evb_offdiag->finite_difference_test();
        evb_offdiag->index = evb_matrix->ndx_offdiag+10*(j-1);

        TIMER_STAMP(EVB_Engine, compute__evb_offdiag__compute);
        evb_offdiag->compute(vflag);  // this is where EVB_PPPM::compute_exch() is invoked by EVB_OffDiag::compute()
        TIMER_CLICK(EVB_Engine, compute__evb_offdiag__compute);

        full_matrix->save_ev_offdiag(false,j-1,vflag);

        Force_Reduce(); // ** AWGL ** //
      }

      // Compute COC
      if (ncomplex == 1) {
        if (!mp_verlet || mp_verlet->is_master) {
          evb_complex->cec->compute_coc();
        }
      }

      // Save status
      evb_complex->save_avec(j);

    } // forloop (j < nstate)
/*
    // kspace part
    for (int j = 1; j < evb_complex->nstate; j++) { 

      int in_my_range = 1;
      #ifdef STATE_DECOMP
      if (flag_mp_state) {
        in_my_range = 0;
        if (j >= jfrom && j < jto) in_my_range = 1;
      }
      #endif

      // Build new state
      evb_complex->build_state(j);

      // Setup diagonal (force and repulsion) and calculate
      //   point atom->f to full_matrix->f_diagonal[j]
      if (natom > 0) atom->f = full_matrix->f_diagonal[j];

      if (in_my_range) {

        TIMER_STAMP(EVB_Engine, compute__Eii_evb_kspace_compute_cplx);
        if(!mp_verlet || mp_verlet->is_master==0) {
          if (evb_kspace_cplx) {
            evb_kspace_cplx->pivot_state = 0;
            evb_kspace_cplx->compute_cplx(vflag);
          } else if (evb_kspace) evb_kspa74a2b6cfb7e47a999d73489bb2a55b35cdb838face->compute_cplx(vflag);
        }
        TIMER_CLICK(EVB_Engine, compute__Eii_evb_kspace_compute_cplx);

        if (evb_kspace_cplx) full_matrix->save_ev_diag(j,vflag,1,1);
        else full_matrix->save_ev_diag(j,vflag,0,1);
      } // endif in_my_range

      evb_complex->save_avec(j);
    }
*/
    #ifdef RELAMBDA
    // ** AWGL : For lambda replica exchange, scale by lambda here ** //
    if (lambda_flag) {
      Scale_Off_By_Lambda();
    }
    #endif

    #ifdef _RAPTOR_GPU
    // ensure that the force compute is done on the GPUs for this part
    set_gpu_pair_offload(true);
    #endif

    full_matrix->total_energy();

    #ifdef STATE_DECOMP
    // ** AWGL : Exchange info b/w state partitions so that everyone has same info ** //
    Communicate_Between_Partitions(evb_complex->nstate, evb_complex->nextra_coupling, vflag);
    #endif
    full_matrix->diagonalize();

    if (ncomplex == 1) {

      atom->f = lmp_f; // get back to the atom force
      full_matrix->compute_hellmann_feynman(vflag);

      #ifdef STATE_DECOMP
      // ** AWGL : Exchange the ground state force between the partitions ** //
      if (flag_mp_state > 2) Communicate_Force_Between_Partitions(atom->f);
      #endif

      if (evb_effpair) evb_effpair->compute_q_eff(false,false);

      if (bEffKSpace) {
        if (!bDelayEff) {
          double *q_save = atom->q;
          atom->q = evb_effpair->q;
          if(!mp_verlet || mp_verlet->is_master==0) evb_kspace->compute_eff(vflag);
          Force_Reduce_f(); // Reduce force across threaded data: atom->f
          atom->q = q_save;
        }
      }

      int pivot = evb_matrix->pivot_state;
      if (no_reaction) pivot = 0;

      evb_complex->load_avec(pivot);
      evb_complex->update_mol_map();
      evb_complex->update_list();

      if (pivot != 0) {
        nreact = 1;
        if (evb_kspace) qsqsum_sys = qsqsum_env+evb_complex->qsqsum;
      }

#ifdef EVB_ENGINE_DEBUG
      int ntmp = 0;
      MPI_Allreduce(&nreact, &ntmp, 1, MPI_INT, MPI_MAX, world);
      nreact = ntmp;
      if (nreact == 1) {
        need_update_pair_list = 1;
        if (comm->me == 0)
          printf("  pivot state = %d; nreact = %d at timestep %d\n",
            evb_matrix->pivot_state, nreact, update->ntimestep);
      } else {
        need_update_pair_list = 0;
      }
#endif

      // Full calculation of effective charges if using real-space for off-diagonals.
      if (flag_DIAG_QEFF && evb_effpair) evb_effpair->compute_q_eff(true,true);

      energy = full_matrix->ground_state_energy + full_matrix->e_env[EDIAG_POT];

    } else { // ncomplex != 1

      evb_complex->load_avec(0);
      evb_complex->update_mol_map();
      evb_complex->update_list();

    } // endif (ncomplex == 1)

    evb_list->change_list(SYS_LIST);

  } // endfor (i < ncomplex)

  TIMER_CLICK(EVB_Engine, compute);
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::post_process(int vflag)
{
    TIMER_STAMP(EVB_Engine, post_process);

    /****************************************************************/
    /***    Multi-complex subroutine   ******************************/
    /****************************************************************/

    if (ncenter > 1) {
      delete_overlap();
      sci_initialize(vflag);
      sci_iteration(vflag);
      sci_finalize(vflag);
    }

    qsqsum_sys = qsqsum_env;
    for(int i = 0; i < ncomplex; i++) qsqsum_sys += all_complex[i]->qsqsum;

    if (!mp_verlet || mp_verlet->is_master) {
      for(int i = 0; i < ncomplex; i++) {
        all_complex[i]->cec->compute();
        all_complex[i]->cec_v2->compute();
      }
    }

    if (universe->nworlds > 1) {
      if (comm->me==0 && universe->iworld == 0)
        evb_output->execute();
    }
    #ifdef RELAMBDA
    else if(lambda_flag) {
      if (comm->me==0 && lambda_flag != 2) {
        // AWGL don't print out if lambda attempted swap, lambda_flag = 2
        evb_output->execute();
      }
    }
    #endif
    else if(comm->me==0) evb_output->execute();

    TIMER_CLICK(EVB_Engine, post_process);
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::state_search()
{
  bool do_state_search = false;
  if (mp_verlet == nullptr) do_state_search = true;
  else if (mp_verlet && mp_verlet->is_master) do_state_search = true;

  if (do_state_search) {
    int me = comm->me;

    // Search states at each proc

    for (int i = 0; i < ncomplex; i++) {
       if (rc_rank[i ]== me) {
        all_complex[i]->rc_start = rc_molecule[i];
        all_complex[i]->search_state();
      }
    }
    // if(debug_sci_mp) {
    //   MPI_Barrier(universe->uworld);
    //   for(int i=0; i<universe->nprocs; i++) {
    //  if(universe->me == i) {
    //    fprintf(stdout,"(%i,%i)  rc_molecule in state_search on rank %i\n",universe->iworld,comm->me,i);
    //    for(int j=0; j<ncomplex; j++) fprintf(stdout,"(%i,%i)  j= %i  rc_molecule= %i\n",universe->iworld,comm->me,j,rc_molecule[j]);
    //  }
    //  MPI_Barrier(universe->uworld);
    //   }
    // }

    // Broadcast states from each proc

    for (int i = 0; i < ncomplex; i++) {

      // all states

      if (rc_rank[i] == me) all_complex[i]->pack_state();
      MPI_Bcast(&(all_complex[i]->buf_size),1,MPI_INT,rc_rank[i],world);
      MPI_Bcast(all_complex[i]->state_buf,all_complex[i]->buf_size,MPI_INT,rc_rank[i],world);
      if (rc_rank[i] != me) all_complex[i]->unpack_state();

      // other things
      MPI_Bcast(&(all_complex[i]->nextra_coupling),1,MPI_INT,rc_rank[i],world);
      MPI_Bcast(all_complex[i]->state_per_shell,evb_chain->max_shell,MPI_INT,rc_rank[i],world);
      all_complex[i]->compute_qsqsum();
    }
  }

  if (mp_verlet) {
    MPI_Status mpi_status;

    for (int i = 0; i < ncomplex; i++) {
      if (mp_verlet->rank_block == 1) {
        MPI_Send(&(all_complex[i]->buf_size), 1, MPI_INT, 0, 0, mp_verlet->block);
        MPI_Send(all_complex[i]->state_buf, all_complex[i]->buf_size, MPI_INT, 0, 0, mp_verlet->block);

        MPI_Send(&(all_complex[i]->nextra_coupling), 1, MPI_INT, 0, 0, mp_verlet->block);
        MPI_Send(all_complex[i]->state_per_shell, evb_chain->max_shell, MPI_INT, 0, 0, mp_verlet->block);

      } else if (mp_verlet->rank_block == 0) {
        MPI_Recv(&(all_complex[i]->buf_size), 1, MPI_INT, 1, 0, mp_verlet->block, &(mpi_status));
        MPI_Recv(all_complex[i]->state_buf, all_complex[i]->buf_size, MPI_INT, 1, 0, mp_verlet->block,&(mpi_status));
        all_complex[i]->unpack_state();

        MPI_Recv(&(all_complex[i]->nextra_coupling), 1, MPI_INT, 1, 0, mp_verlet->block, &(mpi_status));
        MPI_Recv(all_complex[i]->state_per_shell, evb_chain->max_shell, MPI_INT, 1, 0, mp_verlet->block,&(mpi_status));
        all_complex[i]->compute_qsqsum();
      }
    }
  }
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::count_rc()
{
  int* start_rc = evb_type->starting_rc;

  int ncenter_node = 0;
  ncenter = 0;

  for (int i = 0; i < natom; i++)
    if (mol_type[i]>0 && start_rc[mol_type[i]-1]==1 && mol_index[i]==1) ncenter_node++;
  MPI_Allreduce(&ncenter_node,&ncenter,1,MPI_INT,MPI_SUM,world);

  rc_molecule = (int*) memory->srealloc(rc_molecule,sizeof(int)*ncenter, "EVB_Engine:rc_molecule");
  rc_molecule_prev = (int*) memory->srealloc(rc_molecule_prev,sizeof(int)*ncenter,"EVB_Engine:rc_molecule_prev");
  rc_rank = (int*) memory->srealloc(rc_rank,sizeof(int)*ncenter, "EVB_Engine:rc_rank");

  // Initialize to indicate first step
  for (int i = 0; i < ncenter; i++) rc_molecule_prev[i] = -1;

  if (comm->me==0) {
    char line[200];
    sprintf(line,"[EVB]  %d reaction center(s) found.\n",ncenter);
    evb_output->print(line);
  }

  if (ncenter == 0) error->one(FLERR,"[EVB] No reaction center was found!");
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::locate_rc()
{
  int nlocal = atom->nlocal;
  int* start_rc = evb_type->starting_rc;

  // Scan all the molecule and label the RC

  memset(is_center_node,0,sizeof(int)*nmolecule);
  memset(is_center,0,sizeof(int)*nmolecule);

  for (int i = 0; i < nlocal; i++)
    if (mol_type[i] > 0 && start_rc[mol_type[i]-1] == 1 && mol_index[i] == 1) {
      is_center_node[atom->molecule[i]] = comm->me + 1;
    }

  MPI_Allreduce(is_center_node,is_center,nmolecule,MPI_INT,MPI_SUM,world);

  // If first step, then reduce the RC sorted by molecule ID
  if (rc_molecule_prev[0] == -1 || ncenter == 1) {
    int id = 0;
    for (int i = 1; i < nmolecule; i++)
      if (is_center[i]) {
        rc_rank[id] = is_center[i] - 1;
        rc_molecule_prev[id] = i;
        rc_molecule[id++]=i;
      }

    if (id != ncenter) {
      char errline[255];
      sprintf(errline,"[EVB] exception captured in EVB_Engine::locate_rc() -> id(%d)!=ncenter(%d)",id, ncenter);
      error->universe_one(FLERR,errline);
    }

  } else {
    // otherwise match to previous order of RC
    for (int i = 0; i < ncenter; i++) {
      rc_molecule[i] = rc_molecule_prev[i];
      rc_rank[i] = is_center[ rc_molecule_prev[i] ] - 1;
    }
  }
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::build_molecule_map()
{
  int nlocal = atom->nlocal;
  int* molecule= atom->molecule;

  int max1 = 0, max2 = 0;
  for (int i = 0; i < nlocal; i++) {
    max1 = MAX(max1,molecule[i]);
    max2 = MAX(max2,mol_index[i]);
  }

  MPI_Allreduce(&max1,&nmolecule,1,MPI_INT,MPI_MAX,world);
  MPI_Allreduce(&max2,&atoms_per_molecule,1,MPI_INT,MPI_MAX,world);
  atoms_per_molecule=MAX(atoms_per_molecule,evb_type->atom_per_molecule);

  nmolecule++;
  atoms_per_molecule++;

  memory->grow(molecule_map,nmolecule,atoms_per_molecule,"EVB_Engine:molecule_map");

  complex_molecule = (int*) memory->srealloc(complex_molecule,nmolecule*sizeof(int),"EVB_Engine:complex_molecule");
  max_coeff = (double*) memory->srealloc(max_coeff,nmolecule*sizeof(double),"EVB_Engine:max_coeff");

  is_center_node = new int [nmolecule];
  is_center = new int [nmolecule];
}

/* ----------------------------------------------------------------------*/

void EVB_Engine::update_molecule_map()
{
  int* molecule = atom->molecule;

  for (int i = 0; i < nmolecule; i++) {
    memset(&(molecule_map[i][0]), -1, sizeof(int)*(atoms_per_molecule));
    molecule_map[i][0] = 0;
  }

  for (int i = 0; i < natom; i++) {
    int m = molecule[i];
    int n = mol_index[i];
    if (molecule_map[m][n] == -1) {
      molecule_map[m][0]++;
      molecule_map[m][n]=i;
    }
  }
}

/* ---------------------------------------------------------------------- */

void EVB_Engine::clear_virial()
{
  if (force->pair) memset(force->pair->virial,0,sizeof(double)*6);
  if (force->bond) memset(force->bond->virial,0,sizeof(double)*6);
  if (force->angle) memset(force->angle->virial,0,sizeof(double)*6);
  if (force->dihedral) memset(force->dihedral->virial,0,sizeof(double)*6);
  if (force->improper) memset(force->improper->virial,0,sizeof(double)*6);
  if (force->kspace) memset(force->kspace->virial,0,sizeof(double)*6);
}

/* ---------------------------------------------------------------------- */

#ifdef _RAPTOR_GPU
extern double lmp_gpu_forces(double **f, double **tor, double *eatom, double **vatom,
                             double *virial, double &ecoul, int &err_flag);
#endif

/* ---------------------------------------------------------------------- */

void EVB_Engine::compute_diagonal(int vflag)
{
  TIMER_STAMP(EVB_Engine, compute_diagonal);

  int save_ago = neighbor->ago;

  TIMER_STAMP(EVB_Engine, compute__pair_compute);
  if (force->pair) {

    if (bHybridPair) neighbor->ago = 0;
    else neighbor->ago = save_ago;

    #ifdef _RAPTOR_GPU
    bool offload = get_gpu_pair_offload();
    if (fix_gpu && offload) {
      if (!partial_gpu_offload || evb_list->env_list_changed == 1) neighbor->ago = 0; // Set to zero to force copying neighbor list from host to GPU
                                                   // to overwrite previously loaded neighbor list
    }                                              // lmp_gpu_forces() => LJCLMF.compute() => BaseChargeT::compute() => BaseChargeT::reset_nbors()     
    #endif                                     
    force->pair->compute(true,vflag);

    #ifdef _RAPTOR_GPU
    if (fix_gpu && offload) neighbor->ago = save_ago; // Restore
    #endif
  }
  TIMER_CLICK(EVB_Engine, compute__pair_compute);

  TIMER_STAMP(EVB_Engine, compute__bond_compute);
  if (force->bond) {
    if (bHybridBond) neighbor->ago = 0;
    else neighbor->ago = save_ago;
    force->bond->compute(true,vflag);
  }

  if (force->angle) {
    if (bHybridAngle) neighbor->ago = 0;
    else neighbor->ago = save_ago;
    force->angle->compute(true,vflag);
  }

  if (force->dihedral) {
    if (bHybridDihedral) neighbor->ago = 0;
    else neighbor->ago = save_ago;
    force->dihedral->compute(true,vflag);
  }

  if (force->improper) {
    if (bHybridImproper) neighbor->ago = 0;
    else neighbor->ago = save_ago;
    force->improper->compute(true,vflag);
  }
  TIMER_CLICK(EVB_Engine, compute__bond_compute);

  neighbor->ago = save_ago;

  TIMER_CLICK(EVB_Engine, compute_diagonal);
}

/* ---------------------------------------------------------------------- */

void EVB_Engine::compute_repulsive(int vflag)
{
  evb_repulsive = nullptr;

  int status_id = evb_complex->current_status;
  int mol_id = evb_complex->molecule_B[status_id];
  int target_atom = molecule_map[mol_id][1];

  if(target_atom==-1 || target_atom>=atom->nlocal) return;

/* temporary fix to multiple repulsive interactions

  int target_etp = mol_type[target_atom];
  for(int i=0; i<nrepulsive; i++)
    if(all_repulsive[i]->etp_center == target_etp)
    {
      evb_repulsive = all_repulsive[i];
      break;
    }

  if (!evb_repulsive) return;

  evb_repulsive->center_mol_id = mol_id;
  evb_repulsive->compute(vflag);
*/

  int target_etp = mol_type[target_atom];
  double e = 0.0;

  for (int i = 0; i < nrepulsive; i++) {
    if (all_repulsive[i]->etp_center == target_etp) {
      evb_repulsive = all_repulsive[i];
      evb_repulsive->center_mol_id = mol_id;
      evb_repulsive->compute(vflag);
      e += evb_repulsive->energy;
    }
  }

  if (evb_repulsive) evb_repulsive->energy = e;
}

/* ---------------------------------------------------------------------- */

#ifdef DLEVB_MODEL_SUPPORT

void EVB_Engine::compute_LJ14(int vflag)
{
  // Load evb_lj_pair arrays
  evb_list->change_pairlist(EVB_LJ_LIST);

  double save_eng_vdwl = force->pair->eng_vdwl;
  double save_eng_coul = force->pair->eng_coul;
  double *save_virial  = force->pair->virial;

  // Turn off Coulomb scaling for 1-4s
  double save_special_coul = force->special_coul[3];
  force->special_coul[3] = 0.0;

  // Calculate LJ 1-4 terms
  if (force->pair) force->pair->compute(true,vflag);

  // Accumulate energies and virial
  force->pair->eng_vdwl+= save_eng_vdwl;
  force->pair->eng_coul+= save_eng_coul;
  if (vflag) for(int i=0; i<6; i++) force->pair->virial[i]+= save_virial[i];

  // Turn Coulomb scaling back on
  force->special_coul[3] = save_special_coul;

  // Restore evb_pair arrays
  evb_list->change_pairlist(EVB_LIST);
}

#endif

/* ---------------------------------------------------------------------- */

void EVB_Engine::Force_Reduce()
{
#if defined (_OPENMP)
  // ** AWGL ** //
  // Need to reduce forces across threaded data accumulated in lmp_f
  // and store it into the place where atom->f in pointing
  // Also, reset lmp_f

  int nthreads = comm->nthreads;
  int nall3 = 3*(atom->nlocal + atom->nghost);
  double *fv = &(atom->f[0][0]);
  double *lv = &(lmp_f[0][0]);
  int i, t;
  #pragma omp parallel for shared(fv, lv, nall3, nthreads) private(i, t)
  for (i= 0; i < nall3; ++i) {
    for (t = 0; t < nthreads; ++t) {
      fv[i] += lv[nall3*t + i];
      lv[nall3*t + i] = 0.0;
    }
  }
#endif
}

/* ---------------------------------------------------------------------- */

void EVB_Engine::Force_Reduce_f()
{
#if defined (_OPENMP)
  // ** AWGL ** //
  // Need to reduce forces across threaded data accumulated in atom->f
  // Also, reset slave threads

  int nthreads = comm->nthreads;
  int nall3 = 3*(atom->nlocal + atom->nghost);
  double *fv = &(atom->f[0][0]);
  int i, t;
  #pragma omp parallel for shared(fv, nall3, nthreads) private(i, t)
  for (i = 0; i < nall3; ++i) {
    for (t = 1; t < nthreads; ++t) {
      fv[i] += fv[nall3*t + i];
      fv[nall3*t + i] = 0.0;
    }
  }
#endif
}

/* ---------------------------------------------------------------------- */

void EVB_Engine::Force_Clear(int which)
{
  if (which == 1) {
    // AWGL : clear force of current atom pointer
    int nall = atom->nlocal + atom->nghost;
    for (int ii = 0; ii < nall; ++ii) {
      atom->f[ii][0] = 0.0;
      atom->f[ii][1] = 0.0;
      atom->f[ii][2] = 0.0;
    }
  } else if (which == 2) {
    // AWGL : clear out lmp_f pointer for all threads
    #if defined (_OPENMP)
    int nthreads = comm->nthreads;
    #else
    int nthreads = 1;
    #endif
    int nall3 = 3*(atom->nlocal + atom->nghost);
    double *lv = &(lmp_f[0][0]);
    #if defined (_OPENMP)
    int i, t;
    #pragma omp parallel for shared(lv, nall3, nthreads) private(i, t)
    for (i = 0; i < nall3; ++i)
      for (t = 0; t < nthreads; ++t)
        lv[nall3*t + i] = 0.0;
    #else
    int i;
    for (i = 0; i < nall3; ++i) lv[i] = 0.0;
    #endif
  }
}

/* ---------------------------------------------------------------------- */

void EVB_Engine::check_for_complex_atoms()
{
  // Set special atom flags
  const int nall = atom->nlocal + atom->nghost;
  has_complex_atom = 0;
  for (int i = 0; i < nall; ++i) {
    if (complex_atom[i]) {
      has_complex_atom = 1;
      break;
    }
  }
}

/* ---------------------------------------------------------------------- */
#ifdef BGQ
 void EVB_Engine::get_memory()
 {
    uint64_t shared, persist, heapavail, stackavail, stack, heap, guard, mmap;

    Kernel_GetMemorySize(KERNEL_MEMSIZE_SHARED,    &shared);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_PERSIST,   &persist);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &heapavail);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_STACK,     &stack);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAP,      &heap);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_GUARD,     &guard);
    Kernel_GetMemorySize(KERNEL_MEMSIZE_MMAP,      &mmap);

    double s = 1.0 / 1024.0 / 1024.0;
    fprintf(stdout,"(%i,%i) Allocated heap:  %.2f MB, avail. heap:  %.2f MB\n",universe->iworld, comm->me,
      (double)heap*s,  (double)heapavail*s);
    //    fprintf(stdout,"(%i,%i) Allocated stack: %.2f MB, avail. stack: %.2f MB\n",universe->iworld, comm->me,
    //      (double)stack*s, (double)stackavail*s);
    //    fprintf(stdout,"(%i,%i) Memory:  shared: %.2f MB, persist: %.2f MB, guard: %.2f MB, mmap: %.2f MB\n",
    //      universe->iworld, comm->me, (double)shared*2, (double)persist*s, (double)guard*s, (double)mmap*s);
  }
#endif

/* -------------------------------------------------------------------------------- */

#ifdef RELAMBDA
void EVB_Engine::Scale_Off_By_Lambda()
{
  // *** AWGL: Scale energies and forces of off-diagonal couplings by lambda *** //

  EVB_Matrix* mtx = full_matrix; // alias

  for(int i=0; i<evb_complex->nstate-1; i++) {
    mtx->e_offdiag[i][EOFF_ENE]       *= offdiag_lambda;
    mtx->e_offdiag[i][EOFF_ARQ]       *= offdiag_lambda;
    mtx->e_offdiag[i][EOFF_VIJ_CONST] *= offdiag_lambda;
    mtx->e_offdiag[i][EOFF_VIJ]       *= offdiag_lambda;
  }
  for(int i=0; i<evb_complex->nextra_coupling; i++)
    mtx->e_extra[i][EOFF_ENE] *= offdiag_lambda;

  int i;
  for(i=0; i<natom; ++i) {
    for (int istate=0; istate<evb_complex->nstate; ++istate) {
      // ** f_off_diagonal force ** //
      mtx->f_off_diagonal[istate][i][0] *= offdiag_lambda;
      mtx->f_off_diagonal[istate][i][1] *= offdiag_lambda;
      mtx->f_off_diagonal[istate][i][2] *= offdiag_lambda;
     }
    // ** f_extra_coupling force ** //
    for (int icouple=0; icouple<evb_complex->nextra_coupling; ++icouple) {
      mtx->f_extra_coupling[icouple][i][0] *= offdiag_lambda;
      mtx->f_extra_coupling[icouple][i][1] *= offdiag_lambda;
      mtx->f_extra_coupling[icouple][i][2] *= offdiag_lambda;
    }
  }

 }
#endif

/* -------------------------------------------------------------------------------- */

#ifdef _RAPTOR_GPU
void EVB_Engine::get_gpu_data()
{
  if (!fix_gpu) return;
  /*
  if (strcmp(force->pair_style, "lj/charmm/coul/long/gpu") == 0) {
    if (((PairLJCharmmCoulLongGPU*)force->pair)->offload == false) return;
  }
  */
  bool offload = get_gpu_pair_offload();
  if (!offload) return;

  TIMER_STAMP(EVB_Engine, get_gpu_data);

  timer->stamp();
  double lvirial[6];
  for (int i = 0; i < 6; i++) lvirial[i] = 0.0;
  int err_flag = 0;
  double my_eng = lmp_gpu_forces(atom->f, atom->torque, force->pair->eatom,
                                 force->pair->vatom, lvirial,
                                 force->pair->eng_coul, err_flag);

  force->pair->eng_vdwl += my_eng;
  force->pair->virial[0] += lvirial[0];
  force->pair->virial[1] += lvirial[1];
  force->pair->virial[2] += lvirial[2];
  force->pair->virial[3] += lvirial[3];
  force->pair->virial[4] += lvirial[4];
  force->pair->virial[5] += lvirial[5];

  //if (force->pair->vflag_fdotr) force->pair->virial_fdotr_compute();
  timer->stamp(Timer::PAIR);

  TIMER_CLICK(EVB_Engine, get_gpu_data);
}

/* -------------------------------------------------------------------------------- */

void EVB_Engine::set_gpu_pair_offload(bool offload)
{
  if (strcmp(force->pair_style, "lj/charmm/coul/long/gpu") == 0) {
    ((PairLJCharmmCoulLongGPU*)force->pair)->offload = offload;
  }
  if (strcmp(force->pair_style, "lj/cut/coul/long/gpu") == 0) {
    ((PairLJCutCoulLongGPU*)force->pair)->offload = offload;
  }
}

/* -------------------------------------------------------------------------------- */

bool EVB_Engine::get_gpu_pair_offload()
{
  bool offload = true;
  if (strcmp(force->pair_style, "lj/charmm/coul/long/gpu") == 0) {
    return ((PairLJCharmmCoulLongGPU*)force->pair)->offload;
  }
  if (strcmp(force->pair_style, "lj/cut/coul/long/gpu") == 0) {
    return ((PairLJCutCoulLongGPU*)force->pair)->offload;
  }
  return offload;
}

#endif

/* -------------------------------------------------------------------------------- */
/* Apply electric field to effective charges                                        */
/*  -- based on FixEfield::post_force()                                             */
/* -------------------------------------------------------------------------------- */

void EVB_Engine::compute_efield(int is_state, int cplx_id)
{
  double **f = atom->f;
  double **x = atom->x;
  double *q = atom->q;
  imageint * image = atom->image;
  double unwrap[3];

  if (is_state == 0) {

    // Environment contribution

    double e = 0.0;

    #pragma omp parallel for default(shared) private(unwrap) reduction(+:e)
    for (int i = 0; i < atom->nlocal; ++i)
      if (!complex_atom[i]) {
        const double fz = q[i] * efieldz;
        f[i][2] += fz;

        domain->unmap(x[i], image[i], unwrap);
        e -= fz * unwrap[2];
      }

    force->pair->eng_coul += e;

    MPI_Reduce(&e, &efield_energy_env, 1, MPI_DOUBLE, MPI_SUM, 0, world);

  } else {

    // Complex contribution

    double e = 0.0;

    #pragma omp parallel for default(shared) private(unwrap) reduction(+:e)
    for(int i = 0; i < atom->nlocal; ++i)
      if (complex_atom[i] == cplx_id) {
        const double fz = q[i] * efieldz;
        f[i][2] += fz;

        domain->unmap(x[i], image[i], unwrap);
        e -= fz * unwrap[2];
      }

    force->pair->eng_coul += e;
  }
}

/* -------------------------------------------------------------------------------- */

NeighList* EVB_Engine::get_pair_list()
{
  if (bHybridPair) {
    PairHybrid *pair = (PairHybrid*)(force->pair);
    return pair->styles[0]->list;
  } else
    return force->pair->list;
}
