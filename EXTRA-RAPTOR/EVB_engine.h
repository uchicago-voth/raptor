/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_ENGINE_H
#define EVB_ENGINE_H

#include "pointers.h"

#define ENGINE_INDICATOR_COMPUTE    0
#define ENGINE_INDICATOR_INITIALIZE 1
#define ENGINE_INDICATOR_ITERATION  2
#define ENGINE_INDICATOR_FINALIZE   3
#define ENGINE_INDICATOR_SCREEN     4

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Engine : protected Pointers
{ 
 public:

  // Charactor string for names of I/O files

  char cfg_name[255];   // .cfg - models and parameters
  char out_name[255];   // .out - simulation output
  char top_name[255];   // .top - initial topology, when using [read_data] 
  FILE * fp_cfg_out;    // output of processed cfg file
  
  // Physical properties
  int natom;            // # of total atoms = atom->nlocal + atom->ghost
  double virial[6];     // system virial from MS-EVB
  double energy;        // total energy of system
  double cplx_energy;   // complex energy = E(cplx-cplx) + E(cplx-env)
  double inter_energy;  // inter-energy among complexes, if multi-center
  double env_energy;    // enviroment energy : E(env-env)
  
  // RC(reaction center) information
  int ncenter;           // # of reaction centers
  int nreact;            // # of reactions happened in current step
  int *is_center_node;   // Array: # of molecules: is a center : local
  int *is_center;        // Array: # of molecules: is a center : global
  int *rc_molecule;      // Array: # of centers: molecule ID for each RC
  int *rc_molecule_prev; // Array: # of centers: molecule ID for each RC from previous step
  int *rc_rank;          // Array: # of centers: rank ID for each RC

  // EVB Object
  class EVB_KSpace      *evb_kspace;
  class EVB_Type        *evb_type;
  class EVB_Chain       *evb_chain;
  class EVB_List        *evb_list;    
  class EVB_Reaction    *evb_reaction;
  class EVB_Output      *evb_output;
  class EVB_EffPair     *evb_effpair;
  class EVB_Timer       *evb_timer;

  // EVB Object pointers for array members
  class EVB_Complex     *evb_complex;
  class EVB_Matrix      *evb_matrix;
  class EVB_Repulsive   *evb_repulsive;
  class EVB_OffDiag     *evb_offdiag;
  class EVB_MatrixFull  *full_matrix;	
  
  // EVB Object arrays, based on ncenter
  class EVB_Complex    **all_complex;
  class EVB_MatrixSCI  **all_matrix;
  class EVB_Repulsive  **all_repulsive;
  class EVB_OffDiag    **all_offdiag;
  
  // Topology information
  int* mol_type;         // molecule type
  int* mol_index;        // molecule index
  double* charge;        // effective charges
  
  // for repulsive terms
  int nrepulsive;        // # of repulsive terms
  
  // for molecule map    
  int nmolecule;         // # of molecules
  int atoms_per_molecule;// max # of atoms of a molecule
  int **molecule_map;    // molecule topology
                         // molecule_map[i][0] # of domain atoms in moleclue i
                         // molecule_map[i][j] # domain index of atom j in molecule i
  
  // for EVB complex map
  int ncomplex;          // # of EVB complex
  int *complex_molecule; // Array: # of molecules: complex ID this molecule belongs, 0 is env atoms
  int *complex_atom;     // Array: max # of atoms: complex ID of this atoms, 0 is env atoms
  int *kernel_atom;      // Array: max # of atoms: complex ID is it is a kernel atom
  int complex_atom_size; // max # of atoms in this domain
  
  // for kspace
  bool bEffKSpace;                   // If use effective charge
  bool bDelayEff;                    // If kspace is delayed to called, e.g. fix_umbrella
  double qsqsum_env, qsqsum_sys;     // Stored sum of q^2 for env and system
  
  // for state-search
  int bRefineStates;     // If refining states, by EVB2 state-search
  int bExtraCouplings;   // If calculating extra couplings 

  // for effective VDW parameters in SCI
  double *max_coeff;      //
  int *complex_pos;       //
  int flag_EFFPAIR_SUPP;  // Calculate supplamental interactions (beyond LJ+COUL) with pair->single() type functions.

  int SCI_KSPACE_flag;    // flag to choose method for KSPACE forces in SCI

  // for multi-center iteration
  int ncircle;           //
  int nstep;             //

  // disable reaction

  bool no_reaction;

  // lammps pointers;
  double  **lmp_f;       // store original per-atom force array
  
  // hybrid force flag
  bool bHybridPair;
  bool bHybridBond;
  bool bHybridAngle;
  bool bHybridDihedral;
  bool bHybridImproper;

  // Pair List
  class NeighList* get_pair_list();

  // ** AWGL ** //
  void Force_Reduce();   // Reduce OpenMP threaded force data
  void Force_Reduce_f(); // " for atom->f
  void Force_Clear(int); // Clear OpenMP threaded force data or atom->f pointer
  int has_complex_atom;  // Flag for my rank has an atom belonging to complex
  int has_exch_chg;      // Flag for my rank has an exchange charge 
  void check_for_special_atoms(); // function to determine has_complex_atom

#ifdef RELAMBDA
  int lambda_flag;       // Flag for doing replica exchange lambda
  double offdiag_lambda; // lambda scalar for off diagonal coupling
  void Scale_Off_By_Lambda(); // Scales off-diagonal energies and forces by lambda
#endif

#ifdef STATE_DECOMP
  int  flag_mp_state;    // off or on
  int * comm_list;       // communication list between partitions
  MPI_Comm force_comm;   // Collective force communicator
  int group_rank;        // Rank in force group 
  int group_root;        // Universe rank of root of force group
  void Communicate_Between_Partitions(int, int, int);   // Communicates b/w partitions
  void Communicate_Force_Between_Partitions(double **); // Communicates given force b/w partitions
  void Divvy_Out_Partitions(int*);                      // Determines state partitions
#endif

 public:
  EVB_Engine(class LAMMPS *, char*, char*, char*);
  virtual ~EVB_Engine();
  
 public:
  
  // basic functions 
  void construct();
  void init();    
  
  void execute(int);
  
  void pre_process(int);
  void compute(int);
  void post_process(int);
  
  void count_rc();
  void locate_rc();
  
  void build_molecule_map();    
  void update_molecule_map();
  
  void state_search();
  void compute_diagonal(int);
  void compute_repulsive(int);
  
  // SCI-MS-EVB functions
  void delete_overlap();
  void sci_iteration(int);
  void sci_initialize(int);
  void sci_finalize(int);
  
  // Methods to compute PPPM forces in SCI simulations
  int engine_indicator;
  void sci_pppm_polar();

  // operations on system pointers and variables
  void clear_virial();
  void increase_atomic_space();

  // debug functions
  void output_matrix();
  void finite_difference_force();  
  void finite_difference_virial();
  void finite_difference_amplitude();
  void finite_difference_cec();
 
  // data functions
  void init_kspace();
  void data_top();
  int  data_offdiag(char*, int*, int, int);
  int  data_repulsive(char*, int*, int, int);
  int  data_extension(char*, int*, int, int);
  
  // FULL-EFFECTIVE-CHARGE-METHOD
  int flag_ACC;
  int flag_DIAG_QEFF;
  double qsqsum_save;

  // State-Partition methods
  class MP_Verlet *mp_verlet;
  class MP_Verlet_SCI * mp_verlet_sci;

  // SCI_MP functions

  void state_search_sci_mp() {};    // replacement for state_search()
  void post_process_sci_mp(int) {}; // replacement for post_process()
  void compute_diagonal_mp(int) {}; // replacement for compute_diagonal (environment calculation)
  void compute_sci_mp(int) {};      // replacement for compute()
  void delete_overlap_mp() {};      // replacement for delete_overlap()
  void sci_initialize_mp(int) {};   // replacement for sci_initialize()
  void sci_iteration_mp(int) {};    // replacement for sci_iteration()
  void sci_finalize_mp(int) {};     // replacement for sci_finalize()

  void sci_comm_evec_mp() {};       // Sync eigenvectors on all partitions
  void sci_comm_avec() {};          // Sync avec for all states on all partitions
  void sci_comm_avec_pivot() {};    // Sync avec for pivot state on all partitions
  int old_size_gtotal_int;
  int old_size_gtotal_double;
  int *comm_avec_buf_int;
  double *comm_avec_buf_double;

  void setup_lb_mp() {};            // Assign work for all partitions
  void sci_comm_lb_mp_1() {};       // Communicate full_matrix energies between partitions in compute_sci_mp()
  void sci_comm_lb_mp_2() {};       // Communicate all_matrix[i] energies between partitions in initialize_sci_mp()
  void sci_comm_lb_mp_3() {};       // Communicate all_matrix[i] sci energies between partitions in iteration_sci_mp()

  int *lb_tasklist;              // Indicates which partition will evaluate state
  int *lb_cplx_master;           // Indicates the partition that owns state 0 of complex
  int *lb_cplx_split;            // Indicates if a complex is split across partitions
  int *lb_cplx_owned;            // Indicates if partition owns part of a complex
  MPI_Comm *lb_cplx_block;       // Sub-communicators for partitions that own a complex

  int *lb_num_owners;            // Number of partitions that currently own a complex
  int *lb_cplx_owner_list;       // List of partitions that currently own a complex

  int * num_tasks_per_part;      // Number of tasks assigned to partition
  int * cplx_owner_list;         // List of partitions that own a complex
  int * num_part_per_complex;    // Number of partitions assigned to complex
  int * num_states_per_complex;  // Largest number of states from a complex assigned to any partition

  // tmp arrays for sci_comm_lb_mp_#() functions
  int max_comm_ek;
  double * comm_ek;
  double * comm_ek2;

  // tmp arrays for sci_comm_avec() and sci_comm_avec_pivot()
  int * start_indx_int;
  int * start_indx_double;
  int * size_total_int;
  int * size_total_double;

  // tmp arrays for state_search_sci_mp()
  int * buf_size_all;
  int * buf_nextra_all;
  int * buf_shell_all;
  int * start_indx;
  int max_tmp_state_buf;
  int * tmp_state_buf;

  int count_comm_create; // DEBUG frequency of comm creation in setup_lb_mp()
  int count_comm_total;

  int lb_comm_update;            // Indicates if new comms should be created or old ones used.

  void sci_comm_cplx_map();      // Stores natom_cplx and nghost_cplx for all complexes.
  int *sci_cplx_count;

  int max_list_Vij;
  double * list_Vij;

  int max_cc2;
  double * cc2;

  bool debug_sci_mp;

  // Hamiltonian screening
  bool bscreen_hamiltonian;
  double screen_minP;  // tolerance for diagonals
  double screen_minP2; // tolerance for off-diagonals

  int screen_max_list;
  int * screen_del_list;
  int * screen_delsum_list;
  int * screen_important_list;

  void screen_states(int);
  void screen_delete_states();
  void screen_finalize();
  void screen_compute_diagonal(int);

  // Multipro_sci support for Hamiltonian screening
  int * screen_del_list_count;
  int * screen_del_list_global;

  void screen_states_mp(int);
  void screen_delete_states_mp(int);
  void screen_delete_states_comm_mp();
  
#ifdef DLEVB_MODEL_SUPPORT
  bool EVB14;
  void compute_LJ14(int);
#endif

#ifdef BGQ
  void get_memory();
#endif

#ifdef _RAPTOR_GPU
  class Fix* fix_gpu;
  void get_gpu_data();
#endif
  int evb_full_neigh; // Indicate if full neighbor list built (default is half)
                      // This affects calculation of Vij_ex_short in off-diagonals: scale energies/forces by 0.5.

  // Use overlap of eigenvectors as convergence criteria
  double sci_overlap_tol;

  // electric field along z-axis
  int EFIELD_flag; // flag for electric field along z-axis
  double efieldz;
  double efield_energy_env;
  void compute_efield(int,int);
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
  
}

#endif
