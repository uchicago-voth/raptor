/* ----------------------------------------------------------------------
 EVB Package Code
 For Voth Group
 Written by Yuxing Peng
------------------------------------------------------------------------- */
#ifndef EVB_COMPLEX_H
#define EVB_COMPLEX_H

#include "pointers.h"
#include "EVB_pointers.h"
#include "EVB_source.h"

#define MAX_SHELL 5
#define MAX_STATE 200
#define MAX_EXTRA 100

#define SETUP_OFFDIAG_EXCH(a,b) evb_offdiag->ptr_nexch = evb_complex->nexch_##a+b; \
     evb_offdiag->iexch = evb_complex->iexch_##a[b];\
     evb_offdiag->qexch = evb_complex->qexch_##a[b]; \
     evb_complex->nexch_##a[b] = 0
	
#define GET_OFFDIAG_EXCH(cplx) int* nexch_off = cplx->nexch_off; \
     int* nexch_extra = cplx->nexch_extra; \
     int** iexch_off = cplx->iexch_off; \
     int** iexch_extra = cplx->iexch_extra; \
     double** qexch_off = cplx->qexch_off; \
     double** qexch_extra = cplx->qexch_extra; \
	 int *extra_i = cplx->extra_i; \
	 int *extra_j = cplx->extra_j;
			
namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Complex : protected Pointers, protected EVB_Pointers
{
public:
  EVB_Complex(class LAMMPS*, class EVB_Engine*);
  ~EVB_Complex();
  
  int id;
  int rc_start;
  int nstate;
  int natom_cplx;
  int* cplx_list;
  int nghost_cplx;
  int nlocal_cplx;

  int natom;
  int nextra_coupling;
  int state_per_shell[MAX_SHELL];
  
  int shell[MAX_STATE];
  int parent_id[MAX_STATE];
  int molecule_A[MAX_STATE];
  int molecule_B[MAX_STATE];
  int reaction[MAX_STATE];
  int path[MAX_STATE];  
  int extra_coupling[MAX_STATE];
  
  double distance[MAX_STATE];
  
  double qsqsum;
  
  void delete_state(int);
  void delete_multiplestates(int[], int);
  void exchange_state(int,int);
  void search_state();
  void refine_state(int);
  void build_state(int);
  void sci_build_state(int);
  void compute_qsqsum();

  double Cs[MAX_STATE];
  double Cs2[MAX_STATE];

  double Cs_prev[MAX_STATE];
  double Cs_overlap;

  int state_buf[(MAX_STATE*7+2)*sizeof(int)+sizeof(double)*2];
  int buf_size;
  int rc_etype;
  
  void pack_state();
  void unpack_state();
  
  void save_avec(int);
  void load_avec(int);
  void setup_avec();
  void setup_offdiag();
  void update_list();
  void update_pair_list();
  void update_pair_list_omp(); // AWGL: OpenMP threaded version
  void update_bond_list();
  void update_mol_map();
  void update_mol_map_omp(); // OpenMP threaded version
  void build_cplx_map();
  void setup();

  // for state-search
  static int set(char*, int*, int, int);
  static int ss_do_refine;  // If refining states, by EVB2 state-search
  static int ss_do_extra;   // If calculating extra couplings
 
  struct STATE_INFO
  {
    int *type, *mol_type, *mol_index;
    double *q;
    int *molecule;
    int *num_bond;
    int **bond_type,**bond_atom;
    int *num_angle;
    int **angle_type;
    int **angle_atom1,**angle_atom2,**angle_atom3;
    int *num_dihedral;
    int **dihedral_type;
    int **dihedral_atom1,**dihedral_atom2,**dihedral_atom3,**dihedral_atom4;
    int *num_improper;
    int **improper_type;
    int **improper_atom1,**improper_atom2,**improper_atom3,**improper_atom4;
	
    double qsqsum_cplx;
    int nbonds, nangles, ndihedrals, nimpropers;
  };
  
  STATE_INFO* status;
  int current_status;
  int status_nstate, status_natom; 
  
  int max_offdiag, max_extra;
  int *extra_i, *extra_j;
  int *nexch_off, *nexch_extra;
  int **iexch_off, **iexch_extra;
  double **qexch_off, **qexch_extra;
	
  class EVB_CEC *cec;
  class EVB_CEC_V2 *cec_v2;
  
  _EVB_DEFINE_AVEC_POINTERS;
  
#ifdef DLEVB_MODEL_SUPPORT 
  void delete_shell_states(int);
#endif

  inline int sbmask(int j) {
    return j >> SBBITS & 3;
  }
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
}

#endif
