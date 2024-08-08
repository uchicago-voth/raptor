/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   The JACOBI diagonalization is written by Wim R. Cardoen 08/20/2009
------------------------------------------------------------------------- */


#ifndef EVB_MATRIX_H
#define EVB_MATRIX_H

#include "pointers.h"
#include "EVB_pointers.h"
#include "EVB_complex.h"

#define EDIAG_NITEM      8
#define EDIAG_POT        0
#define EDIAG_VDW        1
#define EDIAG_COUL       2
#define EDIAG_BOND       3
#define EDIAG_ANGLE      4
#define EDIAG_DIHEDRAL   5
#define EDIAG_IMPROPER   6
#define EDIAG_KSPACE     7

#define EOFF_NITEM      5
#define EOFF_ENE        0
#define EOFF_ARQ        1
#define EOFF_VIJ        2
#define EOFF_VIJ_CONST  3
#define EOFF_VIJ_LONG   4

#define MATRIX_ENV -1
#define MATRIX_PIVOT_STATE 0

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Matrix : protected Pointers, protected EVB_Pointers
{
 public:
  EVB_Matrix(class LAMMPS *, class EVB_Engine *);
  virtual ~EVB_Matrix();
  
 public:
  double sci_e_env[EDIAG_NITEM];  
  double energy[EDIAG_NITEM /* e_env */ + MAX_STATE*(EDIAG_NITEM+1 /* potential + repulsive */ )];
  double energy_allreduce[EDIAG_NITEM /* e_env */ + MAX_STATE*(EDIAG_NITEM+1 /* potential + repulsive */ )];
  int size_e, size_ediag;
  
  double v[6];
  double v_env[6];
  double v_diagonal[MAX_STATE][6];
  double v_offdiag[MAX_STATE-1][6];
  double v_extra[MAX_EXTRA][6];

  double *e_env;
  double *e_diag;
  double *e_diagonal[MAX_STATE];
  double *e_repulsive;
  
  double e_offdiag[MAX_STATE-1][EOFF_NITEM];
  double e_extra[MAX_EXTRA][EOFF_NITEM];
  
  int ndx_offdiag[MAX_STATE*10];
  int ndx_extra[MAX_EXTRA*10]; 
 
  double **f_env;
  double ***f_diagonal;
  double ***f_off_diagonal;
  double ***f_extra_coupling;

  int natom, nstate, nextra, max_state, max_atom, max_extra;
  int *list;

  double **hamilton;
  double **unitary;
  double eigen_value[MAX_STATE];
  int ground_state,pivot_state;
  double ground_state_energy;
	
  void save_ev_diag(int,bool);
  void save_ev_offdiag(bool,int,bool);
  void total_energy();
  void diagonalize();

  int max_comm_ek;
  double * comm_ek;

  /* ----------------------------------------------------------------------
     The JACOBI diagonalization is written by Wim R. Cardoen 08/20/2009
  ------------------------------------------------------------------------- */
  int num_rot;
  void jacobi(double **, int, double *, double **, int *);
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
