/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   The JACOBI diagonalization is written by Wim R. Cardoen 08/20/2009
------------------------------------------------------------------------- */


#ifndef EVB_MATRIX_SCI_H
#define EVB_MATRIX_SCI_H

#include "EVB_matrix.h"

#define SCI_EDIAG_NITEM     4
#define SCI_EDIAG_POT       0
#define SCI_EDIAG_VDW       1
#define SCI_EDIAG_COUL      2
#define SCI_EDIAG_KSPACE    3

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_MatrixSCI : public EVB_Matrix
{
 public:
  EVB_MatrixSCI(class LAMMPS *, class EVB_Engine *);
  virtual ~EVB_MatrixSCI();
  
  class EVB_Complex *cplx;

  double E,dE;

  double sci_allreduce[MAX_STATE][SCI_EDIAG_NITEM];
  double sci_e_diagonal[MAX_STATE][SCI_EDIAG_NITEM];
  double sci_e_offdiag[MAX_STATE];
  double sci_e_extra[MAX_STATE];

  int first_time_setup;
  
 public:

  void setup();
  void clear(bool,bool,bool);

  void setup_mp();                  // Master partition calls this and only allocates 
  void clear_mp(bool,bool,bool);    //  force arrays for owned complexes.

  void sci_total_energy();
  void sci_save_ev_diag(int,bool);
  void sci_save_ev_offdiag(bool,int,bool);
  void compute_hellmann_feynman();
  void copy_ev(bool);
  void copy_force();
  void accumulate_force();

  void sci_comm_energy_mp(int);
  void copy_ev_full(bool);
  void copy_force_full();
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
