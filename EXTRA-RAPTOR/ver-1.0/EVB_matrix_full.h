/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   The JACOBI diagonalization is written by Wim R. Cardoen 08/20/2009
------------------------------------------------------------------------- */


#ifndef EVB_MATRIX_FULL_H
#define EVB_MATRIX_FULL_H

#include "EVB_matrix.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_MatrixFull : public EVB_Matrix
{
 public:
  EVB_MatrixFull(class LAMMPS *, class EVB_Engine *);
  virtual ~EVB_MatrixFull();

 public:

  void setup();
  void clear(bool,bool,bool);
  void clear(bool,bool,bool,bool);
  void compute_hellmann_feynman(int);
  void compute_hellmann_feynman_omp(int); // ** AWGL ** //
 
  /* ----------------------------------------------------------------------
     The JACOBI diagonalization is written by Wim R. Cardoen 08/20/2009
  ------------------------------------------------------------------------- */
  void jacobi(double **, int, double *, double **, int *);
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
