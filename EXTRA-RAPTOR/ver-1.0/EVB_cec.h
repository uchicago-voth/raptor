/* ----------------------------------------------------------------------
 EVB Package Code
 For Voth Group
 Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_CEC_H
#define EVB_CEC_H

#include "pointers.h"
#include "EVB_pointers.h"
#include "EVB_source.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_CEC : protected Pointers, protected EVB_Pointers
{
 public:
  EVB_CEC(class LAMMPS*, class EVB_Engine*, class EVB_Complex*);
  ~EVB_CEC();
    
  class EVB_Complex* cplx;
  
  double r_cec[3]; 
  double ref[3];

  int natom_coc[MAX_STATE];
  double qsum_coc[MAX_STATE];
  
  double r_coc[MAX_STATE][3];
  int* id_coc[MAX_STATE];
  double* qi_coc[MAX_STATE];

  double array1[MAX_STATE][3];
  double array2[MAX_STATE][3];
  double deltaE[MAX_STATE];
  double q_array1[MAX_STATE];
  double q_array2[MAX_STATE];
  double x_factor[MAX_STATE];

  void clear();
  void compute_coc();
  void compute();
  void broadcast();
  void decompose_force(double *);

  void partial_C_N4(double *force);
  void partial_C_N3(double *force);
  void partial_C_N2(double *force);

  void partial_C_N3_omp(double *force); // AWGL
  
};

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
    
}

# endif
