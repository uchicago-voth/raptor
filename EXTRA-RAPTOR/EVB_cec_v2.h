/* ----------------------------------------------------------------------
 EVB Package Code
 For Voth Group
 Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_CEC_V2_H
#define EVB_CEC_V2_H

#include "pointers.h"
#include "EVB_pointers.h"
#include "EVB_source.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

#define MAX_XATOM 100
#define MAX_HATOM 200

class EVB_CEC_V2 : protected Pointers, protected EVB_Pointers
{
 public:
  EVB_CEC_V2(class LAMMPS*, class EVB_Engine*, class EVB_Complex*);
  ~EVB_CEC_V2();
    
  class EVB_Complex* cplx;

  void clear();
  void compute_coc();
  void compute();
  void decompose_force(double *);
  
  double r_cec[3];

  int numX, numH;
  int XAtom[MAX_XATOM], HAtom[MAX_HATOM];
  double dev_xatom[MAX_XATOM][3][3], dev_hatom[MAX_HATOM][3][3];
  
  static int set(char*, int*, int,int);
  static int *type;
  static double *weight;
  static double RSW;
  static double DSW;
};

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
    
}

# endif
