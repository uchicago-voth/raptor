/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_CHAIN_H
#define EVB_CHAIN_H

#include "pointers.h"
#include "EVB_pointers.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Chain : protected Pointers, protected EVB_Pointers
{
 public:
  EVB_Chain(class LAMMPS*, class EVB_Engine*);
  ~EVB_Chain();
  
  void grow_chain(int n);
  int data_chain(char*, int*,int,int);
  
  int type_count, chain_total;

  int *index;
  int *count;
  int *host, *target, *client, *shell_limit, *reaction, *path;
  double *distance_limit;
  int max_shell;
};

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
