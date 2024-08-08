/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_REACTION_H
#define EVB_REACTION_H

#include "pointers.h"
#include "EVB_pointers.h"
#include "EVB_source.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/


class EVB_Reaction : protected Pointers, protected EVB_Pointers
{
  public:
    EVB_Reaction(class LAMMPS *, class EVB_Engine *);
  virtual ~EVB_Reaction();
  
  public:  
  struct EVB_Path
  {
    int atom_count[3];
    int *moving_part;
    int *first_part;
    int *second_part;
  };
  
  int nPair;
  char** name; 
  int *backward;
  int *reactant_A;
  int *reactant_B;
  int *product_A;
  int *product_B;
  int *nPath;  
  EVB_Path **Path;
  
  int data_reaction(char*, int*, int,  int);
  int get_reaction(char*);
  void setup();
  void change_atom(int,int,int);
  
  _EVB_DEFINE_AVEC_POINTERS;
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
