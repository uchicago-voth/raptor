/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_REPULSIVE_H
#define EVB_REPULSIVE_H

#include "pointers.h"
#include "EVB_pointers.h"
#include "EVB_source.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Repulsive : protected Pointers, protected EVB_Pointers
{
  public:
    EVB_Repulsive(class LAMMPS *, class EVB_Engine*);
    virtual ~EVB_Repulsive();
  
  public:
    virtual void compute(int) =0;
    virtual int data_rep(char*, int*, int, int) =0;
    virtual int checkout(int*) {return 0;};

    virtual void sci_compute(int) {};

    void finite_difference_test();
    void scan_potential_surface() {};
	
    double energy;
    double virial[6];

    char name[25];
    int etp_center,center_mol_id;
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
