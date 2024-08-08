/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight and Gard Nelson
------------------------------------------------------------------------- */

#ifdef EVB_MODULE_REPULSIVE

MODULE_REPULSIVE(Expon2,EVB_Rep_Expon2)

#else


#ifndef EVB_REP_EXPON2_H
#define EVB_REP_EXPON2_H

#include "EVB_repul.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Rep_Expon2 : public EVB_Repulsive
{
  public:
    EVB_Rep_Expon2(class LAMMPS *, class EVB_Engine*);
    virtual ~EVB_Rep_Expon2();
 
  public:
    virtual int data_rep(char*, int*, int, int);

    virtual void compute(int);
    virtual void sci_compute(int);

    virtual int checkout(int*);

    virtual void scan_potential_surface() {};

   
  public:
    int num_pairs;
    double Vii_const;

    int ** setflag_pair;
    int * setflag_type;

    double ** cutsq;
    double ** _a;
    double ** _b;
    double ** _r0;
};
 
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif

#endif
