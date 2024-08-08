/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight and Gard Nelson
------------------------------------------------------------------------- */

#ifdef EVB_MODULE_REPULSIVE

MODULE_REPULSIVE(Expon,EVB_Rep_Expon)

#else


#ifndef EVB_REP_EXPON_H
#define EVB_REP_EXPON_H

#include "EVB_repul.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Rep_Expon : public EVB_Repulsive
{
  public:
    EVB_Rep_Expon(class LAMMPS *, class EVB_Engine*);
    virtual ~EVB_Rep_Expon();
 
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
