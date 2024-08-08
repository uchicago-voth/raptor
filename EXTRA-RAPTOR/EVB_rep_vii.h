/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifdef EVB_MODULE_REPULSIVE

MODULE_REPULSIVE(VII,EVB_Rep_Vii)

#else


#ifndef EVB_REP_VII_H
#define EVB_REP_VII_H

#include "EVB_repul.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Rep_Vii : public EVB_Repulsive
{
  public:
    EVB_Rep_Vii(class LAMMPS *, class EVB_Engine*);
    virtual ~EVB_Rep_Vii();
 
  public:
    virtual int data_rep(char*, int*, int, int);
    virtual void compute(int);
    virtual int checkout(int*);
   
  public:
    double vii;
};
 
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif

#endif
