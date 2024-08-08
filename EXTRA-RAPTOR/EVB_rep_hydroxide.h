/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris
------------------------------------------------------------------------- */

#ifdef EVB_MODULE_REPULSIVE

MODULE_REPULSIVE(Hydroxide,EVB_Rep_Hydroxide)

#else

#ifndef EVB_REP_HYDROXIDE_H
#define EVB_REP_HYDROXIDE_H

#include "EVB_repul.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Rep_Hydroxide : public EVB_Repulsive
{
  public:
    EVB_Rep_Hydroxide(class LAMMPS *, class EVB_Engine*);
    virtual ~EVB_Rep_Hydroxide();
 
  public:
    virtual int data_rep(char*, int*, int, int);

    virtual void compute(int);
    virtual void sci_compute(int);

    virtual int checkout(int*);

    virtual void scan_potential_surface();


  public:
    int atp_OW;
    double B;
    double b1;
    double b2;
    int bEVB3;
    double d_OO;
    double C;
    double c1;
    double d_OH;
    double cutoff_OO[2],cutoff_HO[2];

    double e_oo,e_ho;

    double oo_cutoff_1; // (rc-rs)^(-3) for oo
    double ho_cutoff_1; // (rc-rs)^(-3) for ho

  private:
    double switching(double,double, double, double);
    double dswitching(double,double, double, double);
    double sw;
    double dfx,dfy,dfz;
};
 
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif

#endif
