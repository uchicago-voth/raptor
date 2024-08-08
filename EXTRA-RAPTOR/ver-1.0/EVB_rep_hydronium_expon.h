/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight and Gard Nelson
      based on EVB_rep_hydronium written by Yuxing Peng
------------------------------------------------------------------------- */

#ifdef EVB_MODULE_REPULSIVE

MODULE_REPULSIVE(Hydronium_Expon,EVB_Rep_Hydronium_Expon)

#else


#ifndef EVB_REP_HYDRONIUM_EXPON_H
#define EVB_REP_HYDRONIUM_EXPON_H

#include "EVB_repul.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Rep_Hydronium_Expon : public EVB_Repulsive
{
  public:
    EVB_Rep_Hydronium_Expon(class LAMMPS *, class EVB_Engine*);
    virtual ~EVB_Rep_Hydronium_Expon();
 
  public:
    virtual int data_rep(char*, int*, int, int);

    virtual void compute(int);
    virtual void sci_compute(int);

    virtual int checkout(int*);

    // Original MS-EVB3 definition

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
    double Bb1,Bb2,Cc1;

    double oo_cutoff_1; // (rc-rs)^(-3) for oo
    double oo_cutoff_2; // (3*rc-rs) for oo
    double oo_cutoff_3; // (rc+rs) for oo
    double oo_cutoff_4; // (rc*rs) for oo

    double ho_cutoff_1; // (rc-rs)^(-3) for ho
    double ho_cutoff_2; // (3*rc-rs) for ho
    double ho_cutoff_3; // (rc+rs) for ho
    double ho_cutoff_4; // (rc*rs) for ho

  private:
    double switching(double,double, double, double);
    double dswitching(double,double, double, double);
    double sw;
    double dfx,dfy,dfz;

    // Supplemental pairwise interactions
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
