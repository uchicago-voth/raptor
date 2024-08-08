/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_OffDiag_H
#define EVB_OffDiag_H

#include "pointers.h"
#include "EVB_pointers.h"
#include "EVB_source.h"
#include "error.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_OffDiag : protected Pointers, protected EVB_Pointers
{
  public:
    static int max_nexch;
    
  public:
    EVB_OffDiag(class LAMMPS *, class EVB_Engine*);
    virtual ~EVB_OffDiag();

    char name[25];

    int n_exch_chg;  // total number of exchange charges, i.e. Zundel in water
    int n_exch_chg_local;  // number of local exchange charges
    int *exch_list;        // local exchange atom list
    int *is_exch_chg;         // if atom belongs to exchange charge
    int is_Vij_ex;
    int size_exch_chg;
    int *index;

    /* Used for effective charges */
    int *ptr_nexch;
    int *iexch;
    double *qexch;

    double A_Rq;     // A(R,q)
    double Vij,Vij_const,Vij_ex,Vij_ex_short,Vij_ex_long; 

  public:  
    virtual void compute(int) = 0;
    virtual int data_offdiag(char*,int*,int,int) = 0;
    virtual int checkout(int*) { return 0;} ;

    double cut, cut_sq, kappa;
    double exch_chg_cut(int);
    double exch_chg_debye(int);
    double exch_chg_long(int);
    double exch_chg_wolf(int);
    double exch_chg_cgis(int);
    
    // ** AWGL : OpenMP threaded exch_chg routines ** //
    double exch_chg_long_omp(int);
    template <int VFLAG, int NEWTON_PAIR> double exch_chg_long_omp_eval();
    double exch_chg_cut_omp(int);
    template <int VFLAG, int NEWTON_PAIR> double exch_chg_cut_omp_eval();
    double exch_chg_debye_omp(int);
    template <int VFLAG, int NEWTON_PAIR> double exch_chg_debye_omp_eval();
    double exch_chg_wolf_omp(int);
    template <int VFLAG, int NEWTON_PAIR> double exch_chg_wolf_omp_eval();
    double exch_chg_cgis_omp(int);
    template <int VFLAG, int NEWTON_PAIR> double exch_chg_cgis_omp_eval();
   
    virtual void sci_setup(int) = 0;
    virtual void sci_compute(int) = 0;
    virtual void sci_setup_mp() = 0;
    
    virtual void mp_post_compute(int) {};
   
    void finite_difference_test();
	
    double energy;
    double virial[6];
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
