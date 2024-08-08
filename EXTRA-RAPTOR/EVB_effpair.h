/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_EFF_PAIR_H
#define EVB_EFF_PAIR_H

#include "pointers.h"
#include "EVB_pointers.h"
#include "EVB_source.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/


class EVB_EffPair : protected Pointers, protected EVB_Pointers
{
  public:
    EVB_EffPair(class LAMMPS *, class EVB_Engine *);
    virtual ~EVB_EffPair();

    
    double *q;
    double **lj1,**lj2;
    int *is_exch;
    int nexch, *iexch;
    double *q_exch, *qexch;
    int max_atom;
    
    double *pre_ecoul, *pre_fcoul;
    double *pre_ecoul_exch, *pre_fcoul_exch;
    double *r2inv, *r6inv;
    bool *cut_coul, *cut_lj;
    int max_pair;

    double energy, evdw, ecoul;
    
    void init();
    void setup();
    void compute_para();
    void compute_para_qeff();
    void compute_q_eff(bool,bool);
    void compute_q_eff_offdiag(bool);
    void compute_vdw_eff();
    void compute_vdw_eff_omp();
    void pre_compute();
    void pre_compute_omp();
    void compute_cplx(int);
    void compute_cplx_supp(int);
    void init_exch(bool,int);
    void compute_exch(int);
    void compute_exch_supp(int);
    void compute_finter(int);
    void compute_finter_supp(int);
    void compute_fenv(int);
    void compute_fenv_supp(int);

    void compute_finter_mp(int);
    void compute_finter_supp_mp(int);
    void compute_fenv_mp(int);
    void compute_fenv_supp_mp(int);

    int p_style;
    void compute_pair_mp(int);   // mp replacement for force->pair->compute()
    void compute_pair_mp_1(int); // mp replacement for lj/cut/coul/long
    void compute_pair_mp_2(int); // mp replacement for lj/charmm/coul/long; not yet coded.
    void compute_pair_mp_3(int); // mp replacement for table/lj/cut/coul/long
    void compute_pair_mp_4(int); // mp_replacement for gulp/coul/long
    void compute_pair_mp_5(int); // mp_replacement for electrode
    void compute_pair_mp_6(int); // mp_replacement for electrode/omp

    void setup_pair_mp();

    double **ptrLJ1, **ptrLJ2;
    double **ptrA, **ptrB;
    double cut_coulsq, cut_ljsq;

    class PairTableLJCutCoulLong * ptrPair_1; // pointer to lj/cut/coul/long object
    class PairGulpCoulLong       * ptrPair_4; // pointer to gulp/coul/long   object
    class PairElectrode          * ptrPair_5; // pointer to electrode        object
    class PairElectrodeOMP       * ptrPair_6; // pointer to electrode/omp    object

    inline int sbmask(int j) {
    return j >> SBBITS & 3; }

    // To help avoid confusion, let's make q_offdiag private
 private:
    double * q_offdiag;
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
