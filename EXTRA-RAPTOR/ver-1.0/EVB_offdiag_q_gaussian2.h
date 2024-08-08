/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight
     Based on code by Tianying Yan and Yuxing Peng
------------------------------------------------------------------------- */

#ifdef EVB_MODULE_OFFDIAG

MODULE_OFFDIAG(Q_Gaussian2,EVB_OffDiag_Q_Gaussian2)

#else


#ifndef EVB_OFFDIAGONAL_Q_GAUSSIAN2
#define EVB_OFFDIAGONAL_Q_GAUSSIAN2

#include "EVB_offdiag.h"

//#define OUTPUT_3BODY

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_OffDiag_Q_Gaussian2 : public EVB_OffDiag
{
 public:
  EVB_OffDiag_Q_Gaussian2(class LAMMPS *, class EVB_Engine*);
  virtual ~EVB_OffDiag_Q_Gaussian2();
  
 public:
  virtual int data_offdiag(char*,int*,int,int);
  virtual int checkout(int*);
  virtual void compute(int);
  virtual void sci_setup(int);
  virtual void sci_compute(int);
  virtual void sci_setup_mp();

  virtual void mp_post_compute(int);

  void init_exch_chg();
  void resume_chg();
  
  void cal_g_term_sym() {};
  void cal_f_term_sym();
  void cal_force_sym(int);
  
  void cal_g_term_asym() {};
  void cal_f_term_asym() {};
  void cal_force_asym(int) {};

 public:
  int atom_A_Rq[3], mol_A_Rq[3], index_A_Rq[3];
  double *x_D, *x_A, *x_H;
  double dr_DH[3], dr_AH[3];
  
  /********************************************/
  /******* Parameters set for A(R_DA) *********/
  /********************************************/
  
  /** A(R_DA) = c1 * EXP[-c2 * Q]  **/
  double _c1, _c2;
  
  /******************************************/
  /******************************************/
  /******************************************/
  
  int   etp_A_exch, etp_B_exch; // type containing exchanged-charge
  int     n_A_exch,   n_B_exch; // # of atoms in types
  double *q_A_exch,  *q_B_exch; // # exchanged charges
  double *q_A_save,  *q_B_save; // # saved atomic charges
  double qsum_exch, qsum_save, qsqsum_exch, qsqsum_save;
  
 public:
  double f_Q;
  double df_Q, df_QD, df_QA;
  
 public:
  int **map;
  int natom;
  int icomplex;
  int istate;
  int mol_A;
  int mol_B;

#ifdef OUTPUT_3BODY

 public:
  FILE* fp;
  int timestep;
  int center;
#endif

};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif

#endif
