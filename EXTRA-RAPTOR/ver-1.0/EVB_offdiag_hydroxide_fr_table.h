/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris (based on EVB_offdiag_pt)
------------------------------------------------------------------------- */

#ifdef EVB_MODULE_OFFDIAG

MODULE_OFFDIAG(Hydroxide_FR_Table,EVB_OffDiag_Hydroxide_FR_Table)

#else


#ifndef EVB_OFFDIAGONAL_HYDROXIDE_FR_TABLE
#define EVB_OFFDIAGONAL_HYDROXIDE_FR_TABLE

#include "EVB_offdiag.h"

//#define OUTPUT_3BODY

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_OffDiag_Hydroxide_FR_Table : public EVB_OffDiag
{
public:
  EVB_OffDiag_Hydroxide_FR_Table(class LAMMPS *, class EVB_Engine*);
  virtual ~EVB_OffDiag_Hydroxide_FR_Table();
  
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
  
  void cal_gtheta_term_sym();
  void cal_g_term_sym();
  void cal_f_term_sym();
  void cal_force_sym(int);
  
  void cal_g_term_asym();
  void cal_f_term_asym();
  void cal_force_asym(int);

public:
  int atom_A_Rq[3], mol_A_Rq[3], index_A_Rq[4];
  double *x_D, *x_A, *x_H, *x_HH;
  double dr_DH[3], dr_AH[3], dr_DA[3], dr_OHbond[3];
  
  /******************************************/
  /****** Parameters set for A(R,q) *********/
  /******************************************/
  
  /***** synonyms ***************************/
  double _a;   // alpha
  double _b;   // beta
  
  /***** For symmetric/evb3 *****************/
  double _g;   // g
  double _k;   // k
  
  double _P0;  // P
  double _P1;  // P'
  
  double _D;   // D_oo
  double _R;   // R0_oo
  double _r;   // r0_oo

  double _theta_alpha; // g(theta)_alpha
  double _theta_theta0; // g(theta)_theta0
  
  
  /***** For asymmetric/amc *****************/
  double _rs;     // r0_sc
  double _l;      // lamda
  double _RDA;    // R0_DA
  
  double _C;      // C
  
  double _aDA;    // aDA
  double _bDA;    // bDA
  double _cDA;    // cDA
  
  double _ga  ;   // gammar
  double _eps;    // epsinal
  
  /******************************************/
  /******************************************/
  /******************************************/

  int    type_A_Rq;
  
  int   etp_A_exch, etp_B_exch; // type containing exchanged-charge
  int     n_A_exch,   n_B_exch; // # of atoms in types
  double *q_A_exch,  *q_B_exch; // # exchanged charges
  double *q_A_save,  *q_B_save; // # saved atomic charges
  double qsum_exch, qsum_save, qsqsum_exch, qsqsum_save;
  
public:
  double f_R, g_q, dg_q;
  double df_R, r_da;
  double df_O, df_H;

  // HOO angle scaling factor
  int hoo_type;
  double g_theta, dg_theta, g_theta_cos;

  // for asym func
  double r_sc, q[3], sumq;
  double fa, fb, ftanh;
  
public:
  int **map;
  int natom;
  int icomplex;
  int istate;
  int mol_A;
  int mol_B;
  
 private:

  struct Table {
    int ninput,rflag,fpflag,match,ntablebits;
    int nshiftbits,nmask;
    double rlo,rhi,fplo,fphi,cut;
    double *rfile,*efile,*ffile;
    double *e2file,*f2file;
    double innersq,delta,invdelta,deltasq6;
    double *rsq,*drsq,*e,*de,*f,*df,*e2,*f2;
  };
  
  int me;
  
  int tabstyle,tablength;
  int ntables;
  Table *tables;
  double cutoff[1];
  
  void read_table(Table *, char *, char *);
  void param_extract(Table *, char *);
  void bcast_table(Table *);
  void spline_table(Table *);
  void null_table(Table *);
  void free_table(Table *);
  void compute_table(Table *);
  void spline(double *, double *, int, double, double, double *);
  double splint(double *, double *, double *, int, double);

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
