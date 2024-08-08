/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifdef EVB_MODULE_REPULSIVE

MODULE_REPULSIVE(Hydronium_FR_Table,EVB_Rep_Hydronium_FR_Table)

#else


#ifndef EVB_REP_HYDRONIUM_FR_TABLE_H
#define EVB_REP_HYDRONIUM_FR_TABLE_H

#include "EVB_repul.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Rep_Hydronium_FR_Table : public EVB_Repulsive
{
 public:
  EVB_Rep_Hydronium_FR_Table(class LAMMPS *, class EVB_Engine*);
  virtual ~EVB_Rep_Hydronium_FR_Table();
  
 public:
  virtual int data_rep(char*, int*, int, int);
  
  virtual void compute(int);
  virtual void sci_compute(int);
  
  virtual int checkout(int*);

  virtual void scan_potential_surface();
  
 public:
  int atp_OW;
  int bEVB3;
  
  double e_oo,e_ho;
  
 private:
  double dfx,dfy,dfz;
  
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
  double cutoff[2];
  
  void read_table(Table *, char *, char *);
  void param_extract(Table *, char *);
  void bcast_table(Table *);
  void spline_table(Table *);
  void null_table(Table *);
  void free_table(Table *);
  void compute_table(Table *);
  void spline(double *, double *, int, double, double, double *);
  double splint(double *, double *, double *, int, double);
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif

#endif
