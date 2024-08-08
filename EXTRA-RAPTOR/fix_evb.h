/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
   ------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(evb,FixEVB)
  
#else
  
#ifndef FIX_EVB_H
#define FIX_EVB_H
  
#include "fix.h"
  
namespace LAMMPS_NS {
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
  
class FixEVB : public Fix 
{
public:
  FixEVB(class LAMMPS *, int, char **);
  virtual ~FixEVB();
    
  void set_force_compute(int);

  // Interface setting
  int setmask();
  
  // Basic functions
  void init();
  void setup(int);
  void min_setup(int);
  void min_pre_exchange();

  // Function for [update] entries 
  void post_force(int);
  void pre_force(int);
  void min_post_force(int);

  // Function for [compute] entries
  virtual double compute_scalar();

#ifdef RELAMBDA
  // AWGL: For replica exchange with lambda off-diagonal scaling
  double compute_array(int, int);        
  void modify_fix(int, double *, char *); 
#endif
   
  // Function for [atom] entries
  double memory_usage();

  void   grow_arrays(int);
  void   copy_arrays(int, int, int);
  int    pack_forward_comm(int, int *, double *, int, int *);
  void   unpack_forward_comm(int, int, double *);

  virtual int    pack_exchange(int, double *);
  virtual int  unpack_exchange(int, double *);
  int    pack_restart(int, double *);
  void   unpack_restart(int, int);
    
  int    size_restart(int);
  int    maxsize_restart();
    
public:
  class EVB_Engine *Engine;
  int *mol_type, *mol_index;
  double *charge;
  int size_array;
   
  void pack_dump();
  
  /* ----------------------------- */

  int FDM_flag; // flag for finite difference tests
};
  
  /*------------------------------------------------------------------------*/
  /*------------------------------------------------------------------------*/
  /*------------------------------------------------------------------------*/
  
}

#endif

#endif
