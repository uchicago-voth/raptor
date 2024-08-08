/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Tianying Yan, Yuxing Peng
------------------------------------------------------------------------- */

#ifdef EVB_MODULE_OFFDIAG

MODULE_OFFDIAG(Vij,EVB_OffDiag_VIJ)

#else


#ifndef EVB_OFFDIAGONAL_VIJ
#define EVB_OFFDIAGONAL_VIJ

#include "EVB_offdiag.h"

//#define OUTPUT_3BODY

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_OffDiag_VIJ : public EVB_OffDiag
{
public:
  EVB_OffDiag_VIJ(class LAMMPS *, class EVB_Engine*);
  virtual ~EVB_OffDiag_VIJ();
  
public:
  virtual int data_offdiag(char*,int*,int,int);
  virtual void compute(int);
  virtual void sci_setup(int);
  virtual void sci_compute(int);
  virtual void sci_setup_mp() {};
  
public:
  int istate;

};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif

#endif
