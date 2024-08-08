/* ----------------------------------------------------------------------
 EVB Package Code
 For Voth Group
 Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_API_H
#define EVB_API_H

#include "fix_evb.h"
#include "EVB_engine.h"
#include "EVB_complex.h"
#include "EVB_cec.h"
#include "EVB_cec_v2.h"

#include "modify.h"

namespace LAMMPS_NS
{
    void EVB_GetFixObj(Modify*,FixEVB**);
    void EVB_GetCplx(FixEVB*,int,EVB_Complex**);
    void EVB_GetCEC(EVB_Complex*,double*);
    void EVB_GetCECV2(EVB_Complex*,double*);
    void EVB_PutCEC(EVB_Complex*,double*);
    void EVB_PutCECV2(EVB_Complex*,double*);
}


#endif
