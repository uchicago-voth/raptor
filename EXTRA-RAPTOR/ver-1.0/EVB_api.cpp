/* ----------------------------------------------------------------------
 EVB Package Code
 For Voth Group
 Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "EVB_engine.h"
#include "string.h"
#include "EVB_api.h"

using namespace LAMMPS_NS;

void LAMMPS_NS::EVB_GetFixObj(Modify* modify, FixEVB **pFix)
{
    (*pFix)==NULL;
    
    for(int i=0; i<modify->nfix; i++)
    {
        Fix* pf = modify->fix[i];
	if(strcmp(pf->style,"evb")==0)
        {
            (*pFix)=(FixEVB*)pf;
            break;
        }
    }
}

void LAMMPS_NS::EVB_GetCplx(FixEVB* pFix, int cplx_id, EVB_Complex** pCplx)
{
    (*pCplx)=pFix->Engine->all_complex[cplx_id];
}

void LAMMPS_NS::EVB_GetCEC(EVB_Complex* pC, double* x)
{
    double* r_cec = pC->cec->r_cec;

    x[0] = r_cec[0];
    x[1] = r_cec[1];
    x[2] = r_cec[2];
}

void LAMMPS_NS::EVB_GetCECV2(EVB_Complex* pC, double* x)
{
    double* r_cec = pC->cec_v2->r_cec;

    x[0] = r_cec[0];
    x[1] = r_cec[1];
    x[2] = r_cec[2];
}

void LAMMPS_NS::EVB_PutCEC(EVB_Complex* pC, double* f)
{
    pC->cec->decompose_force(f);
}

void LAMMPS_NS::EVB_PutCECV2(EVB_Complex* pC, double* f)
{
    pC->cec_v2->decompose_force(f);
}
