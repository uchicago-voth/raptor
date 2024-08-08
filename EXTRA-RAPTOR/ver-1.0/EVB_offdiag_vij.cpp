/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Tianying Yan, Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "EVB_complex.h"
#include "EVB_engine.h"
#include "EVB_type.h"
#include "EVB_offdiag_vij.h"
#include "EVB_kspace.h"
#include "EVB_source.h"

#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "universe.h"
#include "comm.h"
#include "update.h"

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_OffDiag_VIJ::EVB_OffDiag_VIJ(LAMMPS *lmp, EVB_Engine *engine) : EVB_OffDiag(lmp,engine)
{

}

/* ---------------------------------------------------------------------- */

EVB_OffDiag_VIJ::~EVB_OffDiag_VIJ()
{

}

/* ---------------------------------------------------------------------- */

int EVB_OffDiag_VIJ::data_offdiag(char *buf, int* offset, int start, int end)
{
  int t = start;
    
  FILE * fp = evb_engine->fp_cfg_out;

  // Input Vij information
  Vij_const = atof(buf+offset[t++]);
  evb_engine->flag_DIAG_QEFF = 1;
  
  if(universe->me == 0) {
    fprintf(fp,"\n   Off-diagonal definition: VIJ= VIJ_CONST.\n");
    fprintf(fp,"\n   VIJ_CONST= %f\n",Vij_const);
  }

  return t;
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_VIJ::compute(int vflag)
{
  Vij = Vij_const;
  energy = Vij;
}

void EVB_OffDiag_VIJ::sci_setup(int vflag)
{
  Vij = Vij_const;
  energy = Vij;  
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_VIJ::sci_compute(int vflag)
{
    istate = evb_complex->current_status;  
    int* parent = evb_complex->parent_id;

    Vij *= 2.0 * evb_complex->Cs[istate] * evb_complex->Cs[parent[istate]];
}
