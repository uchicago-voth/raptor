/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "EVB_complex.h"
#include "EVB_type.h"
#include "EVB_engine.h"
#include "EVB_rep_vii.h"

#include "universe.h"
#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "error.h"

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_Rep_Vii::EVB_Rep_Vii(LAMMPS *lmp, EVB_Engine *engine) : EVB_Repulsive(lmp,engine)
{
  
}

/* ---------------------------------------------------------------------- */

EVB_Rep_Vii::~EVB_Rep_Vii()
{

}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------*/

int EVB_Rep_Vii::data_rep(char *buf,int *offset, int start, int end)
{
  int t=start;
  
  FILE * fp = evb_engine->fp_cfg_out;

  etp_center = evb_type->get_type(buf+offset[t++]);
  if(etp_center==-1) {
    char errline[255];
    sprintf(errline,"[EVB] Undefined molecule_type [%s].", buf+offset[t-1]);
    error->all(FLERR,errline);
  }
  
  if(universe->me == 0) {
    fprintf(fp,"   This interaction computed for all states with molecule present: etp_center= %s.\n",
	    evb_engine->evb_type->name[etp_center-1]);
  }

  vii = atof(buf+offset[t++]);

  if(universe->me == 0) fprintf(fp,"   VJJ = Vii_const\n      Vii_const= %f\n",vii);

  return t;
}

/* ----------------------------------------------------------------------*/

void EVB_Rep_Vii::compute(int vflag)
{
  energy = vii;
}

/* ----------------------------------------------------------------------*/

int EVB_Rep_Vii::checkout(int* _index)
{
  int index_max = 10;

  for(int i=0; i<index_max; i++) _index[i] = -1;
  
  int* cplx_atom = evb_engine->complex_atom; 
  int **map = evb_engine->molecule_map;

  int count = 0;
  _index[count++] = 1; // EVB_Checkout::write2txt will write map[_index[j]] to checkpoint file.
  
  if(strcmp(evb_type->name[etp_center-1],"GLU-P")==0) {
    _index[count++] = map[center_mol_id][6];
    _index[count++] = map[center_mol_id][7];
  }
  
  return index_max;
}
