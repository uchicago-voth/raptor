/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight and Gard Nelson
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "EVB_complex.h"
#include "EVB_type.h"
#include "EVB_engine.h"
#include "EVB_rep_expon.h"

#include "universe.h"
#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "update.h"
#include "error.h"

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_Rep_Expon::EVB_Rep_Expon(LAMMPS *lmp, EVB_Engine *engine) : EVB_Repulsive(lmp,engine)
{
  int n = atom->ntypes;
  
  setflag_type = new int [n+1];

  memory->create(setflag_pair, n+1, n+1, "EVB_rep:setflag_pair");
  memory->create(cutsq, n+1, n+1, "EVB_rep:cutsq");
  memory->create(_a, n+1, n+1, "EVB_rep:_a");
  memory->create(_b, n+1, n+1, "EVB_rep:_b");
  memory->create(_r0, n+1, n+1, "EVB_rep:_r0");

  for(int i=1; i<=n; i++) {
    setflag_type[i] = 0;
    for(int j=1; j<=n; j++) {
      setflag_pair[i][j] = 0;
      cutsq[i][j] = 0.0;
      _a[i][j] = 0.0;
      _b[i][j] = 0.0;
      _r0[i][j] = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

EVB_Rep_Expon::~EVB_Rep_Expon()
{
  delete [] setflag_type;

  memory->destroy(setflag_pair);
  memory->destroy(cutsq);
  memory->destroy(_a);
  memory->destroy(_b);
  memory->destroy(_r0);
}

/* ----------------------------------------------------------------------*/

int EVB_Rep_Expon::data_rep(char *buf, int *offset, int start, int end)
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

  int itype, jtype;
  double a_coef, b_coef, r_coef, cut;

  Vii_const = atof(buf+offset[t++]);
  num_pairs = atoi(buf+offset[t++]);

  if(universe->me == 0) {
    fprintf(fp,"   VJJ(R) = Vii_const + \\sum a * e^(-b * (R - r0))\n");
    fprintf(fp,"\n   Vii_const= %f\n\n   Number of unique pairs of particles: num_pairs= %i\n",Vii_const,num_pairs);
  }

  for (int i=0; i<num_pairs; i++) {
    itype  = atoi(buf+offset[t++]);
    jtype  = atoi(buf+offset[t++]);
    a_coef = atof(buf+offset[t++]);
    b_coef = atof(buf+offset[t++]);
    r_coef = atof(buf+offset[t++]);
    cut    = atof(buf+offset[t++]);
    
    if(universe->me == 0) {
      fprintf(fp,"\n   Pair: %i\n",i);
      fprintf(fp,"   +++++++++++++++++++++++++++++++\n");
      fprintf(fp,"      Atom Types: %i + %i w/ cutoff= %f.\n",itype,jtype,cut);
      fprintf(fp,"      a= %f\n      b= %f\n      r0= %f\n",a_coef,b_coef,r_coef);
    }

    setflag_type[itype] = 1;
    setflag_pair[itype][jtype] = 1;
    cutsq[itype][jtype] = cut * cut;
    _a[itype][jtype] = a_coef;
    _b[itype][jtype] = b_coef;
    _r0[itype][jtype] = r_coef;

    setflag_type[jtype] = 1;
    setflag_pair[jtype][itype] = 1;
    cutsq[jtype][itype] = cutsq[itype][jtype];
    _a[jtype][itype] = a_coef;
    _b[jtype][itype] = b_coef;
    _r0[jtype][itype] = r_coef;
  }
  
  return t;
}

void EVB_Rep_Expon::compute(int vflag)
{
  double *v = virial;                     // virial
  memset(v,0,sizeof(double)*6);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nall = atom->nlocal + atom->nghost;

  int i, ii, itype, imol, j, jtype, jmol;
  double xtmp, ytmp, ztmp, rsq, r2inv, r;
  double delx, dely, delz, aexp, fpair, dfx, dfy, dfz;
  
  int * molecule = atom->molecule;
  int ** map = evb_engine->molecule_map;
  int inum = map[center_mol_id][0];

  energy = Vii_const;

  // loop over atoms in target molecule
  for(ii=1; ii<=inum; ii++) {
    i = map[center_mol_id][ii];
    imol = molecule[i];

    itype = type[i];
    if(!setflag_type[itype]) continue; // No pairs defined for this type

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop over all atoms on processor
    for(j=0; j<nall; j++) {
      jmol = molecule[j];
      if(i == j || imol == jmol || atom->tag[i] == atom->tag[j] || j != atom->map(atom->tag[j])) continue;

      jtype = type[j];
      if(!setflag_pair[itype][jtype]) continue; // Pair not defined

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];

      domain->minimum_image(delx,dely,delz);
      rsq = delx*delx + dely*dely + delz*delz;

      if(rsq < cutsq[itype][jtype]) {
	r2inv = 1.0 / rsq;
	r = sqrt(rsq);
	aexp = _a[itype][jtype] * exp(-_b[itype][jtype] * (r - _r0[itype][jtype]));
	fpair = _b[itype][jtype] * r * aexp * r2inv;

	energy += aexp;
	dfx = delx * fpair;
	dfy = dely * fpair;
	dfz = delz * fpair;

	f[i][0] += dfx;
	f[i][1] += dfy;
	f[i][2] += dfz;
	
	f[j][0] -= dfx;
	f[j][1] -= dfy;
	f[j][2] -= dfz;

	if(vflag) {
	  v[0] += dfx * delx;
	  v[1] += dfy * dely;
	  v[2] += dfz * delz;
	  v[3] += dfx * dely;
	  v[4] += dfx * delz;
	  v[5] += dfy * delz;
	}

      } // if(rsq<cutsq)

    } // for(jj<jnum)

  } // for(ii<inum)

}

/* ----------------------------------------------------------------------*/

void EVB_Rep_Expon::sci_compute(int vflag)
{
  int * cplx_atom = evb_engine->complex_atom; 
  int istate = evb_complex->current_status;
  double cs2 = evb_complex->Cs2[istate];

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nall = atom->nlocal+atom->nghost;

  int i, ii, itype, j, jtype;
  double xtmp, ytmp, ztmp, rsq, r2inv, r;
  double delx, dely, delz, aexp, fpair, dfx, dfy, dfz;

  int ** map = evb_engine->molecule_map;
  int inum = map[center_mol_id][0];

  // loop over atoms in target molecule
  for(ii=1; ii<=inum; ii++) {
    i = map[center_mol_id][ii];

    itype = type[i];
    if(!setflag_type[itype]) continue; // No pairs defined for this type
    
    int cplx_id = cplx_atom[i];
    
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    // loop over all atoms on processor
    for(j=0; j<nall; j++) {
      if(i != j && cplx_atom[j] != cplx_id && j == atom->map(atom->tag[j])) {

	jtype = type[j];
	if(!setflag_pair[itype][jtype]) continue; // Pair not defined
	
	delx = xtmp - x[j][0];
	dely = ytmp - x[j][1];
	delz = ztmp - x[j][2];
	
	domain->minimum_image(delx,dely,delz);
	rsq = delx*delx + dely*dely + delz*delz;
	
	if(rsq < cutsq[itype][jtype]) {
	  r2inv = 1.0 / rsq;
	  r = sqrt(rsq);
	  aexp = _a[itype][jtype] * exp(-_b[itype][jtype] * (r - _r0[itype][jtype]));
	  fpair = _b[itype][jtype] * r * aexp * r2inv;

	  dfx = delx * fpair;
	  dfy = dely * fpair;
	  dfz = delz * fpair;
	  
	  f[i][0] += dfx;
	  f[i][1] += dfy;
	  f[i][2] += dfz;

	} // if(rsq<cutsq)
    
      } // if(inter-complex pair)
      
    } // for(jj<jnum)
      
  } // for(ii<inum)

}

int EVB_Rep_Expon::checkout(int* _index)
{

  int n = atom->ntypes;
  int index_max = 20;
  int count = 0;

  _index[count++] = -2 * num_pairs; // EVB_Checkout::write2txt will write _index[j] to checkpoint file.
  
  // Collect defined pairs
  for(int i=1; i<=n; i++) {
    for(int j=i; j<=n; j++) {
      if(setflag_pair[i][j]) {
	_index[count++] = i;
	_index[count++] = j;
      }
    }
  }

  if(count>index_max) error->all(FLERR,"Error: EVB_Rep_Expon::checkout  count>20.\n");

  for(int i=count; i<index_max; i++) _index[i] = -1;
  return index_max;
}
