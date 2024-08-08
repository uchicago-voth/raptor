/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "EVB_complex.h"
#include "EVB_type.h"
#include "EVB_engine.h"
#include "EVB_rep_hydroxide.h"

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

EVB_Rep_Hydroxide::EVB_Rep_Hydroxide(LAMMPS *lmp, EVB_Engine *engine) : EVB_Repulsive(lmp,engine)
{

}

/* ---------------------------------------------------------------------- */

EVB_Rep_Hydroxide::~EVB_Rep_Hydroxide()
{

}

/* ----------------------------------------------------------------------*/

int EVB_Rep_Hydroxide::data_rep(char *buf, int *offset, int start, int end)
{
  int t=start;
  
  FILE * fp = evb_engine->fp_cfg_out;

  etp_center = evb_type->get_type(buf+offset[t++]);
  if(etp_center==-1) {
    char errline[255];
    sprintf(errline,"[EVB] Undefined molecule_type [%s].", buf+offset[t-1]);
    error->all(FLERR,errline);
  }
  
  if(universe->me == 0) fprintf(fp,"   This interaction computed every state with this molecule present: etp_center= %s.\n",
				evb_engine->evb_type->name[etp_center-1]);

  atp_OW = atoi(buf+offset[t++]);
  B = atof(buf+offset[t++]);
  b1 = atof(buf+offset[t++]);
  b2 = atof(buf+offset[t++]);
  d_OO = atof(buf+offset[t++]);
  C = atof(buf+offset[t++]);
  c1 = atof(buf+offset[t++]);
  d_OH = atof(buf+offset[t++]);
  
  if(universe->me == 0) {
    fprintf(fp,"   Target atom type of hydroxide interaction: %i.\n\n",atp_OW);
    fprintf(fp,"   VJJ = VOO(ROO,q) + VHO(RHO)\n\n");
    fprintf(fp,"   VOO(R,q) = B * e^(-b1 * (R - d_OO)) * \\sum e^(-b2 * q^2)\n");
    fprintf(fp,"      B= %f\n      b1= %f\n      d_OO= %f\n      b2= %f\n",B,b1,d_OO,b2);
    fprintf(fp,"\n   VHO(R) = C * e^(-c * (R - dOH))\n");
    fprintf(fp,"      C= %f\n      c= %f\n      dOH= %f\n",C,c1,d_OH);
  }

  cutoff_OO[0] = atof(buf+offset[t++]);
  cutoff_OO[1] = atof(buf+offset[t++]);
  cutoff_HO[0] = atof(buf+offset[t++]);
  cutoff_HO[1] = atof(buf+offset[t++]);
  
  if(universe->me == 0) {
    fprintf(fp,"\n   Parameters for OO switching function: rs= %f  rc= %f.\n",cutoff_OO[0],cutoff_OO[1]);
    fprintf(fp,"   Parameters for HO switching function: rs= %f  rc= %f.\n",cutoff_HO[0],cutoff_HO[1]);
  }

  oo_cutoff_1 = pow(cutoff_OO[1]-cutoff_OO[0],-3);
  ho_cutoff_1 = pow(cutoff_HO[1]-cutoff_HO[0],-3);
  
  if(fabs(b2)<1e-6) bEVB3 = 0; else bEVB3 = 1;
  
  return t;
}

/* ----------------------------------------------------------------------*/
/*   repulsive term for diagonal state, see JPCB 112(2008)467, Eq. 7-9   */
/*     note: the definition of q_HjOk is described in JPCB 112(2008)7146 */
/* ----------------------------------------------------------------------*/

void EVB_Rep_Hydroxide::compute(int vflag)
{  
  double *v = virial;                     // virial
  memset(v,0,sizeof(double)*6);

  energy = e_oo = e_ho = 0;
  int **map = evb_engine->molecule_map;
  
  int atom_o = map[center_mol_id][1];
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int *molecule = atom->molecule;
  int nall = atom->nlocal+atom->nghost;
  int atp_OH = type[atom_o];

  for (int i = 0; i < nall; i++) 
  {

    if (type[i] == atp_OW || type[i]==atp_OH)
    {
      if( i == atom_o || atom->tag[i] == atom->tag[atom_o] || i != atom->map(atom->tag[i]) ) continue;

      int oh = atom_o, ow = i;
      double dxook,dyook,dzook,dxhok[3],dyhok[3],dzhok[3],ene;
      double dohhx[3],dohhy[3],dohhz[3];
      double dowhx[3],dowhy[3],dowhz[3];
      double r_oo, r_ho, tt;
      double r_ho2[3];
      double exp1,exp2[3],exp2_sum;
      double fo[3],fh[3],fok[3],fhj[3][3];
		
      // calculate distance between r_OH and r_OW

      dxook = x[oh][0]-x[ow][0];
      dyook = x[oh][1]-x[ow][1];
      dzook = x[oh][2]-x[ow][2];

      domain->minimum_image(dxook,dyook,dzook);
      r_oo = sqrt(dxook*dxook+dyook*dyook+dzook*dzook);

      if (r_oo < cutoff_OO[1]) 
	{	
	  exp1 = exp(-b1*(r_oo-d_OO));
	  
	  if(bEVB3)
	    {
	      // {Hydroxide Oxygen}  --Hyd. Bond--  {Water Hydrogen} --Cov. Bond-- {Water Oxygen}
	      exp2_sum = 0.0;
	      int w = molecule[ow];
	      int nA = map[w][0];
	      if( nA > 3) error->all(FLERR,"EVB_Rep_Hydroxide::compute()  water molecule??");
	      for (int k = 0; k < nA; k++) {
		int hw = map[w][k+1]; // Index of water hydrogen?
		if(type[hw] != atp_OW) {
		  
		  dohhx[k] = x[oh][0] - x[hw][0];
		  dohhy[k] = x[oh][1] - x[hw][1];
		  dohhz[k] = x[oh][2] - x[hw][2];
		  domain->minimum_image(dohhx[k],dohhy[k],dohhz[k]);
		  
		  dowhx[k] = x[ow][0] - x[hw][0];
		  dowhy[k] = x[ow][1] - x[hw][1];
		  dowhz[k] = x[ow][2] - x[hw][2];
		  domain->minimum_image(dowhx[k],dowhy[k],dowhz[k]);
		  
		  dxhok[k] = (dohhx[k] + dowhx[k]) / 2.0;
		  dyhok[k] = (dohhy[k] + dowhy[k]) / 2.0;
		  dzhok[k] = (dohhz[k] + dowhz[k]) / 2.0;
		  domain->minimum_image(dxhok[k],dyhok[k],dzhok[k]);
		  
		  r_ho2[k] = dxhok[k]*dxhok[k]+dyhok[k]*dyhok[k]+dzhok[k]*dzhok[k];
		  exp2[k] = exp(-b2 * r_ho2[k]);
		  exp2_sum += exp2[k];
		}
	      }
	      
	      ene = B * exp1 * exp2_sum;
	    }
	  else { ene = B * exp1; }
	  
	  // energy by V_OOk_rep, Eq. 7 in JPCB 112(2008)467
	  
	  if (r_oo < cutoff_OO[0])  
	    e_oo += ene;
	  else  {
	    sw =  switching(oo_cutoff_1, cutoff_OO[0], cutoff_OO[1], r_oo);
	    e_oo += ene * sw;
	  }
	  
	  // force by r_oo, first term of Eq. 7 in JPCB 112(2008)467
	  
	  tt = b1 * ene / r_oo;
	  if (r_oo >= cutoff_OO[0]) 
	    tt = tt*sw + ene * dswitching(oo_cutoff_1, cutoff_OO[0], cutoff_OO[1], r_oo) / r_oo;
	  dfx = tt * dxook;
	  f[oh][0] += dfx;
	  f[ow][0] -=dfx;
	  dfy = tt * dyook;
	  f[oh][1] += dfy;
	  f[ow][1] -=dfy;
	  dfz = tt * dzook;
	  f[oh][2] += dfz;
	  f[ow][2] -=dfz;
	  
	  // virial by r_oo, first term of Eq. 7 in JPCB 112(2008)467
	  
	  v[0] += dfx * dxook;
	  v[1] += dfy * dyook;
	  v[2] += dfz * dzook;
	  v[3] += dfx * dyook;
	  v[4] += dfx * dzook;
	  v[5] += dfy * dzook;
	  
	  
	  int w = molecule[i];
	  int nA = map[w][0];
	  if (bEVB3) for (int k = 0; k < nA; k++) {
	      int hw = map[w][k+1];
	      if(type[hw] != atp_OW) {
		
		tt = b2 *  B * exp1 * exp2[k];
		if (r_oo >= cutoff_OO[0]) tt *= sw;
		
		// force by R_HjOk, second term of Eq. 7 in JPCB 112(2008)467
		
		dfx = tt * dxhok[k];
		f[oh][0] += dfx;
		f[ow][0] += dfx;
		f[hw][0] -= dfx * 2.0;
		dfy = tt * dyhok[k];
		f[oh][1] += dfy;
		f[ow][1] += dfy;
		f[hw][1] -= dfy * 2.0;
		dfz = tt * dzhok[k];
		f[oh][2] += dfz;
		f[ow][2] += dfz;
		f[hw][2] -= dfz * 2.0;
		
		// virial by R_HjOk, second term of Eq. 7 in JPCB 112(2008)467
		
		v[0] += dfx * (dohhx[k] + dowhx[k]);
		v[1] += dfy * (dohhy[k] + dowhy[k]);
		v[2] += dfz * (dohhz[k] + dowhz[k]);
		v[3] += dfx * (dohhy[k] + dowhy[k]);
		v[4] += dfx * (dohhz[k] + dowhz[k]);
		v[5] += dfy * (dohhz[k] + dowhz[k]);
	      }
	    }
	}
      
      // energy, force, and virial by V_HOk_rep, Eq. 8 in JPCB 112(2008)467

      // {Hydroxide Oxygen}  --Hyd. Bond--  {Water Hydrogen}
      int k = 0;
      int w = molecule[ow];
      int nA = map[w][0];
      if( nA > 3) error->all(FLERR,"EVB_Rep_Hydroxide::compute()  water molecule??");
      for (int k = 0; k < nA; k++) {
	int hw = map[w][k+1]; // Index of water hydrogen
	if(type[hw] != atp_OW) {
	  
	  dowhx[k] = x[hw][0] - x[oh][0];
	  dowhy[k] = x[hw][1] - x[oh][1];
	  dowhz[k] = x[hw][2] - x[oh][2];

	  domain->minimum_image(dowhx[k],dowhy[k],dowhz[k]);
	  r_ho = sqrt(dowhx[k]*dowhx[k] + dowhy[k]*dowhy[k] + dowhz[k]*dowhz[k]);

	  if (r_ho < cutoff_HO[1]) {
	    ene = C * exp(-c1*(r_ho-d_OH));
	    if (r_ho < cutoff_HO[0])
	      e_ho += ene;
	    else {
	      sw =  switching(ho_cutoff_1, cutoff_HO[0], cutoff_HO[1], r_ho);
	      e_ho += ene * sw;
	    }
	
	    tt = c1 * ene / r_ho;
	    if (r_ho >= cutoff_HO[0])
	      tt = tt * sw + ene * dswitching(ho_cutoff_1, cutoff_HO[0], cutoff_HO[1], r_ho) / r_ho;
	
	    dfx = tt * dowhx[k];
	    f[hw][0] += dfx;
	    f[oh][0] -= dfx;
	    dfy = tt * dowhy[k];
	    f[hw][1] += dfy;
	    f[oh][1] -= dfy;
	    dfz = tt * dowhz[k];
	    f[hw][2] += dfz;
	    f[oh][2] -= dfz;
	    
	    v[0] += dfx * dowhx[k];
	    v[1] += dfy * dowhy[k];
	    v[2] += dfz * dowhz[k];
	    v[3] += dfx * dowhy[k];
	    v[4] += dfx * dowhz[k];
	    v[5] += dfy * dowhz[k];
	  }
	}
      }

    } // if right atom type
  } // Loop over all atoms
  
  energy = e_oo + e_ho;
}

/* ----------------------------------------------------------------------*/

void EVB_Rep_Hydroxide::scan_potential_surface()
{
  int **map = evb_engine->molecule_map;  
  int atom_o = map[center_mol_id][1]; 
  int *type = atom->type;
  double **x = atom->x;
  double **f = atom->f;
  
  fprintf(screen,"******************************************************\n");
  fprintf(screen,"****** Scan Potential Surface of Repulsive Term ******\n");
  fprintf(screen,"******************************************************\n");
  
  FILE *output1 = fopen("repul_pes.xvg","w");
  FILE *output2 = fopen("repul_f.xvg","w");
  FILE *output3 = fopen("repul_e.xvg","w");
  
  double start    = 1.5;
  double interval = 0.00001;
  int nsample  = 200000;
  
  double* r = new double[nsample+2];
  double* fr= new double[nsample+2];
  double* e = new double[nsample+2];
  double* e1= new double[nsample+2];
  double* e2= new double[nsample+2];
  double* de= new double[nsample+2];
  double* fi[3];
  fi[0] = new double[nsample+2];
  fi[1] = new double[nsample+2];
  fi[2] = new double[nsample+2];
  
  for(int i=0; i<nsample+2; i++) r[i] = start+interval*(i-1);
  
  int target = 0;
  for(int i=0; i<atom->nlocal+atom->nlocal; i++)
    if(type[i]==atp_OW) 
	{
	  target = i;
	  break;
	}
  
  double d[3];
  for(int i=0; i<3; i++)  d[i] = x[target][i]-x[atom_o][i];
  double dr = sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
  double c[3];
  for(int i=0; i<3; i++)  c[i] = d[i]/dr;
  
  for(int i=0; i<nsample+2; i++)
  {
    for(int j=0; j<3; j++) f[target][j]=0.0;
	for(int j=0; j<3; j++) x[target][j]=x[atom_o][j]+c[j]*r[i];
	
	compute(false);
	
	e[i]=energy; e1[i]=e_oo; e2[i]=e_ho;
	for(int j=0; j<3; j++) fi[j][i]=f[target][j];
	fr[i] = sqrt(fi[0][i]*fi[0][i]+fi[1][i]*fi[1][i]+fi[2][i]*fi[2][i]);
  }
  
  for(int i=1; i<=nsample; i++)
  {
    de[i] = (e[i-1]-e[i+1])/2/interval;
    if(i%100==0) fprintf(screen,"r=%-12lf   analytic=%-12lf   numeric=%-12lf   error=%-12lf\n",r[i],fr[i],de[i],fr[i]-de[i]);
  }
  
  for(int i=0; i<nsample+2; i++) e[i]-=e[nsample+1];
  
  for(int i=1; i<=nsample; i++)
    fprintf(output1,"%lf %lf %lf %lf\n",r[i],e[i],de[i],fr[i]);
	
  for(int i=1; i<=nsample; i++)
    fprintf(output2, "%lf %lf %lf %lf\n", r[i],fi[0][i],fi[1][i],fi[2][i]);
  
  for(int i=1; i<=nsample; i++)
    fprintf(output3, "%lf %lf %lf %lf\n", r[i],e1[i],e2[i],e[i]);
	
  fclose(output1);
  fclose(output2);
  fclose(output3);
  
  exit(0);
}

/* ----------------------------------------------------------------------*/
/* switching function, JPCB 112(2008)467, Eq. 9                          */
/* ----------------------------------------------------------------------*/

double EVB_Rep_Hydroxide::switching(double denominator, double rs, double rc, double rr)
{
  return 1.0-(rr-rs)*(rr-rs)*(3.0*rc-rs-2.0*rr)*denominator;
}

/* ----------------------------------------------------------------------*/
/* derivative of switching function, JPCB 112(2008)467, Eq. 9                          */
/* ----------------------------------------------------------------------*/

double EVB_Rep_Hydroxide::dswitching(double denominator, double rs, double rc, double rr)
{
  return 6.0*(rr-rs)*(rc-rr)*denominator;
}


/*** SCI ***/


void EVB_Rep_Hydroxide::sci_compute(int vflag)
{  
  int* cplx_atom = evb_engine->complex_atom;
  int istate = evb_complex->current_status;
  double cs2 = evb_complex->Cs2[istate];
  int **map = evb_engine->molecule_map;
  
  int atom_o = map[center_mol_id][1];
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int *molecule = atom->molecule;
  int nall = atom->nlocal+atom->nghost;

  int cplx_id = cplx_atom[atom_o];
  int atp_OH = type[atom_o];

  error->one(FLERR,"Check to make sure EVB_Rep_Hydroxide::sci_compute() is correct.");

  for (int i = 0; i < nall; i++) 
  {
    if ( (type[i] == atp_OW || type[i]==atp_OH ) && cplx_atom[i]!=cplx_id && i==atom->map(atom->tag[i]) )
    {
      int oh = atom_o, ow = i;
      double dxook,dyook,dzook,dxhok[3],dyhok[3],dzhok[3],ene;
      double dohhx[3],dohhy[3],dohhz[3];
      double dowhx[3],dowhy[3],dowhz[3];
      double r_oo, r_ho, tt;
      double r_ho2[3];
      double exp1,exp2[3],exp2_sum;
      double fo[3],fh[3],fok[3],fhj[3][3];
		
      // calculate distance between r_OH and r_OW

      dxook = x[oh][0]-x[ow][0];
      dyook = x[oh][1]-x[ow][1];
      dzook = x[oh][2]-x[ow][2];

      domain->minimum_image(dxook,dyook,dzook);
      r_oo = sqrt(dxook*dxook+dyook*dyook+dzook*dzook);

      if (r_oo < cutoff_OO[1]) 
	{	
	  exp1 = exp(-b1*(r_oo-d_OO));
	  
	  if(bEVB3)
	    {
	      // {Hydroxide Oxygen}  --Hyd. Bond--  {Water Hydrogen} --Cov. Bond-- {Water Oxygen}
	      exp2_sum = 0.0;
	      int w = molecule[i];
	      int nA = map[w][0];
	      if( nA > 3) error->all(FLERR,"EVB_Rep_Hydroxide::compute()  water molecule??");
	      for (int k = 0; k < nA; k++) {
		int hw = map[w][k+1]; // Index of water hydrogen?
		if(type[hw] != atp_OW) {
		  
		  dohhx[k] = x[oh][0] - x[hw][0];
		  dohhy[k] = x[oh][1] - x[hw][1];
		  dohhz[k] = x[oh][2] - x[hw][2];
		  domain->minimum_image(dohhx[k],dohhy[k],dohhz[k]);
		  
		  dowhx[k] = x[ow][0] - x[hw][0];
		  dowhy[k] = x[ow][1] - x[hw][1];
		  dowhz[k] = x[ow][2] - x[hw][2];
		  domain->minimum_image(dowhx[k],dowhy[k],dowhz[k]);
		  
		  dxhok[k] = (dohhx[k] + dowhx[k]) / 2.0;
		  dyhok[k] = (dohhy[k] + dowhy[k]) / 2.0;
		  dzhok[k] = (dohhz[k] + dowhz[k]) / 2.0;
		  domain->minimum_image(dxhok[k],dyhok[k],dzhok[k]);
		  
		  r_ho2[k] = dxhok[k]*dxhok[k]+dyhok[k]*dyhok[k]+dzhok[k]*dzhok[k];
		  exp2[k] = exp(-b2 * r_ho2[k]);
		  exp2_sum += exp2[k];
		}
	      }
	      
	      ene = B * exp1 * exp2_sum;
	    }
	  else { ene = B * exp1; }
	  
		
	  // force by r_oo, first term of Eq. 7 in JPCB 112(2008)467
	  
	  if (r_oo > cutoff_OO[0])  
	    sw =  switching(oo_cutoff_1, cutoff_OO[0], cutoff_OO[1], r_oo);
	  
	  tt = b1 * ene / r_oo;
	  if (r_oo >= cutoff_OO[0]) 
	    tt = tt*sw + ene * dswitching(oo_cutoff_1, cutoff_OO[0], cutoff_OO[1], r_oo) / r_oo;
	  
	  tt *= cs2;  
	  
	  dfx = tt * dxook;
	  f[ow][0] -= dfx;
	  dfy = tt * dyook;
	  f[ow][1] -= dfy;
	  dfz = tt * dzook;
	  f[ow][2] -= dfz;
	  
	  int w = molecule[i];
	  int nA = map[w][0];
	  if (bEVB3) for (int k = 0; k < nA; k++) {
	      int hw = map[w][k+1];
	      if(type[hw] != atp_OW) {
		
		tt = b2 *  B * exp1 * exp2[k];
		if (r_oo >= cutoff_OO[0]) tt *= sw;
		
		// force by R_HjOk, second term of Eq. 7 in JPCB 112(2008)467
		
		tt *= cs2;
		
		dfx = tt * dxhok[k];
		f[ow][0] += dfx;
		dfy = tt * dyhok[k];
		f[ow][1] += dfy;
		dfz = tt * dzhok[k];
		f[ow][2] += dfz;
	      }
	    }
	}
      
      // energy, force, and virial by V_HOk_rep, Eq. 8 in JPCB 112(2008)467

      // {Hydroxide Oxygen}  --Hyd. Bond--  {Water Hydrogen}
      int k = 0;
      int w = molecule[ow];
      int nA = map[w][0];
      if( nA > 3) error->all(FLERR,"EVB_Rep_Hydroxide::compute()  water molecule??");
      for (int k = 0; k < nA; k++) {
	int hw = map[w][k+1]; // Index of water hydrogen
	if(type[hw] != atp_OW) {
	  
	  dowhx[k] = x[hw][0] - x[oh][0];
	  dowhy[k] = x[hw][1] - x[oh][1];
	  dowhz[k] = x[hw][2] - x[oh][2];

	  domain->minimum_image(dowhx[k],dowhy[k],dowhz[k]);
	  r_ho = sqrt(dowhx[k]*dowhx[k] + dowhy[k]*dowhy[k] + dowhz[k]*dowhz[k]);

	  if (r_ho < cutoff_HO[1]) {
	    ene = C * exp(-c1*(r_ho-d_OH));
	    if (r_ho > cutoff_HO[0]) sw =  switching(ho_cutoff_1, cutoff_HO[0], cutoff_HO[1], r_ho);
	
	    tt = c1 * ene / r_ho;
	    if (r_ho >= cutoff_HO[0])
	      tt = tt * sw + ene * dswitching(ho_cutoff_1, cutoff_HO[0], cutoff_HO[1], r_ho) / r_ho;

	    tt *= cs2;
	
	    dfx = tt * dowhx[k];
	    f[hw][0] += dfx;
	    dfy = tt * dowhy[k];
	    f[hw][1] += dfy;
	    dfz = tt * dowhz[k];
	    f[hw][2] += dfz;
	    
	  }
	}
      }

    }
  }
}

int EVB_Rep_Hydroxide::checkout(int* _index)
{
  int index_max = 20;
  int **map = evb_engine->molecule_map;
  
  int atom_o = map[center_mol_id][1];
  
  double **x = atom->x;
  int *type = atom->type;
  int *molecule = atom->molecule;
  int nall = atom->nlocal+atom->nghost;
  
  int atp_OH = type[atom_o];
  
  int count = 0;
  _index[count++] = 1; // EVB_Checkout::write2txt will write map[_index[j]] to checkpoint file.
  _index[count++] = atom_o;
  
  for(int i=0; i<nall; i++) {
    if (type[i] == atp_OW || type[i]==atp_OH) {
      if(i==atom_o || atom->tag[i]==atom->tag[atom_o] || i!=atom->map(atom->tag[i])) continue;
      int oh = atom_o, ow = i;
      double dxook,dyook,dzook;
      double dowhx,dowhy,dowhz;
      double r_oo, r_ho;
      
      // calculate distance between r_OH and r_OW
      
      dxook = x[oh][0]-x[ow][0];
      dyook = x[oh][1]-x[ow][1];
      dzook = x[oh][2]-x[ow][2];
      domain->minimum_image(dxook,dyook,dzook);
      r_oo = sqrt(dxook*dxook+dyook*dyook+dzook*dzook);
      
      int w = molecule[ow];
      int nA = map[w][0];
      int test = 0;
      if(nA>3) error->one(FLERR,"EVB_Rep_Hydroxide::checkout()  water molecule??");
      if (r_oo < cutoff_OO[1]) test = 1;
      else { // If need be, calculate distance between each r_HH and r_OW
	for(int k=0; k<nA; k++) {
	  int hw = map[w][k+1];
	  if(type[hw] != atp_OW) {
	    dowhx = x[oh][0] - x[hw][0]; 
	    dowhy = x[oh][1] - x[hw][1];
	    dowhz = x[oh][2] - x[hw][2];
	    domain->minimum_image(dowhx,dowhy,dowhz);
	    r_ho = sqrt(dowhx*dowhx + dowhy*dowhy + dowhz*dowhz);
	    if(r_ho < cutoff_HO[1]) test = 1;
	  }
	}
      }

      if(test) {
	_index[count++] = ow;
	for(int k=0; k<nA; k++) {
	  int hw = map[w][k+1];  // Index of water hydrogen?
	  if(type[hw] != atp_OW) _index[count++] = hw;
	}
      }

    } // if type
  } // Loop over atoms
  
  if(count>index_max) fprintf(stdout,"Warning: EVB_rep_hydroxide::checkout  count>index_max.\n");
  
  for(int i=count; i<index_max; i++) _index[i] = -1;
  return index_max;
}
