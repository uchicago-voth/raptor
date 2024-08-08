/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight and Gard Nelson
     based on EVB_rep_hydronium written by Yuxing Peng, Tianying Yan
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "EVB_complex.h"
#include "EVB_type.h"
#include "EVB_engine.h"
#include "EVB_rep_hydronium_expon.h"

#include "universe.h"
#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "update.h"
#include "error.h"

#include "comm.h"

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_Rep_Hydronium_Expon::EVB_Rep_Hydronium_Expon(LAMMPS *lmp, EVB_Engine *engine) : EVB_Repulsive(lmp,engine)
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

EVB_Rep_Hydronium_Expon::~EVB_Rep_Hydronium_Expon()
{
  delete [] setflag_type;

  memory->destroy(setflag_pair);
  memory->destroy(cutsq);
  memory->destroy(_a);
  memory->destroy(_b);
  memory->destroy(_r0);
}

/* ----------------------------------------------------------------------*/

int EVB_Rep_Hydronium_Expon::data_rep(char *buf, int *offset, int start, int end)
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

  // Parameters for original MS-EVB3 definition

  atp_OW = atoi(buf+offset[t++]);
  B = atof(buf+offset[t++]);
  b1 = atof(buf+offset[t++]);
  b2 = atof(buf+offset[t++]);
  d_OO = atof(buf+offset[t++]);
  C = atof(buf+offset[t++]);
  c1 = atof(buf+offset[t++]);
  d_OH = atof(buf+offset[t++]);
  
  if(universe->me == 0) {
    fprintf(fp,"   Target atom type of hydronium interaction: %i.\n\n",atp_OW);
    fprintf(fp,"   VJJ = VOO(ROO,q) + VHO(RHO) + V_expon\n\n");
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

  Bb1 = B*b1;
  Bb2 = B*b2;
  Cc1 = C*c1;
  
  oo_cutoff_1 = pow(cutoff_OO[1]-cutoff_OO[0],-3);
  oo_cutoff_2 = 3*cutoff_OO[1]-cutoff_OO[0];
  oo_cutoff_3 = cutoff_OO[0]+cutoff_OO[1];
  oo_cutoff_4 = cutoff_OO[0]*cutoff_OO[1];
  
  ho_cutoff_1 = pow(cutoff_HO[1]-cutoff_HO[0],-3);
  ho_cutoff_2 = 3*cutoff_HO[1]-cutoff_HO[0];
  ho_cutoff_3 = cutoff_HO[0]+cutoff_HO[1];
  ho_cutoff_4 = cutoff_HO[0]*cutoff_HO[1];
  
  if(fabs(b2)<1e-6) bEVB3 = 0; else bEVB3 = 1;

  // Parameters for supplemental pairwise interactions
  
  int itype, jtype;
  double a_coef, b_coef, r_coef, cut;

  Vii_const = atof(buf+offset[t++]);
  num_pairs = atoi(buf+offset[t++]);

  if(universe->me == 0) {
    fprintf(fp,"\n   V_expon = Vii_const + \\sum a * e^(-b * (R - r0))\n");
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
    _a[itype][jtype] = a_coef;
    _b[itype][jtype] = b_coef;
    _r0[itype][jtype] = r_coef;
  }
  
  return t;
}

/* ----------------------------------------------------------------------*/
/*   repulsive term for diagonal state, see JPCB 112(2008)467, Eq. 7-9   */
/*     note: the definition of q_HjOk is described in JPCB 112(2008)7146 */
/* ----------------------------------------------------------------------*/

void EVB_Rep_Hydronium_Expon::compute(int vflag)
{
  double *v = virial;                     // virial
  memset(v,0,sizeof(double)*6);

  energy = e_oo = e_ho = 0.0; 
  int **map = evb_engine->molecule_map;

  int atom_o = map[center_mol_id][1];  
  int atom_h[3];  
  atom_h[0] = map[center_mol_id][2];
  atom_h[1] = map[center_mol_id][3];
  atom_h[2] = map[center_mol_id][4];

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int *molecule = atom->molecule;
  int nall = atom->nlocal+atom->nghost;
  int atp_OH = type[atom_o];

  // Original MS-EVB3 definition

  for(int i=0; i<nall; i++) {
    
    if (type[i] == atp_OW || type[i]==atp_OH) {
      if(i==atom_o || atom->tag[i]==atom->tag[atom_o] || i!=atom->map(atom->tag[i])) continue;
      
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
 
      if (r_oo < cutoff_OO[1]) {	
        exp1 = exp(-b1*(r_oo-d_OO));
        
        if(bEVB3) {
          exp2_sum = 0.0;
	  
          for (int k = 0; k < 3; k++) {
            int h = atom_h[k];
	    
            // Correction JPCB, 112, 7146 (2008)

            dohhx[k] = x[oh][0] - x[h][0];
            dohhy[k] = x[oh][1] - x[h][1];
            dohhz[k] = x[oh][2] - x[h][2];
            domain->minimum_image(dohhx[k],dohhy[k],dohhz[k]);

            dowhx[k] = x[ow][0] - x[h][0];
            dowhy[k] = x[ow][1] - x[h][1];
            dowhz[k] = x[ow][2] - x[h][2];
            domain->minimum_image(dowhx[k],dowhy[k],dowhz[k]);

            dxhok[k] = (dohhx[k] + dowhx[k]) / 2.0;
            dyhok[k] = (dohhy[k] + dowhy[k]) / 2.0;
            dzhok[k] = (dohhz[k] + dowhz[k]) / 2.0;
            domain->minimum_image(dxhok[k],dyhok[k],dzhok[k]);

            r_ho2[k] = dxhok[k]*dxhok[k]+dyhok[k]*dyhok[k]+dzhok[k]*dzhok[k];
            exp2[k] = exp(-b2 * r_ho2[k]);
            exp2_sum += exp2[k];
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
        f[ow][0] -= dfx;
        dfy = tt * dyook;
        f[oh][1] += dfy;
        f[ow][1] -= dfy;
        dfz = tt * dzook;
        f[oh][2] += dfz;
        f[ow][2] -= dfz;

        // virial by r_oo, first term of Eq. 7 in JPCB 112(2008)467

        v[0] += dfx * dxook;
        v[1] += dfy * dyook;
        v[2] += dfz * dzook;
        v[3] += dfx * dyook;
        v[4] += dfx * dzook;
        v[5] += dfy * dzook;

        if (bEVB3) for (int k = 0; k < 3; k++) {
          int h = atom_h[k];
          tt = b2 *  B * exp1 * exp2[k];
          if (r_oo >= cutoff_OO[0]) tt *= sw;

          // force by R_HjOk, second term of Eq. 7 in JPCB 112(2008)467

          dfx = tt * dxhok[k];
          f[oh][0] += dfx;
          f[ow][0] += dfx;
          f[h][0] -= dfx * 2.0;
          dfy = tt * dyhok[k];
          f[oh][1] += dfy;
          f[ow][1] += dfy;
          f[h][1] -= dfy * 2.0;
          dfz = tt * dzhok[k];
          f[oh][2] += dfz;
          f[ow][2] += dfz;
          f[h][2] -= dfz * 2.0;

          // virial by R_HjOk, second term of Eq. 7 in JPCB 112(2008)467

          v[0] += dfx * (dohhx[k] + dowhx[k]);
          v[1] += dfy * (dohhy[k] + dowhy[k]);
          v[2] += dfz * (dohhz[k] + dowhz[k]);
          v[3] += dfx * (dohhy[k] + dowhy[k]);
          v[4] += dfx * (dohhz[k] + dowhz[k]);
          v[5] += dfy * (dohhz[k] + dowhz[k]);
        }
      }

      // energy, force, and virial by V_HOk_rep, Eq. 8 in JPCB 112(2008)467

      for (int k = 0; k < 3; k++) {
        int h = atom_h[k];
        dowhx[k] = x[ow][0] - x[h][0];
        dowhy[k] = x[ow][1] - x[h][1];
        dowhz[k] = x[ow][2] - x[h][2];

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
          f[ow][0] += dfx;
          f[h][0] -= dfx;
          dfy = tt * dowhy[k];
          f[ow][1] += dfy;
          f[h][1] -= dfy;
          dfz = tt * dowhz[k];
          f[ow][2] += dfz;
          f[h][2] -= dfz;

          v[0] += dfx * dowhx[k];
          v[1] += dfy * dowhy[k];
          v[2] += dfz * dowhz[k];
          v[3] += dfx * dowhy[k];
          v[4] += dfx * dowhz[k];
          v[5] += dfy * dowhz[k];
        }
      }

    }
  }

  energy = e_oo + e_ho;

  // Supplemental pairwise interactions

  int i, ii, itype, imol, j, jtype, jmol;
  double xtmp, ytmp, ztmp, rsq, r2inv, r;
  double delx, dely, delz, aexp, fpair, dfx, dfy, dfz;

  int inum = map[center_mol_id][0];

  energy += Vii_const;

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
/* switching function, JPCB 112(2008)467, Eq. 9                          */
/* ----------------------------------------------------------------------*/

double EVB_Rep_Hydronium_Expon::switching(double denominator, double rs, double rc, double rr)
{
  return 1.0-(rr-rs)*(rr-rs)*(3.0*rc-rs-2.0*rr)*denominator;
}

/* ----------------------------------------------------------------------*/
/* derivative of switching function, JPCB 112(2008)467, Eq. 9                          */
/* ----------------------------------------------------------------------*/

double EVB_Rep_Hydronium_Expon::dswitching(double denominator, double rs, double rc, double rr)
{
  return 6.0*(rr-rs)*(rc-rr)*denominator;
}

/*** SCI ***/

void EVB_Rep_Hydronium_Expon::sci_compute(int vflag)
{
  int* cplx_atom = evb_engine->complex_atom; 
  int istate = evb_complex->current_status;
  double cs2 = evb_complex->Cs2[istate];
  int **map = evb_engine->molecule_map;

  int atom_o = map[center_mol_id][1];  
  int atom_h[3];
  atom_h[0] = map[center_mol_id][2];
  atom_h[1] = map[center_mol_id][3];
  atom_h[2] = map[center_mol_id][4];

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int *molecule = atom->molecule;
  int nall = atom->nlocal+atom->nghost;
 
  int cplx_id = cplx_atom[atom_o];
  int atp_OH = type[atom_o];

  // Original MS-EVB3 definition

  for (int i = 0; i < nall; i++) {      
    if((type[i] == atp_OW || type[i] == atp_OH ) && cplx_atom[i]!=cplx_id && i==atom->map(atom->tag[i])) {
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

      if (r_oo < cutoff_OO[1]) {	
        exp1 = exp(-b1*(r_oo-d_OO));
        
        if(bEVB3) {
          exp2_sum = 0.0;
	  
          for (int k = 0; k < 3; k++) {
            int h = atom_h[k];

            dohhx[k] = x[oh][0] - x[h][0];
            dohhy[k] = x[oh][1] - x[h][1];
            dohhz[k] = x[oh][2] - x[h][2];
            domain->minimum_image(dohhx[k],dohhy[k],dohhz[k]);

            dowhx[k] = x[ow][0] - x[h][0];
            dowhy[k] = x[ow][1] - x[h][1];
            dowhz[k] = x[ow][2] - x[h][2];
            domain->minimum_image(dowhx[k],dowhy[k],dowhz[k]);

            dxhok[k] = (dohhx[k] + dowhx[k]) / 2.0;
            dyhok[k] = (dohhy[k] + dowhy[k]) / 2.0;
            dzhok[k] = (dohhz[k] + dowhz[k]) / 2.0;
            domain->minimum_image(dxhok[k],dyhok[k],dzhok[k]);

            r_ho2[k] = dxhok[k]*dxhok[k]+dyhok[k]*dyhok[k]+dzhok[k]*dzhok[k];
            exp2[k] = exp(-b2 * r_ho2[k]);
            exp2_sum += exp2[k];
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


        if (bEVB3) for (int k = 0; k < 3; k++) {

          int h = atom_h[k];
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

      // energy, force, and virial by V_HOk_rep, Eq. 8 in JPCB 112(2008)467

      for (int k = 0; k < 3; k++) {
        int h = atom_h[k];
        dowhx[k] = x[ow][0] - x[h][0];
        dowhy[k] = x[ow][1] - x[h][1];
        dowhz[k] = x[ow][2] - x[h][2];

        domain->minimum_image(dowhx[k],dowhy[k],dowhz[k]);
        r_ho = sqrt(dowhx[k]*dowhx[k] + dowhy[k]*dowhy[k] + dowhz[k]*dowhz[k]);
		
        if(r_ho <cutoff_HO[1]) {  
          ene = C * exp(-c1*(r_ho-d_OH));
          if(r_ho > cutoff_HO[0]) sw =  switching(ho_cutoff_1, cutoff_HO[0], cutoff_HO[1], r_ho);
           
          tt = c1 * ene / r_ho;
          if (r_ho >= cutoff_HO[0])
            tt = tt * sw + ene * dswitching(ho_cutoff_1, cutoff_HO[0], cutoff_HO[1], r_ho) / r_ho;
          
          tt *= cs2;

          dfx = tt * dowhx[k];
          f[ow][0] += dfx;
          dfy = tt * dowhy[k];
          f[ow][1] += dfy;
          dfz = tt * dowhz[k];
          f[ow][2] += dfz;
        }
      }
    }
  }

  // Supplemental pairwise interactions
  
  int i, ii, itype, j, jtype;
  double xtmp, ytmp, ztmp, rsq, r2inv, r;
  double delx, dely, delz, aexp, fpair, dfx, dfy, dfz;

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

int EVB_Rep_Hydronium_Expon::checkout(int* _index)
{
  int n = atom->ntypes;
  int index_max = 30;

  int count = 0;
  _index[count++] = -2 * num_pairs; // Write the first set of integers to file as is, then write as map[_index[j]]

  // Supplemental pairwise interactions

  for(int i=1; i<=n; i++) {
    for(int j=i; j<=n; j++) {
      if(setflag_pair[i][j]) {
	_index[count++] = i;
	_index[count++] = j;
      }
    }
  }
  
  // Original MS-EVB3 definition

  int* cplx_atom = evb_engine->complex_atom; 
  int **map = evb_engine->molecule_map;

  int atom_o = map[center_mol_id][1];  
  int atom_h[3];
  atom_h[0] = map[center_mol_id][2];
  atom_h[1] = map[center_mol_id][3];
  atom_h[2] = map[center_mol_id][4];

  double **x = atom->x;
  int *type = atom->type;
  int nall = atom->nlocal+atom->nghost;
 
  int atp_OH = type[atom_o];

  _index[count++] = atom_o;
  _index[count++] = atom_h[0];
  _index[count++] = atom_h[1];
  _index[count++] = atom_h[2];
  
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
      
      if (r_oo < cutoff_OO[1]) _index[count++] = ow;
      else { // If need be, calculate distance between each r_HH and r_OW
	int test = 0;
	for(int k=0; k<3; k++) {
	  int h = atom_h[k];
	  dowhx = x[h][0] - x[ow][0]; 
	  dowhy = x[h][1] - x[ow][1];
	  dowhz = x[h][2] - x[ow][2];
	  domain->minimum_image(dowhx,dowhy,dowhz);
	  r_ho = sqrt(dowhx*dowhx + dowhy*dowhy + dowhz*dowhz);
	  if(r_ho < cutoff_HO[1]) test = 1;
	}
	if(test) _index[count++] = ow;
      }
    }
  }

  if(count>index_max) error->all(FLERR,"Error: EVB_Rep_Expon::checkout  count>30.\n");  

  for(int i=count; i<index_max; i++) _index[i] = -1;
  return index_max;
}

