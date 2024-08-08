/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris (based on EVB_rep_hydronium and pair_table)
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "EVB_complex.h"
#include "EVB_type.h"
#include "EVB_engine.h"
#include "EVB_rep_hydroxide_fr_table.h"

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

#define LOOKUP 0
#define LINEAR 1  // Currently, the only supported version
#define SPLINE 2
#define BITMAP 3

#define R   1
#define RSQ 2
#define BMP 3

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_Rep_Hydroxide_FR_Table::EVB_Rep_Hydroxide_FR_Table(LAMMPS *lmp, EVB_Engine *engine) : EVB_Repulsive(lmp,engine)
{
  ntables = 0;
  tables = NULL;

  flag_SW = 0;
}

/* ---------------------------------------------------------------------- */

EVB_Rep_Hydroxide_FR_Table::~EVB_Rep_Hydroxide_FR_Table()
{
  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);
}

/* ----------------------------------------------------------------------*/

int EVB_Rep_Hydroxide_FR_Table::data_rep(char *buf, int *offset, int start, int end)
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

  if(comm->me == 0 && screen) fprintf(screen,"[EVB] Rep_Hydroxide_FR_Table\n");
  ntables = 2; // One for hydroxide oxygen and one for hydroxide hydrogen

  atp_OW = atoi(buf+offset[t++]);
  
  if(universe->me == 0) {
    fprintf(fp,"   Target atom type of hydroxide interaction: %i.\n\n",atp_OW);
    fprintf(fp,"   VJJ(ROO,RHO) = f(ROO) + g(RHO), which are tabulated.\n");
  }

  char *file = buf+offset[t++];
  char *tstyle = buf+offset[t++];
  if(strcmp(tstyle,"LINEAR") == 0) tabstyle = LINEAR;
  else error->all(FLERR,"EVB_Rep_Hydroxide_FR_Table: Unsupported table style");
  
  if(universe->me == 0) {
    fprintf(fp,"\n   Number of tabulated potentials: ntables= %i.\n",ntables);
    fprintf(fp,"   Name of table potential file: file= %s.\n",file);
    fprintf(fp,"   Interpolation style: tstyle= %s.\n",tstyle);
  }

  tablength = atoi( buf+offset[t++] );

  if(universe->me == 0) fprintf(fp,"\n   Number of grid points: tablength= %i.\n\n",tablength);

  char *keyword[2];
  for (int i=0; i<ntables; i++) {
    keyword[i] = buf+offset[t++];
    cutoff[i]  = atof( buf+offset[t++]);
  }
  bEVB3 = atoi( buf+offset[t++] );

  if(universe->me == 0) {
    fprintf(fp,"   Gaussian HO term: bEVB3= %i\n",bEVB3);
    if(bEVB3) fprintf(fp,"   Gaussian HO term scales f(ROO).\n");
  }

  // Stillinger-Weber Parameters
  if(comm->me ==0 && screen) fprintf(screen,"[EVB] Reading SW Parameters\n");
  flag_SW    = atoi(buf+offset[t++]);
  if(universe->me == 0) fprintf(fp,"\n   Compute OW-OH-OW Stillinger Weber interaction: flag_SW= %i\n",flag_SW);

  if(flag_SW) {
    epsilon_SW = atof(buf+offset[t++]);
    cut_SW      = atof(buf+offset[t++]);
    T0_SW      = atof(buf+offset[t++]);

    if(universe->me == 0) fprintf(fp,"\n      epsilon= %f\n      cut= %f      theta0= %f\n",epsilon_SW,cut_SW,T0_SW);
  }

  // Initialize MS-EVB Tables for hydroxide repulsion
  MPI_Comm_rank(world,&me);
 
  for (int i=0; i < ntables; i++) {
    tables = (Table *)
      memory -> srealloc(tables, (i+1)*sizeof(Table),"evb_rep_hydroxide_fr_table:tables"); 
    Table *tb = &tables[i];
    null_table(tb);
    if(me==0) read_table(tb,file,keyword[i]);
    bcast_table(tb);

    tb->cut = cutoff[i];
    tb->match = 0;
    if (tabstyle == LINEAR && tb->ninput == tablength && 
	tb->rflag == RSQ && tb->rhi == tb->cut) tb->match = 1;

    // spline read-in values and compute r,e,f vectors within table

    if (tb->match == 0) spline_table(tb);
    compute_table(tb);
  }
  
  // --------------------------------------------------------------
  
  return t;
}

/* ----------------------------------------------------------------------*/
/*   repulsive term for diagonal state, see JPCB 112(2008)467, Eq. 7-9   */
/*     note: the definition of q_HjOk is described in JPCB 112(2008)7146 */
/* ----------------------------------------------------------------------*/

void EVB_Rep_Hydroxide_FR_Table::compute(int vflag)
{  
  double *v = virial;                     // virial
  memset(v,0,sizeof(double)*6);

  energy = e_oo = e_ho = 0;
  int **map = evb_engine->molecule_map;
  
  int atom_o = map[center_mol_id][1];

  // SW array
  int num_sw_list = 0;
  int sw_list[100];
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int *molecule = atom->molecule;
  int nall = atom->nlocal+atom->nghost;
  int atp_OH = type[atom_o];

  for (int i = 0; i < nall; i++) {
    
    if (type[i] == atp_OW || type[i]==atp_OH) {
      if( i == atom_o || atom->tag[i] == atom->tag[atom_o] || i != atom->map(atom->tag[i]) ) continue;
      
      int oh = atom_o, ow = i;
      double dxook,dyook,dzook,dxhok[3],dyhok[3],dzhok[3],ene;
      double dohhx[3],dohhy[3],dohhz[3];
      double dowhx[3],dowhy[3],dowhz[3];
      double r_oo, r_ho, tt;
      double r_ho2[3];
      double exp1,exp2[3],exp2_sum;
      double fo[3],fh[3],fok[3],fhj[3][3];
	
      Table *tb;
      int tlm1 = tablength - 1;
      int itable;
      double rsq,value,fraction;
      double f_R,df_R,g_q[3],dg_q[3],g_q_sum;	
      // calculate distance between r_OH and r_OW

      dxook = x[oh][0]-x[ow][0];
      dyook = x[oh][1]-x[ow][1];
      dzook = x[oh][2]-x[ow][2];

      domain->minimum_image(dxook,dyook,dzook);
      rsq  = dxook*dxook+dyook*dyook+dzook*dzook;
      r_oo = sqrt(rsq);

      // Add to SW neighborlist?
      if(flag_SW && r_oo < cut_SW) sw_list[num_sw_list++] = i;

      if (r_oo < cutoff[0]) {	
	tb = &tables[0];
	if(rsq < tb->innersq) error->one(FLERR,"Hydroxide Oxygen Repulsion distance #1 < table inner cutoff");
	if(tabstyle == LINEAR) {
	  itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	  if (itable >= tlm1) error->one(FLERR,"Hydroxide Oxygen Repulsion distance #1 > table outer cutoff");
	  fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
	  f_R  = tb->e[itable] + fraction*tb->de[itable];
	  df_R = tb->f[itable] + fraction*tb->df[itable];
	}
	
	ene = f_R;
	e_oo += ene;
	
	// force by r_oo
	
	tt = df_R;

	dfx = tt * dxook;
	f[oh][0] += dfx;
	f[ow][0] -=dfx;
	dfy = tt * dyook;
	f[oh][1] += dfy;
	f[ow][1] -=dfy;
	dfz = tt * dzook;
	f[oh][2] += dfz;
	f[ow][2] -=dfz;
	
	// virial by r_oo
	
	v[0] += dfx * dxook;
	v[1] += dfy * dyook;
	v[2] += dfz * dzook;
	v[3] += dfx * dyook;
	v[4] += dfx * dzook;
	v[5] += dfy * dzook;
	
	// energy, force, and virial by V_HOk_rep

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
	    rsq  = dowhx[k]*dowhx[k] + dowhy[k]*dowhy[k] + dowhz[k]*dowhz[k];
	    r_ho = sqrt(rsq);
	    
	    if (r_ho < cutoff[1]) {
	      tb = &tables[1];
	      if(rsq < tb->innersq) error->one(FLERR,"Hydroxide Hydrogen Repulsion distance #3 < table inner cutoff");
	      if(tabstyle == LINEAR) {
		itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
		if (itable >= tlm1) error->one(FLERR,"Hydroxide Hydrogen Repulsion distance #3 > table outer cutoff");
		fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
		f_R  = tb->e[itable] + fraction*tb->de[itable];
		df_R = tb->f[itable] + fraction*tb->df[itable];
	      }
	    
	      ene   = f_R;
	      e_ho += ene;
	      
	      tt = df_R;
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
      }
      
    } // if right atom type
  } // Loop over all atoms
  
  // SW potential: Based on pair_sw::compute() and pair_sw::threebody()
  double e_sw = 0.0;
  if(flag_SW) {
    double del1x,del1y,del1z,r1sq,r1,rinvsq1,rainv1,gsrainv1,gsrainvsq1,expgsrainv1;
    double del2x,del2y,del2z,r2sq,r2,rinvsq2,rainv2,gsrainv2,gsrainvsq2,expgsrainv2;
    double rinv12,cs,delcs,delcssq;
    double facexp,facrad,frad1,frad2,facang,facang12,csfacang,csfac1,csfac2;
    double fj[3],fk[3];
    
    double xtmp = x[atom_o][0];
    double ytmp = x[atom_o][1];
    double ztmp = x[atom_o][2];
    for(int i=0; i<num_sw_list-1; i++) {
      int indx = sw_list[i];
      del1x = x[indx][0] - xtmp;
      del1y = x[indx][1] - ytmp;
      del1z = x[indx][2] - ztmp;
      domain->minimum_image(del1x,del1y,del1z);
      
      r1sq = del1x*del1x + del1y*del1y + del1z*del1z;
      r1 = sqrt(r1sq);
      rinvsq1     = 1.0 / r1sq;
      rainv1      = 1.0 / (r1 - cut_SW);
      gsrainv1    = cut_SW * rainv1;
      gsrainvsq1  = gsrainv1 * rainv1 / r1;
      expgsrainv1 = exp(gsrainv1);

      int num_OHi = 1;
      if(type[indx] == atp_OH) num_OHi++;
      
      for(int j=i+1; j<num_sw_list; j++) {
	int jndx = sw_list[j];
	del2x = x[jndx][0] - xtmp;
	del2y = x[jndx][1] - ytmp;
	del2z = x[jndx][2] - ztmp;
	domain->minimum_image(del2x,del2y,del2z);

	// Because repulsion separately computed for each hydroxide, scale by number of hydroxides 
	// in triplet to prevent over-counting.
	int num_OH = num_OHi;
	if(type[jndx] == atp_OH) num_OH++;
	const double rnum_OH = 1.0 / double(num_OH);
	
	r2sq = del2x*del2x + del2y*del2y + del2z*del2z;
	r2 = sqrt(r2sq);
	rinvsq2     = 1.0 / r2sq;
	rainv2      = 1.0 / (r2 - cut_SW);
	gsrainv2    = cut_SW * rainv2;
	gsrainvsq2  = gsrainv2 * rainv2 / r2;
	expgsrainv2 = exp(gsrainv2);
	
	rinv12 = 1.0 / (r1 * r2);
	cs = (del1x*del2x + del1y*del2y + del1z*del2z) * rinv12;
	delcs = cs - T0_SW;
	delcssq = delcs * delcs;
	
	facexp = expgsrainv1 * expgsrainv2;
	
	facrad   = rnum_OH * epsilon_SW * facexp * delcssq;
	frad1    = facrad * gsrainvsq1;
	frad2    = facrad * gsrainvsq2;
	facang   = rnum_OH * epsilon_SW * 2.0 * facexp * delcs;
	facang12 = rinv12 * facang;
	csfacang = cs * facang;
	csfac1   = rinvsq1 * csfacang;
	
	fj[0] = del1x * (frad1 + csfac1) - del2x * facang12;
	fj[1] = del1y * (frad1 + csfac1) - del2y * facang12;
	fj[2] = del1z * (frad1 + csfac1) - del2z * facang12;

	csfac2 = rinvsq2 * csfacang;
	
	fk[0] = del2x * (frad2 + csfac2) - del1x * facang12;
	fk[1] = del2y * (frad2 + csfac2) - del1y * facang12;
	fk[2] = del2z * (frad2 + csfac2) - del1z * facang12;
	
	e_sw += facrad;
      
	f[atom_o][0] -= fj[0] + fk[0];
	f[atom_o][1] -= fj[1] + fk[1];
	f[atom_o][2] -= fj[2] + fk[2];
	
	f[indx][0] += fj[0];
	f[indx][1] += fj[1];
	f[indx][2] += fj[2];
	
	f[jndx][0] += fk[0];
	f[jndx][1] += fk[1];
	f[jndx][2] += fk[2];
      }
    }
  } // if(flag_SW)
    
  energy = e_oo + e_ho + e_sw;
}

/* ----------------------------------------------------------------------*/

void EVB_Rep_Hydroxide_FR_Table::scan_potential_surface()
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
    if(type[i]==atp_OW) {
      target = i;
      break;
    }
  
  double d[3];
  for(int i=0; i<3; i++)  d[i] = x[target][i]-x[atom_o][i];
  double dr = sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
  double c[3];
  for(int i=0; i<3; i++)  c[i] = d[i]/dr;
  
  for(int i=0; i<nsample+2; i++) {
    for(int j=0; j<3; j++) f[target][j]=0.0;
    for(int j=0; j<3; j++) x[target][j]=x[atom_o][j]+c[j]*r[i];
    
    compute(false);
    
    e[i]=energy; e1[i]=e_oo; e2[i]=e_ho;
    for(int j=0; j<3; j++) fi[j][i]=f[target][j];
    fr[i] = sqrt(fi[0][i]*fi[0][i]+fi[1][i]*fi[1][i]+fi[2][i]*fi[2][i]);
  }
  
  for(int i=1; i<=nsample; i++) {
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

/*** SCI ***/

void EVB_Rep_Hydroxide_FR_Table::sci_compute(int vflag)
{
  int *cplx_atom = evb_engine->complex_atom;
  int istate = evb_complex->current_status;
  double cs2 = evb_complex->Cs2[istate];
  int **map = evb_engine->molecule_map;
  
  int atom_o = map[center_mol_id][1];

  // SW array
  int num_sw_list = 0;
  int sw_list[100];
  int sw_list_cplx[100];
  
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int *molecule = atom->molecule;
  int nall = atom->nlocal+atom->nghost;

  int cplx_id = cplx_atom[atom_o];
  int atp_OH = type[atom_o];

  // The SW interaction needs to be carefully checked.
  
  for (int i = 0; i < nall; i++) {
    
    if( (type[i] == atp_OW || type[i]==atp_OH) ) {
      if( i == atom_o || atom->tag[i] == atom->tag[atom_o] || i != atom->map(atom->tag[i]) ) continue;
      
      int oh = atom_o, ow = i;
      double dxook,dyook,dzook,rsq,r_oo;

      // calculate distance between r_OH and r_OW
      dxook = x[oh][0]-x[ow][0];
      dyook = x[oh][1]-x[ow][1];
      dzook = x[oh][2]-x[ow][2];

      domain->minimum_image(dxook,dyook,dzook);
      rsq  = dxook*dxook+dyook*dyook+dzook*dzook;
      r_oo = sqrt(rsq);

      // Add to SW neighborlist? This is a full list regardless of whether atom is in/out of complex.
      if(flag_SW && r_oo < cut_SW) {
	sw_list[num_sw_list] = i;

	// Keep track of which atoms are outside of complex. If inside, then skip rest of iteration.
	if(cplx_atom[i]==cplx_id) sw_list_cplx[num_sw_list++] = 0;
	else sw_list_cplx[num_sw_list++] = 1;
      }
      if(cplx_atom[i] == cplx_id) continue;

      double dxhok[3],dyhok[3],dzhok[3],ene;
      double dohhx[3],dohhy[3],dohhz[3];
      double dowhx[3],dowhy[3],dowhz[3];
      double r_ho, tt;
      double r_ho2[3];
      double exp1,exp2[3],exp2_sum;
      double fo[3],fh[3],fok[3],fhj[3][3];
	
      Table *tb;
      int tlm1 = tablength - 1;
      int itable;
      double value,fraction;
      double f_R,df_R,g_q[3],dg_q[3],g_q_sum;

      if (r_oo < cutoff[0]) {	
	tb = &tables[0];
	if(rsq < tb->innersq) error->one(FLERR,"Hydroxide Oxygen Repulsion distance #1 < table inner cutoff");
	if(tabstyle == LINEAR) {
	  itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
	  if (itable >= tlm1) error->one(FLERR,"Hydroxide Oxygen Repulsion distance #1 > table outer cutoff");
	  fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
	  df_R = tb->f[itable] + fraction*tb->df[itable];
	}
	
	// force by r_oo
	
	tt = df_R * cs2;

	dfx = tt * dxook;
	f[ow][0] -=dfx;
	dfy = tt * dyook;
	f[ow][1] -=dfy;
	dfz = tt * dzook;
	f[ow][2] -=dfz;
		
	// energy, force, and virial by V_HOk_rep

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
	    rsq  = dowhx[k]*dowhx[k] + dowhy[k]*dowhy[k] + dowhz[k]*dowhz[k];
	    r_ho = sqrt(rsq);
	    
	    if (r_ho < cutoff[1]) {
	      tb = &tables[1];
	      if(rsq < tb->innersq) error->one(FLERR,"Hydroxide Hydrogen Repulsion distance #3 < table inner cutoff");
	      if(tabstyle == LINEAR) {
		itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
		if (itable >= tlm1) error->one(FLERR,"Hydroxide Hydrogen Repulsion distance #3 > table outer cutoff");
		fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
		df_R = tb->f[itable] + fraction*tb->df[itable];
	      }
	      
	      tt = df_R * cs2;
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
      
    } // if right atom type
  } // Loop over all atoms
  
  // SW potential: Based on pair_sw::compute() and pair_sw::threebody()
  double e_sw = 0.0;
  if(flag_SW) {
    double del1x,del1y,del1z,r1sq,r1,rinvsq1,rainv1,gsrainv1,gsrainvsq1,expgsrainv1;
    double del2x,del2y,del2z,r2sq,r2,rinvsq2,rainv2,gsrainv2,gsrainvsq2,expgsrainv2;
    double rinv12,cs,delcs,delcssq;
    double facexp,facrad,frad1,frad2,facang,facang12,csfacang,csfac1,csfac2;
    double fj[3],fk[3];
    
    double xtmp = x[atom_o][0];
    double ytmp = x[atom_o][1];
    double ztmp = x[atom_o][2];
    for(int i=0; i<num_sw_list-1; i++) {
      int indx = sw_list[i];
      del1x = x[indx][0] - xtmp;
      del1y = x[indx][1] - ytmp;
      del1z = x[indx][2] - ztmp;
      domain->minimum_image(del1x,del1y,del1z);
      
      r1sq = del1x*del1x + del1y*del1y + del1z*del1z;
      r1 = sqrt(r1sq);
      rinvsq1     = 1.0 / r1sq;
      rainv1      = 1.0 / (r1 - cut_SW);
      gsrainv1    = cut_SW * rainv1;
      gsrainvsq1  = gsrainv1 * rainv1 / r1;
      expgsrainv1 = exp(gsrainv1);

      int num_OHi = 1;
      if(type[indx] == atp_OH) num_OHi++;

      for(int j=i+1; j<num_sw_list; j++) {
	int jndx = sw_list[j];
	del2x = x[jndx][0] - xtmp;
	del2y = x[jndx][1] - ytmp;
	del2z = x[jndx][2] - ztmp;
	domain->minimum_image(del2x,del2y,del2z);
	
	// Because repulsion separately computed for each hydroxide, scale by number of hydroxides 
	// in triplet to prevent over-counting.
	int num_OH = num_OHi;
	if(type[jndx] == atp_OH) num_OH++;
	const double rnum_OH = 1.0 / double(num_OH);

	r2sq = del2x*del2x + del2y*del2y + del2z*del2z;
	r2 = sqrt(r2sq);
	rinvsq2     = 1.0 / r2sq;
	rainv2      = 1.0 / (r2 - cut_SW);
	gsrainv2    = cut_SW * rainv2;
	gsrainvsq2  = gsrainv2 * rainv2 / r2;
	expgsrainv2 = exp(gsrainv2);
	
	rinv12 = 1.0 / (r1 * r2);
	cs = (del1x*del2x + del1y*del2y + del1z*del2z) * rinv12;
	delcs = cs - T0_SW;
	delcssq = delcs * delcs;
	
	facexp = expgsrainv1 * expgsrainv2;
	
	facrad   = rnum_OH * epsilon_SW * facexp * delcssq;
	frad1    = facrad * gsrainvsq1;
	frad2    = facrad * gsrainvsq2;
	facang   = rnum_OH * epsilon_SW * 2.0 * facexp * delcs;
	facang12 = rinv12 * facang;
	csfacang = cs * facang;
	csfac1   = rinvsq1 * csfacang;
	
	fj[0] = del1x * (frad1 + csfac1) - del2x * facang12;
	fj[1] = del1y * (frad1 + csfac1) - del2y * facang12;
	fj[2] = del1z * (frad1 + csfac1) - del2z * facang12;

	csfac2 = rinvsq2 * csfacang;
	
	fk[0] = del2x * (frad2 + csfac2) - del1x * facang12;
	fk[1] = del2y * (frad2 + csfac2) - del1y * facang12;
	fk[2] = del2z * (frad2 + csfac2) - del1z * facang12;
	
	// Only add force if atom outside of current complex.
	if(sw_list_cplx[i]) {
	  f[indx][0] += cs2 * fj[0];
	  f[indx][1] += cs2 * fj[1];
	  f[indx][2] += cs2 * fj[2];
	}
	
	if(sw_list_cplx[j]) {
	  f[jndx][0] += cs2 * fk[0];
	  f[jndx][1] += cs2 * fk[1];
	  f[jndx][2] += cs2 * fk[2];
	}
      }
    }
  } // if(flag_SW)
}

int EVB_Rep_Hydroxide_FR_Table::checkout(int* _index)
{
  int index_max = 30;
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
      if (r_oo < cutoff[0]) test = 1;
      else { // If need be, calculate distance between each r_HH and r_OW
	for(int k=0; k<nA; k++) {
	  int hw = map[w][k+1];
	  if(type[hw] != atp_OW) {
	    dowhx = x[oh][0] - x[hw][0]; 
	    dowhy = x[oh][1] - x[hw][1];
	    dowhz = x[oh][2] - x[hw][2];
	    domain->minimum_image(dowhx,dowhy,dowhz);
	    r_ho = sqrt(dowhx*dowhx + dowhy*dowhy + dowhz*dowhz);
	    if(r_ho < cutoff[1]) test = 1;
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
  
  if(count>index_max) error->all(FLERR,"Warning: EVB_rep_hydroxide_FR_Table::checkout  count>index_max.\n");
  
  for(int i=count; i<index_max; i++) _index[i] = -1;
  return index_max;
}

/* ----------------------------------------------------------------------
   All of the routines below were copied from the table pairstyle.
   There have been some minor changes: only linear is supported
   ---------------------------------------------------------------------- */



/* ----------------------------------------------------------------------
   set all ptrs in a table to NULL, so can be freed safely
------------------------------------------------------------------------- */

void EVB_Rep_Hydroxide_FR_Table::null_table(Table *tb)
{
  tb->rfile = tb->efile = tb->ffile = NULL;
  tb->e2file = tb->f2file = NULL;
  tb->rsq = tb->drsq = tb->e = tb->de = NULL;
  tb->f = tb->df = tb->e2 = tb->f2 = NULL;
}
/* ----------------------------------------------------------------------*/

void EVB_Rep_Hydroxide_FR_Table::free_table(Table *tb)
{
  memory->sfree(tb->rfile);
  memory->sfree(tb->efile);
  memory->sfree(tb->ffile);
  memory->sfree(tb->e2file);
  memory->sfree(tb->f2file);

  memory->sfree(tb->rsq);
  memory->sfree(tb->drsq);
  memory->sfree(tb->e);
  memory->sfree(tb->de);
  memory->sfree(tb->f);
  memory->sfree(tb->df);
  memory->sfree(tb->e2);
  memory->sfree(tb->f2);
}

/* ----------------------------------------------------------------------*/

void EVB_Rep_Hydroxide_FR_Table::read_table(Table *tb, char *file, char *keyword)
{
  char line[MAXLINE];

  // open file

  FILE *fp = fopen(file,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open file %s",file);
    error->one(FLERR,str);
  }
  
  if(comm->me==0 && screen) fprintf(screen,"[EVB] Looking for keyword: %s",keyword);

  while (1) {
    if (fgets(line,MAXLINE,fp) == NULL)
      error->one(FLERR,"Did not find keyword in table file");
    if (strspn(line," \t\n") == strlen(line)) continue;    // blank line
    if (line[0] == '#') continue;                          // comment
    if (strstr(line,keyword) == line) break;               // matching keyword
    fgets(line,MAXLINE,fp);                         // no match, skip section
    param_extract(tb,line);
    fgets(line,MAXLINE,fp);
    for (int i = 0; i < tb->ninput; i++) fgets(line,MAXLINE,fp);
  }

  // read args on 2nd line of section
  // allocate table arrays for file values

  fgets(line,MAXLINE,fp);
  param_extract(tb,line);
  tb->rfile = (double *) 
    memory->smalloc(tb->ninput*sizeof(double),"evb_rep_hydroxide_fr_table:rfile");
  tb->efile = (double *) 
    memory->smalloc(tb->ninput*sizeof(double),"evb_rep_hydroxide_fr_table:efile");
  tb->ffile = (double *) 
    memory->smalloc(tb->ninput*sizeof(double),"evb_rep_hydroxide_fr_table:ffile");

  // read r,e,f table values from file
  // if rflag set, compute r
  // if rflag not set, use r from file

  int itmp;
  double rtmp;

  fgets(line,MAXLINE,fp);
  for (int i = 0; i < tb->ninput; i++) {
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d %lg %lg %lg",&itmp,&rtmp,&tb->efile[i],&tb->ffile[i]);

    if (tb->rflag == R)
      rtmp = tb->rlo + (tb->rhi - tb->rlo)*i/(tb->ninput-1);
    else if (tb->rflag == RSQ) {
      rtmp = tb->rlo*tb->rlo + 
	(tb->rhi*tb->rhi - tb->rlo*tb->rlo)*i/(tb->ninput-1);
      rtmp = sqrt(rtmp);
    }

    tb->rfile[i] = rtmp;
  }

  // close file

  fclose(fp);
}

/* ----------------------------------------------------------------------
   broadcast read-in table info from proc 0 to other procs
   this function communicates these values in Table:
     ninput,rfile,efile,ffile,rflag,rlo,rhi,fpflag,fplo,fphi
------------------------------------------------------------------------- */

void EVB_Rep_Hydroxide_FR_Table::bcast_table(Table *tb)
{
  MPI_Bcast(&tb->ninput,1,MPI_INT,0,world);

  int me;
  MPI_Comm_rank(world,&me);
  if (me > 0) {
    tb->rfile = (double *) 
      memory->smalloc(tb->ninput*sizeof(double),"evb_rep_hydroxide_fr_table:rfile");
    tb->efile = (double *) 
      memory->smalloc(tb->ninput*sizeof(double),"evb_rep_hydroxide_fr_table:efile");
    tb->ffile = (double *) 
      memory->smalloc(tb->ninput*sizeof(double),"evb_rep_hydroxide_fr_table:ffile");
  }

  MPI_Bcast(tb->rfile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->efile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->ffile,tb->ninput,MPI_DOUBLE,0,world);

  MPI_Bcast(&tb->rflag,1,MPI_INT,0,world);
  if (tb->rflag) {
    MPI_Bcast(&tb->rlo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->rhi,1,MPI_DOUBLE,0,world);
  }
  MPI_Bcast(&tb->fpflag,1,MPI_INT,0,world);
  if (tb->fpflag) {
    MPI_Bcast(&tb->fplo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->fphi,1,MPI_DOUBLE,0,world);
  }
}

/* ----------------------------------------------------------------------
   extract attributes from parameter line in table section
   format of line: N value R/RSQ/BITMAP lo hi FP fplo fphi
   N is required, other params are optional
------------------------------------------------------------------------- */

void EVB_Rep_Hydroxide_FR_Table::param_extract(Table *tb, char *line)
{
  tb->ninput = 0;
  tb->rflag = 0;
  tb->fpflag = 0;
  
  char *word = strtok(line," \t\n\r\f");
  while (word) {
    if (strcmp(word,"N") == 0) {
      word = strtok(NULL," \t\n\r\f");
      tb->ninput = atoi(word);
    } else if (strcmp(word,"R") == 0 || strcmp(word,"RSQ") == 0 ||
	       strcmp(word,"BITMAP") == 0) {
      if (strcmp(word,"R") == 0) tb->rflag = R;
      else if (strcmp(word,"RSQ") == 0) tb->rflag = RSQ;
      else if (strcmp(word,"BITMAP") == 0) tb->rflag = BMP;
      word = strtok(NULL," \t\n\r\f");
      tb->rlo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->rhi = atof(word);
    } else if (strcmp(word,"FP") == 0) {
      tb->fpflag = 1;
      word = strtok(NULL," \t\n\r\f");
      tb->fplo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->fphi = atof(word);
    } else {
      printf("WORD: %s\n",word);
      error->one(FLERR,"Invalid keyword in pair table parameters");
    }
    word = strtok(NULL," \t\n\r\f");
  }

  if (tb->ninput == 0) error->one(FLERR,"Pair table parameters did not set N");
}

/* ----------------------------------------------------------------------
   compute r,e,f vectors from splined values
------------------------------------------------------------------------- */

void EVB_Rep_Hydroxide_FR_Table::compute_table(Table *tb)
{
  if(comm->me==0 && screen) fprintf(screen,"  Computing Table\n");
  double tol_zero = 0.0000001; // For when the grid point 0.0 is included in table.
  int tlm1 = tablength-1;

  // inner = inner table bound
  // cut = outer table bound
  // delta = table spacing in rsq for N-1 bins

  double inner;
  if (tb->rflag) inner = tb->rlo;
  else inner = tb->rfile[0];
  tb->innersq = inner*inner;
  tb->delta = (tb->cut*tb->cut - tb->innersq) / tlm1;
  tb->invdelta = 1.0/tb->delta;

  // linear tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // de,df values = delta from lower edge to upper edge of bin
  // rsq,e,f are N in length so de,df arrays can compute difference
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == LINEAR) {
    tb->rsq = (double *) memory->smalloc(tablength*sizeof(double),"evb_rep_hydroxide_fr_table:rsq");
    tb->e = (double *) memory->smalloc(tablength*sizeof(double),"evb_rep_hydroxide_fr_table:e");
    tb->f = (double *) memory->smalloc(tablength*sizeof(double),"evb_rep_hydroxide_fr_table:f");
    tb->de = (double *) memory->smalloc(tlm1*sizeof(double),"evb_rep_hydroxide_fr_table:de");
    tb->df = (double *) memory->smalloc(tlm1*sizeof(double),"evb_rep_hydroxide_fr_table:df");

    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
	tb->e[i] = tb->efile[i];
	if(r < tol_zero) r = tol_zero;
	tb->f[i] = tb->ffile[i]/r;
      } else {
	tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
	if(r < tol_zero) r = tol_zero;
	tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
    }
    
    for (int i = 0; i < tlm1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
    }
  }

} 

/* ----------------------------------------------------------------------
   spline and splint routines modified from Numerical Recipes
------------------------------------------------------------------------- */

void EVB_Rep_Hydroxide_FR_Table::spline(double *x, double *y, int n,
		       double yp1, double ypn, double *y2)
{
  int i,k;
  double p,qn,sig,un;
  double *u = new double[n];

  if (yp1 > 0.99e30) y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0]) / (x[1]-x[0]) - yp1);
  }
  for (i = 1; i < n-1; i++) {
    sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
    p = sig*y2[i-1] + 2.0;
    y2[i] = (sig-1.0) / p;
    u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1]);
    u[i] = (6.0*u[i] / (x[i+1]-x[i-1]) - sig*u[i-1]) / p;
  }
  if (ypn > 0.99e30) qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0/(x[n-1]-x[n-2])) * (ypn - (y[n-1]-y[n-2]) / (x[n-1]-x[n-2]));
  }
  y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2] + 1.0);
  for (k = n-2; k >= 0; k--) y2[k] = y2[k]*y2[k+1] + u[k];

  delete [] u;
}

/* ---------------------------------------------------------------------- */

double EVB_Rep_Hydroxide_FR_Table::splint(double *xa, double *ya, double *y2a, int n, double x)
{
  int klo,khi,k;
  double h,b,a,y;

  klo = 0;
  khi = n-1;
  while (khi-klo > 1) {
    k = (khi+klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  h = xa[khi]-xa[klo];
  a = (xa[khi]-x) / h;
  b = (x-xa[klo]) / h;
  y = a*ya[klo] + b*ya[khi] + 
    ((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi]) * (h*h)/6.0;
  return y;
}

/* ----------------------------------------------------------------------
   build spline representation of e,f over entire range of read-in table
   this function sets these values in Table: e2file,f2file
------------------------------------------------------------------------- */

void EVB_Rep_Hydroxide_FR_Table::spline_table(Table *tb)
{
  tb->e2file = (double *) 
    memory->smalloc(tb->ninput*sizeof(double),"evb_rep_hydroxide_fr_table:e2file");
  tb->f2file = (double *) 
    memory->smalloc(tb->ninput*sizeof(double),"evb_rep_hydroxide_fr_table:f2file");

  double ep0 = - tb->ffile[0];
  double epn = - tb->ffile[tb->ninput-1];
  spline(tb->rfile,tb->efile,tb->ninput,ep0,epn,tb->e2file);

  if (tb->fpflag == 0) {
    tb->fplo = (tb->ffile[1] - tb->ffile[0]) / (tb->rfile[1] - tb->rfile[0]);
    tb->fphi = (tb->ffile[tb->ninput-1] - tb->ffile[tb->ninput-2]) / 
      (tb->rfile[tb->ninput-1] - tb->rfile[tb->ninput-2]);
  }

  double fp0 = tb->fplo;
  double fpn = tb->fphi;
  spline(tb->rfile,tb->ffile,tb->ninput,fp0,fpn,tb->f2file);
}
