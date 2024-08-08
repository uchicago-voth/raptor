/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: chris
  
   This is derived from the table and lj/cut/coul/long pairstyles.
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_table_lj_cut_coul_long.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "kspace.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

#define R   1
#define RSQ 2
#define BMP 3

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairTableLJCutCoulLong::PairTableLJCutCoulLong(LAMMPS *lmp) : Pair(lmp)
{
  ewaldflag = pppmflag = 1;
  respa_enable = 1;
  ftable = NULL;
  qdist = 0.0;
  
  ntables = 0;
  tables = NULL;
}

/* ---------------------------------------------------------------------- */

PairTableLJCutCoulLong::~PairTableLJCutCoulLong()
{
  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(tabflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);

    memory->destroy(tabindex);
  }
  if (ftable) free_tables();
}

/* ---------------------------------------------------------------------- */

void PairTableLJCutCoulLong::compute(int eflag, int vflag)
{ 
  int i,j,ii,jj,inum,jnum,itype,jtype,itable;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double fraction,table;
  double r,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double grij,expm2,prefactor,t,erfc;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double rsq;

  Table *tb;
  double value,a,b;
  union_int_float_t rsq_lookup;
  int tlm1 = tablength - 1;

  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
	r2inv = 1.0/rsq;

	if (rsq < cut_coulsq) {
	  if (!ncoultablebits || rsq <= tabinnersq) {
	    r = sqrt(rsq);
	    grij = g_ewald * r;
	    expm2 = exp(-grij*grij);
	    t = 1.0 / (1.0 + EWALD_P*grij);
	    erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
	    prefactor = qqrd2e * qtmp*q[j]/r;
	    forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
	    if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
	  } else {
	    rsq_lookup.f = rsq;
	    itable = rsq_lookup.i & ncoulmask;
	    itable >>= ncoulshiftbits;
	    fraction = (rsq_lookup.f - rtable[itable]) * drtable[itable];
	    table = ftable[itable] + fraction*dftable[itable];
	    forcecoul = qtmp*q[j] * table;
	    if (factor_coul < 1.0) {
	      table = ctable[itable] + fraction*dctable[itable];
	      prefactor = qtmp*q[j] * table;
	      forcecoul -= (1.0-factor_coul)*prefactor;
	    }
	  }
	} else forcecoul = 0.0;

	if (rsq < cut_ljsq[itype][jtype]) {
	  r6inv = r2inv*r2inv*r2inv;
	  forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
	} else forcelj = 0.0;

	fpair = (forcecoul + factor_lj*forcelj) * r2inv;

	f[i][0] += delx*fpair;
	f[i][1] += dely*fpair;
	f[i][2] += delz*fpair;
	if (newton_pair || j < nlocal) {
	  f[j][0] -= delx*fpair;
	  f[j][1] -= dely*fpair;
	  f[j][2] -= delz*fpair;
	}

	if (eflag) {
	  if (rsq < cut_coulsq) {
	    if (!ncoultablebits || rsq <= tabinnersq)
	      ecoul = prefactor*erfc;
	    else {
	      table = etable[itable] + fraction*detable[itable];
	      ecoul = qtmp*q[j] * table;
	    }
	    if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
	  } else ecoul = 0.0;

	  if (rsq < cut_ljsq[itype][jtype]) {
	    evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
	      offset[itype][jtype];
	    evdwl *= factor_lj;
	  } else evdwl = 0.0;
	}

	if (evflag) ev_tally(i,j,nlocal,newton_pair,
			     evdwl,ecoul,fpair,delx,dely,delz);

	// Add contribution from tabulated potential
  	if(tabflag[itype][jtype] && rsq < cut_ljsq[itype][jtype]) { 
  	  tb = &tables[tabindex[itype][jtype]];
  	  if (rsq < tb->innersq) {
	    fprintf(stdout,"\nTable %i   r = %f\n",tabindex[itype][jtype],sqrt(rsq));
  	    error->one(FLERR,"Pair distance < table inner cutoff");
	  }
 	  
  	  if (tabstyle == LOOKUP) {
  	    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
  	    if (itable >= tlm1)
  	      error->one(FLERR,"Pair distance > table outer cutoff");
  	    fpair = factor_lj * tb->f[itable];
  	  } else if (tabstyle == LINEAR) {
  	    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
  	    if (itable >= tlm1)
  	      error->one(FLERR,"Pair distance > table outer cutoff");
  	    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
  	    value = tb->f[itable] + fraction*tb->df[itable];
  	    fpair = factor_lj * value;
  	  } else if (tabstyle == SPLINE) {
  	    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
  	    if (itable >= tlm1)
  	      error->one(FLERR,"Pair distance > table outer cutoff");
  	    b = (rsq - tb->rsq[itable]) * tb->invdelta;
  	    a = 1.0 - b;
  	    value = a * tb->f[itable] + b * tb->f[itable+1] + 
  	      ((a*a*a-a)*tb->f2[itable] + (b*b*b-b)*tb->f2[itable+1]) * 
  	      tb->deltasq6;
  	    fpair = factor_lj * value;
  	  } else {
  	    rsq_lookup.f = rsq;
  	    itable = rsq_lookup.i & tb->nmask;
  	    itable >>= tb->nshiftbits;
  	    fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
  	    value = tb->f[itable] + fraction*tb->df[itable];
  	    fpair = factor_lj * value;
  	  }
 	  
  	  f[i][0] += delx*fpair;
  	  f[i][1] += dely*fpair;
  	  f[i][2] += delz*fpair;
  	  if (newton_pair || j < nlocal) {
  	    f[j][0] -= delx*fpair;
  	    f[j][1] -= dely*fpair;
  	    f[j][2] -= delz*fpair;
  	  }
 	  
  	  if (eflag) {
  	    if (tabstyle == LOOKUP)
  	      evdwl = tb->e[itable];
  	    else if (tabstyle == LINEAR || tabstyle == BITMAP)
  	      evdwl = tb->e[itable] + fraction*tb->de[itable];
  	    else
  	      evdwl = a * tb->e[itable] + b * tb->e[itable+1] + 
  		((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) * 
  		tb->deltasq6;
  	    evdwl *= factor_lj;
  	  }
 	  
  	  if (evflag) ev_tally(i,j,nlocal,newton_pair,
  			       evdwl,0.0,fpair,delx,dely,delz);
  	}  
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(tabflag,n+1,n+1,"pair:tabflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) {
      setflag[i][j] = 0;
      tabflag[i][j] = 0;
    }

  memory->create(cutsq,   n+1,n+1,"pair:cutsq");
  memory->create(cut_lj,  n+1,n+1,"pair:cut_lj");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");
  memory->create(epsilon, n+1,n+1,"pair:epsilon");
  memory->create(sigma,   n+1,n+1,"pair:sigma");

  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");

  memory->create(offset,n+1,n+1,"pair:offset");
  memory->create(tabindex,n+1,n+1,"pair:tabindex");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::settings(int narg, char **arg)
{
  int indx;
  if (narg < 3 || narg > 4) error->one(FLERR,"Illegal pair_style command");
  
  cut_lj_global = force->numeric(FLERR,arg[0]);
  if (narg == 3) {
    cut_coul = cut_lj_global;
    indx = 1;
  } else {
    cut_coul = force->numeric(FLERR,arg[1]);
    indx = 2;
  }
  
  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
	if (setflag[i][j]) cut_lj[i][j] = cut_lj_global;
  }

  // new table settings
  
  if (strcmp(arg[indx],"lookup") == 0) tabstyle = LOOKUP;
  else if (strcmp(arg[indx],"linear") == 0) tabstyle = LINEAR;
  else if (strcmp(arg[indx],"spline") == 0) tabstyle = SPLINE;
  else if (strcmp(arg[indx],"bitmap") == 0) tabstyle = BITMAP;
  else error->one(FLERR,"Unknown table style in pair_style command");

  if(tabstyle != LINEAR) error->one(FLERR,"Table style not yet supported.  Only linear.");
  
  tablength = force->inumeric(FLERR,arg[indx+1]);

 // delete old tables, since cannot just change settings

  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

  ntables = 0;
  tables = NULL;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6) error->one(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR, arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR, arg[1],atom->ntypes,jlo,jhi);

  // Coefficients for lj/cut/coul/long
  if (strcmp(arg[2],"lj") == 0) {
    
    double epsilon_one = force->numeric(FLERR,arg[3]);
    double sigma_one = force->numeric(FLERR,arg[4]);

    double cut_lj_one = cut_lj_global;
    if (narg == 6) cut_lj_one = force->numeric(FLERR,arg[5]);

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
	epsilon[i][j] = epsilon_one;
	sigma[i][j] = sigma_one;
	cut_lj[i][j] = cut_lj_one;
	setflag[i][j] = 1;
	count++;
      }
    }
    if (count == 0) error->one(FLERR,"Incorrect args for pair coefficients");

    // Coefficients for table
  } else if (strcmp(arg[2],"tab") == 0) {
    int me;
    MPI_Comm_rank(world,&me);
    tables = (Table *) 
      memory->srealloc(tables,(ntables+1)*sizeof(Table),"pair:tables");
    Table *tb = &tables[ntables];
    null_table(tb);
    if (me == 0) read_table(tb,arg[3],arg[4]);
    bcast_table(tb);
  
   // set table cutoff

    if (narg == 6) tb->cut = force->numeric(FLERR,arg[5]);
    else if (tb->rflag) tb->cut = tb->rhi;
    else tb->cut = tb->rfile[tb->ninput-1];
  
   // error check on table parameters
   // insure cutoff is within table
   // for BITMAP tables, file values can be in non-ascending order

    if (tb->ninput <= 1) error->one(FLERR,"Invalid pair table length");
    double rlo,rhi;
    if (tb->rflag == 0) {
      rlo = tb->rfile[0];
      rhi = tb->rfile[tb->ninput-1];
    } else {
      rlo = tb->rlo;
      rhi = tb->rhi;
    }
    if (tb->cut <= rlo || tb->cut > rhi) error->one(FLERR,"Invalid pair table cutoff");
    if (rlo <= 0.0) error->one(FLERR,"Invalid pair table cutoff");

    // match = 1 if don't need to spline read-in tables
    // this is only the case if r values needed by final tables
    //   exactly match r values read from file

    tb->match = 0;
    if (tabstyle == LINEAR && tb->ninput == tablength && 
     	tb->rflag == RSQ && tb->rhi == tb->cut) tb->match = 1;
    if (tabstyle == SPLINE && tb->ninput == tablength && 
     	tb->rflag == RSQ && tb->rhi == tb->cut) tb->match = 1;
    if (tabstyle == BITMAP && tb->ninput == 1 << tablength && 
     	tb->rflag == BMP && tb->rhi == tb->cut) tb->match = 1;

    if (tb->rflag == BMP && tb->match == 0)
      error->one(FLERR,"Bitmapped table in file does not match requested table");

   // spline read-in values and compute r,e,f vectors within table

    if (tb->match == 0) spline_table(tb);  //breaks MS-EVB (parent_id array)
    compute_table(tb);

   // store ptr to table in tabindex

    int count = 0;
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo,i); j <= jhi; j++) {
      	tabindex[i][j] = ntables;
      	tabflag[i][j] = 1;
      	count++;
      }
    }
    
    if (count == 0) error->one(FLERR,"Incorrect args for pair coefficients");
    ntables++;
    
   } else error->one(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::init_style()
{
  if (!atom->q_flag)
    error->one(FLERR,"Pair style lj/cut/coul/long requires atom attribute q");

  // request regular or rRESPA neighbor lists

  int irequest;

  if (update->whichflag == 1 && strcmp(update->integrate_style,"respa") == 0) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;

    if (respa == 0) irequest = neighbor->request(this,instance_me);
    else if (respa == 1) {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    } else {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 2;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respamiddle = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->half = 0;
      neighbor->requests[irequest]->respaouter = 1;
    }

  } else irequest = neighbor->request(this,instance_me);

  cut_coulsq = cut_coul * cut_coul;

  // set rRESPA cutoffs

  if (strcmp(update->integrate_style,"respa") == 0 &&
      ((Respa *) update->integrate)->level_inner >= 0)
    cut_respa = ((Respa *) update->integrate)->cutoff;
  else cut_respa = NULL;

  // insure use of KSpace long-range solver, set g_ewald

  if (force->kspace == NULL)
    error->one(FLERR,"Pair style is incompatible with KSpace style");
  g_ewald = force->kspace->g_ewald;

  // setup force tables

  if (ncoultablebits) init_tables();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTableLJCutCoulLong::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
			       sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i],cut_lj[j][j]);
  }

  double cut = MAX(cut_lj[i][j],cut_coul+2.0*qdist);
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  cut_ljsq[j][i] = cut_ljsq[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double sig2 = sigma[i][j]*sigma[i][j];
    double sig6 = sig2*sig2*sig2;
    double rc3 = cut_lj[i][j]*cut_lj[i][j]*cut_lj[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] * 
      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9); 
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] * 
      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9); 
  } 

  tabindex[j][i] = tabindex[i][j];
  tabflag[j][i]  = tabflag[i][j];
  return cut;
} 

/* ----------------------------------------------------------------------
   setup force tables used in compute routines
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::init_tables()
{
  int masklo,maskhi;
  double r,grij,expm2,derfc,rsw;
  double qqrd2e = force->qqrd2e;

  tabinnersq = tabinner*tabinner;
  init_bitmap(tabinner,cut_coul,ncoultablebits,
	      masklo,maskhi,ncoulmask,ncoulshiftbits);
  
  int ntable = 1;
  for (int i = 0; i < ncoultablebits; i++) ntable *= 2;
  
  // linear lookup tables of length N = 2^ncoultablebits
  // stored value = value at lower edge of bin
  // d values = delta from lower edge to upper edge of bin

  if (ftable) free_tables();
  
  memory->create(rtable,ntable,"pair:rtable");
  memory->create(ftable,ntable,"pair:ftable");
  memory->create(ctable,ntable,"pair:ctable");
  memory->create(etable,ntable,"pair:etable");
  memory->create(drtable,ntable,"pair:drtable");
  memory->create(dftable,ntable,"pair:dftable");
  memory->create(dctable,ntable,"pair:dctable");
  memory->create(detable,ntable,"pair:detable");

  if (cut_respa == NULL) {
    vtable = ptable = dvtable = dptable = NULL;
  } else {
    memory->create(vtable,ntable,"pair:vtable");
    memory->create(ptable,ntable,"pair:ptable");
    memory->create(dvtable,ntable,"pair:dvtable");
    memory->create(dptable,ntable,"pair:dptable");
  }

  union_int_float_t rsq_lookup;
  union_int_float_t minrsq_lookup;
  int itablemin;
  minrsq_lookup.i = 0 << ncoulshiftbits;
  minrsq_lookup.i |= maskhi;
    
  for (int i = 0; i < ntable; i++) {
    rsq_lookup.i = i << ncoulshiftbits;
    rsq_lookup.i |= masklo;
    if (rsq_lookup.f < tabinnersq) {
      rsq_lookup.i = i << ncoulshiftbits;
      rsq_lookup.i |= maskhi;
    }
    r = sqrtf(rsq_lookup.f);
    grij = g_ewald * r;
    expm2 = exp(-grij*grij);
    derfc = erfc(grij);
    if (cut_respa == NULL) {
      rtable[i] = rsq_lookup.f;
      ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      ctable[i] = qqrd2e/r;
      etable[i] = qqrd2e/r * derfc;
    } else {
      rtable[i] = rsq_lookup.f;
      ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2 - 1.0);
      ctable[i] = 0.0;
      etable[i] = qqrd2e/r * derfc;
      ptable[i] = qqrd2e/r;
      vtable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      if (rsq_lookup.f > cut_respa[2]*cut_respa[2]) {
	if (rsq_lookup.f < cut_respa[3]*cut_respa[3]) {
	  rsw = (r - cut_respa[2])/(cut_respa[3] - cut_respa[2]); 
	  ftable[i] += qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
	  ctable[i] = qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
	} else {
	  ftable[i] = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
	  ctable[i] = qqrd2e/r;
	}
      }
    }
    minrsq_lookup.f = MIN(minrsq_lookup.f,rsq_lookup.f);
  }

  tabinnersq = minrsq_lookup.f;
  
  int ntablem1 = ntable - 1;
  
  for (int i = 0; i < ntablem1; i++) {
    drtable[i] = 1.0/(rtable[i+1] - rtable[i]);
    dftable[i] = ftable[i+1] - ftable[i];
    dctable[i] = ctable[i+1] - ctable[i];
    detable[i] = etable[i+1] - etable[i];
  }

  if (cut_respa) {
    for (int i = 0; i < ntablem1; i++) {
      dvtable[i] = vtable[i+1] - vtable[i];
      dptable[i] = ptable[i+1] - ptable[i];
    }
  }
  
  // get the delta values for the last table entries 
  // tables are connected periodically between 0 and ntablem1
    
  drtable[ntablem1] = 1.0/(rtable[0] - rtable[ntablem1]);
  dftable[ntablem1] = ftable[0] - ftable[ntablem1];
  dctable[ntablem1] = ctable[0] - ctable[ntablem1];
  detable[ntablem1] = etable[0] - etable[ntablem1];
  if (cut_respa) {
    dvtable[ntablem1] = vtable[0] - vtable[ntablem1];
    dptable[ntablem1] = ptable[0] - ptable[ntablem1];
  }

  // get the correct delta values at itablemax    
  // smallest r is in bin itablemin
  // largest r is in bin itablemax, which is itablemin-1,
  //   or ntablem1 if itablemin=0
  // deltas at itablemax only needed if corresponding rsq < cut*cut
  // if so, compute deltas between rsq and cut*cut 

  double f_tmp,c_tmp,e_tmp,p_tmp,v_tmp;
  itablemin = minrsq_lookup.i & ncoulmask;
  itablemin >>= ncoulshiftbits;  
  int itablemax = itablemin - 1; 
  if (itablemin == 0) itablemax = ntablem1;     
  rsq_lookup.i = itablemax << ncoulshiftbits;
  rsq_lookup.i |= maskhi;

  if (rsq_lookup.f < cut_coulsq) {
    rsq_lookup.f = cut_coulsq;  
    r = sqrtf(rsq_lookup.f);
    grij = g_ewald * r;
    expm2 = exp(-grij*grij);
    derfc = erfc(grij);

    if (cut_respa == NULL) {
      f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      c_tmp = qqrd2e/r;
      e_tmp = qqrd2e/r * derfc;
    } else {
      f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2 - 1.0);
      c_tmp = 0.0;
      e_tmp = qqrd2e/r * derfc;
      p_tmp = qqrd2e/r;
      v_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
      if (rsq_lookup.f > cut_respa[2]*cut_respa[2]) {
        if (rsq_lookup.f < cut_respa[3]*cut_respa[3]) {
          rsw = (r - cut_respa[2])/(cut_respa[3] - cut_respa[2]); 
          f_tmp += qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
          c_tmp = qqrd2e/r * rsw*rsw*(3.0 - 2.0*rsw);
        } else {
          f_tmp = qqrd2e/r * (derfc + EWALD_F*grij*expm2);
          c_tmp = qqrd2e/r;
        }
      }
    }

    drtable[itablemax] = 1.0/(rsq_lookup.f - rtable[itablemax]);   
    dftable[itablemax] = f_tmp - ftable[itablemax];
    dctable[itablemax] = c_tmp - ctable[itablemax];
    detable[itablemax] = e_tmp - etable[itablemax];
    if (cut_respa) {
      dvtable[itablemax] = v_tmp - vtable[itablemax];
      dptable[itablemax] = p_tmp - ptable[itablemax];
    }   
  }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::write_restart(FILE *fp)
{

}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::read_restart(FILE *fp)
{

}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::write_restart_settings(FILE *fp)
{

}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::read_restart_settings(FILE *fp)
{

}

/* ----------------------------------------------------------------------
   free memory for tables used in pair computations
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::free_tables()
{
  memory->destroy( rtable);
  memory->destroy(drtable);
  memory->destroy( ftable);
  memory->destroy(dftable);
  memory->destroy( ctable);
  memory->destroy(dctable);
  memory->destroy( etable);
  memory->destroy(detable);
  memory->destroy( vtable);
  memory->destroy(dvtable);
  memory->destroy( ptable);
  memory->destroy(dptable);

}

/* ---------------------------------------------------------------------- */

void *PairTableLJCutCoulLong::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_coul;
  return NULL;
}

/* ----------------------------------------------------------------------
   free all arrays in a table
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::free_table(Table *tb)
{
  memory->destroy(tb->rfile);
  memory->destroy(tb->efile);
  memory->destroy(tb->ffile);
  memory->destroy(tb->e2file);
  memory->destroy(tb->f2file);

  memory->destroy(tb->rsq);
  memory->destroy(tb->drsq);
  memory->destroy(tb->e);
  memory->destroy(tb->de);
  memory->destroy(tb->f);
  memory->destroy(tb->df);
  memory->destroy(tb->e2);
  memory->destroy(tb->f2);
}

/* ----------------------------------------------------------------------
   set all ptrs in a table to NULL, so can be freed safely
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::null_table(Table *tb)
{
  tb->rfile = tb->efile = tb->ffile = NULL;
  tb->e2file = tb->f2file = NULL;
  tb->rsq = tb->drsq = tb->e = tb->de = NULL;
  tb->f = tb->df = tb->e2 = tb->f2 = NULL;
}

/* ----------------------------------------------------------------------
   read a table section from a tabulated potential file
   only called by proc 0
   this function sets these values in Table: 
     ninput,rfile,efile,ffile,rflag,rlo,rhi,fpflag,fplo,fphi,ntablebits
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::read_table(Table *tb, char *file, char *keyword)
{
  char line[MAXLINE];

  // open file

  FILE *fp = fopen(file,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open file %s",file);
    error->one(FLERR,str);
  }

  // loop until section found with matching keyword

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
  memory->create(tb->rfile,tb->ninput,"pair:rfile");
  memory->create(tb->efile,tb->ninput,"pair:efile");
  memory->create(tb->ffile,tb->ninput,"pair:ffile");

  // setup bitmap parameters for table to read in

  tb->ntablebits = 0;
  int masklo,maskhi,nmask,nshiftbits;
  if (tb->rflag == BMP) {
    while (1 << tb->ntablebits < tb->ninput) tb->ntablebits++;
    if (1 << tb->ntablebits != tb->ninput)
      error->one(FLERR,"Bitmapped table is incorrect length in table file");
    init_bitmap(tb->rlo,tb->rhi,tb->ntablebits,masklo,maskhi,nmask,nshiftbits);
  }

  // read r,e,f table values from file
  // if rflag set, compute r
  // if rflag not set, use r from file

  int itmp;
  double rtmp;
  union_int_float_t rsq_lookup;

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
    } else if (tb->rflag == BMP) {
      rsq_lookup.i = i << nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->rlo*tb->rlo) {
        rsq_lookup.i = i << nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      rtmp = sqrtf(rsq_lookup.f);
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

void PairTableLJCutCoulLong::bcast_table(Table *tb)
{
  MPI_Bcast(&tb->ninput,1,MPI_INT,0,world);

  int me;
  MPI_Comm_rank(world,&me);
  if (me > 0) {
    memory->create(tb->rfile,tb->ninput,"pair:rfile");
    memory->create(tb->efile,tb->ninput,"pair:efile");
    memory->create(tb->ffile,tb->ninput,"pair:ffile");
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
   build spline representation of e,f over entire range of read-in table
   this function sets these values in Table: e2file,f2file
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::spline_table(Table *tb)
{
  memory->create(tb->e2file,tb->ninput,"pair:e2file");
  memory->create(tb->f2file,tb->ninput,"pair:f2file");

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

/* ----------------------------------------------------------------------
   extract attributes from parameter line in table section
   format of line: N value R/RSQ/BITMAP lo hi FP fplo fphi
   N is required, other params are optional
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::param_extract(Table *tb, char *line)
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

void PairTableLJCutCoulLong::compute_table(Table *tb)
{
  int tlm1 = tablength - 1;

  // inner = inner table bound
  // cut = outer table bound
  // delta = table spacing in rsq for N-1 bins

  double inner;
  if (tb->rflag) inner = tb->rlo;
  else inner = tb->rfile[0];
  tb->innersq = inner*inner;
  tb->delta = (tb->cut*tb->cut - tb->innersq) / tlm1;
  tb->invdelta = 1.0/tb->delta;

  // direct lookup tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // e,f = value at midpt of bin
  // e,f are N-1 in length since store 1 value at bin midpt
  // f is converted to f/r when stored in f[i]
  // e,f are never a match to read-in values, always computed via spline interp

  if (tabstyle == LOOKUP) {
    memory->create(tb->e,tlm1,"pair:e");
    memory->create(tb->f,tlm1,"pair:f");

    double r,rsq;
    for (int i = 0; i < tlm1; i++) {
      rsq = tb->innersq + (i+0.5)*tb->delta;
      r = sqrt(rsq);
      tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
      tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
    }
  }

  // linear tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // de,df values = delta from lower edge to upper edge of bin
  // rsq,e,f are N in length so de,df arrays can compute difference
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == LINEAR) {
    memory->create(tb->rsq,tablength,"pair:rsq");
    memory->create(tb->e,  tablength,"pair:e");
    memory->create(tb->f,  tablength,"pair:f");
    memory->create(tb->de, tlm1,     "pair:de");
    memory->create(tb->df, tlm1,     "pair:df");

    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
	tb->e[i] = tb->efile[i];
	tb->f[i] = tb->ffile[i]/r;
      } else {
	tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
	tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
    }
    
    for (int i = 0; i < tlm1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
    }
  }

  // cubic spline tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // e2,f2 = spline coefficient for each bin
  // rsq,e,f,e2,f2 are N in length so have N-1 spline bins
  // f is converted to f/r after e is splined
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == SPLINE) {
    memory->create(tb->rsq,tablength,"pair:rsq");
    memory->create(tb->e,  tablength,"pair:e");
    memory->create(tb->f,  tablength,"pair:f");
    memory->create(tb->e2, tablength,"pair:e2");
    memory->create(tb->f2, tablength,"pair:f2");

    tb->deltasq6 = tb->delta*tb->delta / 6.0;

    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
	tb->e[i] = tb->efile[i];
	tb->f[i] = tb->ffile[i]/r;
      } else {
	tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
	tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r);
      }
    }

    // ep0,epn = dE/dr at inner and at cut

    double ep0 = - tb->f[0];
    double epn = - tb->f[tlm1];
    spline(tb->rsq,tb->e,tablength,ep0,epn,tb->e2);

    // fp0,fpn = dh/dg at inner and at cut
    // h(r) = f(r)/r and g(r) = r^2
    // dh/dg = (1/r df/dr - f/r^2) / 2r
    // dh/dg in secant approx = (f(r2)/r2 - f(r1)/r1) / (g(r2) - g(r1))

    double fp0,fpn;
    double secant_factor = 0.1;
    if (tb->fpflag) fp0 = (tb->fplo/sqrt(tb->innersq) - tb->f[0]/tb->innersq) /
      (2.0 * sqrt(tb->innersq));
    else {
      double rsq1 = tb->innersq;
      double rsq2 = rsq1 + secant_factor*tb->delta;
      fp0 = (splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,sqrt(rsq2)) /
	     sqrt(rsq2) - tb->f[0] / sqrt(rsq1)) / (secant_factor*tb->delta);
    }

    if (tb->fpflag && tb->cut == tb->rfile[tb->ninput-1]) fpn =
      (tb->fphi/tb->cut - tb->f[tlm1]/(tb->cut*tb->cut)) / (2.0 * tb->cut);
    else {
      double rsq2 = tb->cut * tb->cut;
      double rsq1 = rsq2 - secant_factor*tb->delta;
      fpn = (tb->f[tlm1] / sqrt(rsq2) - 
	     splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,sqrt(rsq1)) /
	     sqrt(rsq1)) / (secant_factor*tb->delta);
    }

    for (int i = 0; i < tablength; i++) tb->f[i] /= sqrt(tb->rsq[i]);
    spline(tb->rsq,tb->f,tablength,fp0,fpn,tb->f2);
  }

  // bitmapped linear tables
  // 2^N bins from inner to cut, spaced in bitmapped manner
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == BITMAP) {
    double r;
    union_int_float_t rsq_lookup;
    int masklo,maskhi;

    // linear lookup tables of length ntable = 2^n
    // stored value = value at lower edge of bin
	
    init_bitmap(inner,tb->cut,tablength,masklo,maskhi,tb->nmask,tb->nshiftbits);
    int ntable = 1 << tablength;
    int ntablem1 = ntable - 1;

    memory->create(tb->rsq, ntable,"pair:rsq");
    memory->create(tb->e,   ntable,"pair:e");
    memory->create(tb->f,   ntable,"pair:f");
    memory->create(tb->de,  ntable,"pair:de");
    memory->create(tb->df,  ntable,"pair:df");
    memory->create(tb->drsq,ntable,"pair:drsq");
  
    union_int_float_t minrsq_lookup;
    minrsq_lookup.i = 0 << tb->nshiftbits;
    minrsq_lookup.i |= maskhi;

    for (int i = 0; i < ntable; i++) {
      rsq_lookup.i = i << tb->nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->innersq) {
        rsq_lookup.i = i << tb->nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      r = sqrtf(rsq_lookup.f);
      tb->rsq[i] = rsq_lookup.f;
      if (tb->match) {
	tb->e[i] = tb->efile[i];
	tb->f[i] = tb->ffile[i]/r;
      } else {
	tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
	tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
      minrsq_lookup.f = MIN(minrsq_lookup.f,rsq_lookup.f);
    }

    tb->innersq = minrsq_lookup.f;
    
    for (int i = 0; i < ntablem1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
      tb->drsq[i] = 1.0/(tb->rsq[i+1] - tb->rsq[i]);
    } 

    // get the delta values for the last table entries 
    // tables are connected periodically between 0 and ntablem1
    
    tb->de[ntablem1] = tb->e[0] - tb->e[ntablem1];
    tb->df[ntablem1] = tb->f[0] - tb->f[ntablem1];
    tb->drsq[ntablem1] = 1.0/(tb->rsq[0] - tb->rsq[ntablem1]);

    // get the correct delta values at itablemax    
    // smallest r is in bin itablemin
    // largest r is in bin itablemax, which is itablemin-1,
    //   or ntablem1 if itablemin=0

    // deltas at itablemax only needed if corresponding rsq < cut*cut
    // if so, compute deltas between rsq and cut*cut 
    //   if tb->match, data at cut*cut is unavailable, so we'll take
    //   deltas at itablemax-1 as a good approximation
	
    double e_tmp,f_tmp;
    int itablemin = minrsq_lookup.i & tb->nmask;
    itablemin >>= tb->nshiftbits;  
    int itablemax = itablemin - 1; 
    if (itablemin == 0) itablemax = ntablem1;     
    int itablemaxm1 = itablemax - 1; 
    if (itablemax == 0) itablemaxm1 = ntablem1;       
    rsq_lookup.i = itablemax << tb->nshiftbits;
    rsq_lookup.i |= maskhi;          
    if (rsq_lookup.f < tb->cut*tb->cut) {
      if (tb->match) {
        tb->de[itablemax] = tb->de[itablemaxm1];
        tb->df[itablemax] = tb->df[itablemaxm1];
        tb->drsq[itablemax] = tb->drsq[itablemaxm1];
      } else {
	    rsq_lookup.f = tb->cut*tb->cut;   
        r = sqrtf(rsq_lookup.f);
        e_tmp = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        f_tmp = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
        tb->de[itablemax] = e_tmp - tb->e[itablemax];
        tb->df[itablemax] = f_tmp - tb->f[itablemax];
        tb->drsq[itablemax] = 1.0/(rsq_lookup.f - tb->rsq[itablemax]);
      }
    }
  } 
}


/* ----------------------------------------------------------------------
   spline and splint routines modified from Numerical Recipes
------------------------------------------------------------------------- */

void PairTableLJCutCoulLong::spline(double *x, double *y, int n,
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

double PairTableLJCutCoulLong::splint(double *xa, double *ya, double *y2a, int n, double x)
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

/* ---------------------------------------------------------------------- */

double PairTableLJCutCoulLong::single(int i, int j, int itype, int jtype,
				      double factor_lj, double fflag_doub,
				      double dum1, double & dum2)
{
  double r2inv,r6inv,r,t,refactor;
  double fraction,table,forcelj,phicoul,philj;
  int itable;
  int tlm1 = tablength - 1;

  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double delx,dely,delz,rsq;
  double eng,fpair;

  int fflag = 0;
  if(fflag_doub > 0.0) fflag = 1;

  eng = 0.0;

  if( tabflag[itype][jtype] ) {

    delx = x[i][0] - x[j][0];
    dely = x[i][1] - x[j][1];
    delz = x[i][2] - x[j][2];
    rsq = delx*delx + dely*dely + delz*delz;

    if(rsq < cut_ljsq[itype][jtype]) {
      double fraction,value,a,b,phi;
      
      Table *tb = &tables[tabindex[itype][jtype]];
      if (rsq < tb->innersq) {
	fprintf(stdout,"\nTable %i   r = %f  cutoff = %f\n",tabindex[itype][jtype],sqrt(rsq),sqrt(tb->innersq));
	error->one(FLERR,"Pair distance < table inner cutoff");
      }
      
      if (tabstyle == LINEAR) {
	itable = static_cast<int> ((rsq-tb->innersq) * tb->invdelta);
	if (itable >= tlm1) {
	  fprintf(stdout,"\nTable %i   r = %f  itable = %i  indx. cutoff = %i\n",tabindex[itype][jtype],sqrt(rsq),itable,tlm1);
	  error->one(FLERR,"Pair distance > table outer cutoff");
	}
	fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
	value = tb->f[itable] + fraction*tb->df[itable];
	fpair = factor_lj * value;
      }
      
      phi = tb->e[itable] + fraction*tb->de[itable];
      
      eng = factor_lj * phi;
      fprintf(stdout,"r = %f  e = %f  eng = %f\n",sqrt(rsq),phi,eng);
      
      if(fflag) {
	f[i][0] += delx*fpair;
	f[i][1] += dely*fpair;
	f[i][2] += delz*fpair;
	if (newton_pair || j < nlocal) {
	  f[j][0] -= delx*fpair;
	  f[j][1] -= dely*fpair;
	  f[j][2] -= delz*fpair;
	}
      }
    }
  }
  
  return eng;
}

/* ---------------------------------------------------------------------- */

double PairTableLJCutCoulLong::single_ener_noljcoul(int i, int j, int itype, int jtype, double rsq, double factor_lj)
{
  double eng = 0.0;
  if(tabflag[itype][jtype] && rsq < cut_ljsq[itype][jtype]) {
    
    int itable;
    double fraction,value,a,b,phi;
    int tlm1 = tablength - 1;
    
    Table *tb = &tables[tabindex[itype][jtype]];
    if (rsq < tb->innersq) {
      fprintf(stdout,"\nTable %i   r = %f  cutoff = %f\n",tabindex[itype][jtype],sqrt(rsq),sqrt(tb->innersq));
      error->one(FLERR,"Pair distance < table inner cutoff");
    }
    
    if (tabstyle == LINEAR) {
      itable = static_cast<int> ((rsq-tb->innersq) * tb->invdelta);
      if (itable >= tlm1) {
	fprintf(stdout,"\nTable %i   r = %f  itable = %i  indx. cutoff = %i\n",tabindex[itype][jtype],sqrt(rsq),itable,tlm1);
	error->one(FLERR,"Pair distance > table outer cutoff");
      }
      fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
      phi = tb->e[itable] + fraction*tb->de[itable];
    }
    eng = factor_lj * phi;
  }

  return eng;
}

/* ---------------------------------------------------------------------- */

double PairTableLJCutCoulLong::single_fpair_noljcoul(int i, int j, int itype, int jtype, double rsq, double factor_lj)
{
  double fpair = 0.0;
  if(tabflag[itype][jtype] && rsq < cut_ljsq[itype][jtype]) {
    int itable;
    double fraction,value,a,b,phi;
    int tlm1 = tablength - 1;

    Table *tb = &tables[tabindex[itype][jtype]];
    if (rsq < tb->innersq) {
      fprintf(stdout,"\nTable %i   r = %f  cutoff = %f\n",tabindex[itype][jtype],sqrt(rsq),sqrt(tb->innersq));
      error->one(FLERR,"Pair distance < table inner cutoff");
    }
    
    if (tabstyle == LINEAR) {
      itable = static_cast<int> ((rsq-tb->innersq) * tb->invdelta);
      if (itable >= tlm1) {
	fprintf(stdout,"\nTable %i   r = %f  itable = %i  indx. cutoff = %i\n",tabindex[itype][jtype],sqrt(rsq),itable,tlm1);
	error->one(FLERR,"Pair distance > table outer cutoff");
      }
      fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
      value = tb->f[itable] + fraction*tb->df[itable];
    }
    fpair = factor_lj * value;
  }

  return fpair;
}
