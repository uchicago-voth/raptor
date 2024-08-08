/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris Knight
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define _CRACKER_NEIGHBOR
#include "EVB_cracker.h"

#include "memory.h"
#include "error.h"
#include "update.h"
#include "atom.h"
#include "atom_vec.h"
#include "neigh_list.h"
#include "force.h"
#include "pair.h"

#include "EVB_engine.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_chain.h"
#include "EVB_complex.h"
#include "EVB_reaction.h"
#include "EVB_list.h"

#include "comm.h"  // REMOVE ME
#include "universe.h"

#if defined (_OPENMP)
#include <omp.h>
#endif
using namespace LAMMPS_NS;

#define INT_EXCHG(A,B) { int chg_tmp = A; A=B; B=chg_tmp; }

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

void EVB_List::multi_split_omp()
{
  const int *complex_atom = evb_engine->complex_atom;
  const int nall = evb_engine->natom;
  const int cplx_id = evb_complex->id;
  
  /*******************************************************************/
  /***   Spliting of pair-list   *************************************/
  /*******************************************************************/

  int i, j, num_evb;

  int *save_evb = NULL;
  save_evb = new int [sys_pair.inum];
  memset(&(save_evb[0]), -1, sys_pair.inum*sizeof(int));

#if defined (_OPENMP)
#pragma omp parallel default (none)\
  shared(save_evb, complex_atom)	   \
  private(i, j, num_evb)
#endif
  {

#if defined (_OPENMP)
    const int tid = omp_get_thread_num();
    const int dn = sys_pair.inum / comm->nthreads + 1;
    const int istart = tid * dn;
    int ii = istart + dn;
    if(ii > sys_pair.inum) ii = sys_pair.inum;
    const int iend = ii;
#else
    const int tid = 0;
    const int istart = 0;
    const int iend = sys_pair.inum;
#endif

    for(i=istart; i<iend; i++) {
      const int atom_i = sys_pair.ilist[i];
      const int numj = sys_pair.numneigh[atom_i];
      int *jlist = sys_pair.firstneigh[atom_i];
      
      if(complex_atom[atom_i]==0) {
	num_evb = numj;
	
	for(j=0; j<num_evb; j++) {
	  const int atom_j = jlist[j] & NEIGHMASK;
	  if(complex_atom[atom_j] != cplx_id) {
	    num_evb--;
	    INT_EXCHG(jlist[j],jlist[num_evb]);
	    j--;
	  }
	}
	
	if(num_evb>0) {
	  save_evb[i] = atom_i;
	  evb_pair.numneigh[atom_i] = num_evb;
	  evb_pair.firstneigh[atom_i] = jlist;
	}
      } else if(complex_atom[atom_i] == cplx_id) {
	num_evb = numj;
	
	for(j=0; j<num_evb; j++) {
	  const int atom_j = jlist[j] & NEIGHMASK;
	  if(complex_atom[atom_j]!=0 && complex_atom[atom_j]!=cplx_id) {
	    num_evb--;
	    INT_EXCHG(jlist[j],jlist[num_evb]);
	    j--;
	  }
	}
	
	if(num_evb>0) {
	  save_evb[i] = atom_i;
	  evb_pair.numneigh[atom_i] = num_evb;
	  evb_pair.firstneigh[atom_i] = jlist;
	}
      }

    } // for(i<sys_pair.inum)
  } // openmp parallel

  // Populate evb_pair.ilist
  int n = 0;
  for(i=0; i<sys_pair.inum; i++) if(save_evb[i] > -1) evb_pair.ilist[n++] = save_evb[i];
  evb_pair.inum = n;

  delete [] save_evb;
}

/* ---------------------------------------------------------------------- */


void EVB_List::sci_split_env_omp()
{
  const int nall = evb_engine->natom;
  const int *complex_atom = evb_engine->complex_atom;
  const int *kernel_atom = evb_engine->kernel_atom;

  int i, j, num_env;

  int *save_env;
  save_env = new int [sys_pair.inum];
  memset(&(save_env[0]), -1, sys_pair.inum*sizeof(int));

  /*******************************************************************/
  /***   Spliting of pair-list   *************************************/
  /*******************************************************************/

#if defined (_OPENMP)
#pragma omp parallel default(none)\
  shared(save_env, complex_atom)\
  private(i, j, num_env)
#endif
  {

#if defined (_OPENMP)
    const int tid = omp_get_thread_num();
    const int dn = sys_pair.inum / comm->nthreads + 1;
    const int istart = tid * dn;
    int ii = istart + dn;
    if(ii > sys_pair.inum) ii = sys_pair.inum;
    const int iend = ii;
#else
    const int tid = 0;
    const int istart = 0;
    const int iend = sys_pair.inum;
#endif

    for(i=istart; i<iend; i++) {
      const int atom_i = sys_pair.ilist[i];
      const int jnum = sys_pair.numneigh[atom_i];
      int *jlist = sys_pair.firstneigh[atom_i];
      
      if(!complex_atom[atom_i]) {
	// Re-sort the list to put env atoms at last
	num_env = jnum;
	
	for(j=0; j<num_env; j++) {
	  const int atom_j = jlist[j] & NEIGHMASK;
	  
	  if(complex_atom[atom_j]) {
	    num_env--;
	    INT_EXCHG(jlist[j],jlist[num_env]);
	    j--;
	  }
	}
	
	if(num_env > 0) {
	  save_env[i] = atom_i;
	  env_pair.numneigh[atom_i] = num_env;
	  env_pair.firstneigh[atom_i] = jlist;
	}
      }
    } // for(i<sys_pair.inum)
  } // openmp parallel

  // Populate env_pair.ilist
  int n = 0;
  for(i=0; i<sys_pair.inum; i++) if(save_env[i] > -1) env_pair.ilist[n++] = save_env[i];
  env_pair.inum = n;

  delete [] save_env;

  /*******************************************************************/
  /***   Spliting of bond-list   *************************************/
  /*******************************************************************/
  n_env_bond = 0;
  
  for(i=0; i<n_sys_bond; i++) {
    if(!kernel_atom[sys_bond[i][0]] || !complex_atom[sys_bond[i][0]]) {
      memcpy(env_bond[n_env_bond],sys_bond[i],sizeof(int)*3);
      n_env_bond++;
    }
  }
  
  /*******************************************************************/
  /***   Spliting of angle-list   ************************************/
  /*******************************************************************/
  n_env_angle = 0;
  
  for(i=0; i<n_sys_angle; i++) {
    if(!kernel_atom[sys_angle[i][1]] || !complex_atom[sys_angle[i][1]]) {
      memcpy(env_angle[n_env_angle],sys_angle[i],sizeof(int)*4);
      n_env_angle++;
    }
  }
  
  /*******************************************************************/
  /***   Spliting of dihedral-list   *********************************/
  /*******************************************************************/
  n_env_dihedral = 0;
  
  for(i=0; i<n_sys_dihedral; i++) {
    if(!kernel_atom[sys_dihedral[i][1]] || !complex_atom[sys_dihedral[i][1]]) {
      memcpy(env_dihedral[n_env_dihedral],sys_dihedral[i],sizeof(int)*5);
      n_env_dihedral++;
    }
  }
  
  /*******************************************************************/
  /***   Spliting of improper-list   *********************************/
  /*******************************************************************/
  n_env_improper = 0;
  
  for(i=0; i<n_sys_improper; i++) {
    if(!kernel_atom[sys_improper[i][1]] || !complex_atom[sys_improper[i][1]]) {
      memcpy(env_improper[n_env_improper],sys_improper[i],sizeof(int)*5);
      n_env_improper++;
    }
  }
}

/* ---------------------------------------------------------------------- */

void EVB_List::single_split_omp()
{ 
  const int nall = evb_engine->natom;
  const int *complex_atom = evb_engine->complex_atom;
  const int *kernel_atom = evb_engine->kernel_atom;
  const int has_cplx_atom = evb_engine->has_complex_atom;

  int i, j, num_non_evb;

  int *save_env, *save_evb;
  save_env = new int [sys_pair.inum];
  save_evb = new int [sys_pair.inum];

  memset(&(save_env[0]), -1, sys_pair.inum*sizeof(int));
  memset(&(save_evb[0]), -1, sys_pair.inum*sizeof(int));

  /*******************************************************************/
  /***   Spliting of pair-list   *************************************/
  /*******************************************************************/

  if (has_cplx_atom) {

#if defined (_OPENMP)
#pragma omp parallel default (none)\
  shared(save_env, save_evb, complex_atom)\
  private(i, j, num_non_evb)
#endif
    {

#if defined (_OPENMP)
      const int tid = omp_get_thread_num();
      const int dn = sys_pair.inum / comm->nthreads + 1;
      const int istart = tid * dn;
      int ii = istart + dn;
      if(ii > sys_pair.inum) ii = sys_pair.inum;
      const int iend = ii;
#else
      const int tid = 0;
      const int istart = 0;
      const int iend = sys_pair.inum;
#endif

      for(i=istart; i<iend; i++) {
	const int atom_i = sys_pair.ilist[i];
	const int numj = sys_pair.numneigh[atom_i];
	int *jlist = sys_pair.firstneigh[atom_i];
	
	num_non_evb = numj;
	
	const int cplx_i = complex_atom[atom_i];
	// loop all pairs in a list
	if (cplx_i) {
	  for(j=0; j<num_non_evb; j++) {
	    const int atom_j = jlist[j] & NEIGHMASK;
	    num_non_evb--;
	    INT_EXCHG(jlist[j],jlist[num_non_evb]);
	    j--;
	  }
	} else {
	  for(j=0; j<num_non_evb; j++) {
	    const int atom_j = jlist[j] & NEIGHMASK;
	    if(complex_atom[atom_j]) {
	      num_non_evb--;
	      INT_EXCHG(jlist[j],jlist[num_non_evb]);
	      j--;
	    }
	  }
	}
	
	// place the list
	if(num_non_evb == 0) {                      // EVB pair_list
	  save_evb[i] = atom_i;
	  evb_pair.numneigh[atom_i]=numj;
	  evb_pair.firstneigh[atom_i]=jlist;
	} else if(num_non_evb == numj) {            // non-EVB pair_list
	  save_env[i] = atom_i;
	  env_pair.numneigh[atom_i]=numj;
	  env_pair.firstneigh[atom_i]=jlist;
	} else {                                    // splitted  pair_list
	  save_env[i] = atom_i;
	  env_pair.numneigh[atom_i]=num_non_evb;
	  env_pair.firstneigh[atom_i]=jlist;
	  
	  save_evb[i] = atom_i;
	  evb_pair.numneigh[atom_i]=numj-num_non_evb;
	  evb_pair.firstneigh[atom_i]=jlist+num_non_evb;
	}
	
      } // for(i<sys_pair.inum)
    } // openmp parallel
    
    // Populate evb_pair.ilist and env_pair.ilist
    int n_evb = 0;
    int n_env = 0;
    for(i=0; i<sys_pair.inum; i++) {
      if(save_evb[i] > -1) evb_pair.ilist[n_evb++] = save_evb[i];
      if(save_env[i] > -1) env_pair.ilist[n_env++] = save_env[i];
    }
    evb_pair.inum = n_evb;
    env_pair.inum = n_env;
    
  } else {
    // ** No complex atom ** //
    evb_pair.inum = 0;
    env_pair.inum = 0;
    for(i=0; i<sys_pair.inum; i++) {
      const int atom_i = sys_pair.ilist[i];
      const int numj = sys_pair.numneigh[atom_i];
      int* jlist = sys_pair.firstneigh[atom_i];
      
      // place the list, all non-evb
      env_pair.ilist[env_pair.inum++] = atom_i;
      env_pair.numneigh[atom_i] = numj;
      env_pair.firstneigh[atom_i] = jlist;
    }
  }
  
  delete [] save_env;
  delete [] save_evb;

#ifdef DLEVB_MODEL_SUPPORT
  // Search for Coulomb 1-4 pairs which cross the cplx+env boundary.
  if(evb_engine->EVB14) search_14coul();
#endif
  
  /*******************************************************************/
  /***   Spliting of bond-list   *************************************/
  /*******************************************************************/

  n_env_bond = n_evb_bond = 0;
  
  if (has_cplx_atom) {
    for(i=0; i<n_sys_bond; i++) {
      if(kernel_atom[sys_bond[i][0]] && complex_atom[sys_bond[i][0]]) {
        memcpy(evb_bond[n_evb_bond],sys_bond[i],sizeof(int)*3);
        n_evb_bond++;
      } else {
        memcpy(env_bond[n_env_bond],sys_bond[i],sizeof(int)*3);
        n_env_bond++;
      }
    }
  } else {
    n_env_bond = n_sys_bond;
    for(i=0; i<n_sys_bond; i++) memcpy(env_bond[i],sys_bond[i],sizeof(int)*3);
  }  
  
  //fprintf(screen,"sys: %d, env: %d, evb: %d\n", n_sys_bond, n_env_bond, n_evb_bond);
  //exit(0);
  
  /*******************************************************************/
  /***   Spliting of angle-list   ************************************/
  /*******************************************************************/
  
  n_env_angle = 0;
  n_evb_angle = 0;
  
  if (has_cplx_atom) {
    for(i=0; i<n_sys_angle; i++) {
      if(kernel_atom[sys_angle[i][1]] && complex_atom[sys_angle[i][1]]) {
        memcpy(evb_angle[n_evb_angle],sys_angle[i],sizeof(int)*4);
        n_evb_angle++;
      } else {
        memcpy(env_angle[n_env_angle],sys_angle[i],sizeof(int)*4);
        n_env_angle++;
      }
    }
  } else {
    n_env_angle = n_sys_angle;
    for(i=0; i<n_sys_angle; i++) memcpy(env_angle[i],sys_angle[i],sizeof(int)*4);
  }
  
  //fprintf(screen,"sys: %d, env: %d, evb: %d\n", n_sys_angle, n_env_angle, n_evb_angle);
  
  /*******************************************************************/
  /***   Spliting of dihedral-list   *********************************/
  /*******************************************************************/
  
  n_env_dihedral = 0;
  n_evb_dihedral = 0;
  
  if (has_cplx_atom) {
    for(i=0; i<n_sys_dihedral; i++) {
      if(kernel_atom[sys_dihedral[i][1]] && complex_atom[sys_dihedral[i][1]]) {
	memcpy(evb_dihedral[n_evb_dihedral],sys_dihedral[i],sizeof(int)*5);
	n_evb_dihedral++;
      } else {
	memcpy(env_dihedral[n_env_dihedral],sys_dihedral[i],sizeof(int)*5);
	n_env_dihedral++;
      }
    }
  } else {
    n_env_dihedral = n_sys_dihedral;
    for(i=0; i<n_sys_dihedral; i++) memcpy(env_dihedral[i],sys_dihedral[i],sizeof(int)*5);
  }
  
  /*******************************************************************/
  /***   Spliting of improper-list   *********************************/
  /*******************************************************************/
  
  n_env_improper = 0;
  n_evb_improper = 0;
  
  if (has_cplx_atom) {
    for(i=0; i<n_sys_improper; i++) {
      if(kernel_atom[sys_improper[i][1]] && complex_atom[sys_improper[i][1]]) {
        memcpy(evb_improper[n_evb_improper],sys_improper[i],sizeof(int)*5);
        n_evb_improper++;
      } else {
        memcpy(env_improper[n_env_improper],sys_improper[i],sizeof(int)*5);
        n_env_improper++;
      }
    }
  } else {
    n_env_improper = n_sys_improper;
    for(i=0; i<n_sys_improper; i++) memcpy(env_improper[i],sys_improper[i],sizeof(int)*5);
  }
}

/* ---------------------------------------------------------------------- */

void EVB_List::sci_split_inter_omp()
{
  const int nall = evb_engine->natom;
  const int *complex_atom = evb_engine->complex_atom;
  npair_cpl = 0;

  int i, j, num_cpl;

  int *save_cpl = NULL;
  save_cpl = new int [sys_pair.inum];
  memset(&(save_cpl[0]), -1, sys_pair.inum*sizeof(int));

#if defined (_OPENMP)
#pragma omp parallel default (none)\
  shared(save_cpl, complex_atom)\
  private(i, j, num_cpl)
#endif
  {
  
#if defined (_OPENMP)
    const int tid = omp_get_thread_num();
    const int dn = sys_pair.inum / comm->nthreads + 1;
    const int istart = tid * dn;
    int ii = istart + dn;
    if(ii > sys_pair.inum) ii = sys_pair.inum;
    const int iend = ii;
#else
    const int tid = 0;
    const int istart = 0;
    const int iend = sys_pair.inum;
#endif

    // Splitting of pair-list
    for(i=istart; i<iend; i++) {
      const int atom_i = sys_pair.ilist[i];
      const int jnum = sys_pair.numneigh[atom_i];
      int *jlist = sys_pair.firstneigh[atom_i];
      
      if(complex_atom[atom_i]==0) continue;
      
      // Re-sort the list to put env atoms at last
      num_cpl = jnum;
      
      for(j=0; j<num_cpl; j++) {
	const int atom_j = jlist[j] & NEIGHMASK;
	
	if(complex_atom[atom_j]==0 || complex_atom[atom_j]==complex_atom[atom_i]) {
	  num_cpl--;
	  INT_EXCHG(jlist[j],jlist[num_cpl]);
	  j--;
	}
      }
      
      if(num_cpl > 0) {
	save_cpl[i] = atom_i;
	cpl_pair.numneigh[atom_i] = num_cpl;
	cpl_pair.firstneigh[atom_i] = jlist;
      }
    } // for(i<sys_pair.inum)
  } // openmp parallel
    
    // Populate cpl_pair.ilist
    int n = 0;
    npair_cpl = 0;
    for(i=0; i<sys_pair.inum; i++) {
      int atomi = save_cpl[i];
      if(atomi > -1) {
	cpl_pair.ilist[n++] = atomi;
	npair_cpl+= cpl_pair.numneigh[atomi];
      }
    }
    cpl_pair.inum = n;
    
    delete [] save_cpl;
  }
