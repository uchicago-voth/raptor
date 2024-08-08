/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define _CRACKER_NEIGHBOR
#include "EVB_cracker.h"
#undef _CRACKER_NEIGHBOR

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

using namespace LAMMPS_NS;

#define INT_EXCHG(A,B) { int chg_tmp = A; A=B; B=chg_tmp; }

#define CHANGE_LIST(t) {save_pairlist->inum=t##_pair.inum;           \
    save_pairlist->ilist=t##_pair.ilist;                               \
    save_pairlist->numneigh=t##_pair.numneigh;                         \
    save_pairlist->firstneigh=t##_pair.firstneigh;                     \
    neighbor->nbondlist=n_##t##_bond;                                \
    neighbor->bondlist=t##_bond;                                   \
    neighbor->nanglelist=n_##t##_angle;                              \
    neighbor->anglelist=t##_angle;                                 \
    neighbor->ndihedrallist=n_##t##_dihedral;                        \
    neighbor->dihedrallist=t##_dihedral;                           \
    neighbor->nimproperlist=n_##t##_improper;                        \
    neighbor->improperlist=t##_improper;}  
        
#ifdef DLEVB_MODEL_SUPPORT
#define CHANGE_PAIRLIST(t) {save_pairlist->inum=t##_pair.inum;           \
    save_pairlist->ilist=t##_pair.ilist;                               \
    save_pairlist->numneigh=t##_pair.numneigh;                         \
    save_pairlist->firstneigh=t##_pair.firstneigh;}  
#endif

 inline int sbmask(int j) {
   return j >> SBBITS & 3;
 }
                                                                  
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_List::EVB_List(LAMMPS *lmp, EVB_Engine *engine) : Pointers(lmp), EVB_Pointers(engine)
{
    npair_cpl = max_inum = evb_pair.inum = evb_pair.inum = 0;
    env_pair.ilist = env_pair.numneigh = NULL;
    evb_pair.ilist = evb_pair.numneigh = NULL;
    cpl_pair.ilist = cpl_pair.numneigh = NULL;
    env_pair.firstneigh = evb_pair.firstneigh = cpl_pair.firstneigh = NULL;
    
    n_env_bond = 0;
    n_evb_bond = 0;
    max_bond = 1000;
    memory->create(env_bond,max_bond,3,"EVB_List::env_bond");
    memory->create(evb_bond,max_bond,3,"EVB_List::evb_bond");
    
    n_env_angle = 0;
    n_evb_angle = 0;
    max_angle = 1000;
    memory->create(env_angle,max_angle,4,"EVB_List::env_angle");
    memory->create(evb_angle,max_angle,4,"EVB_List::evb_angle");
    
    n_env_dihedral = 0;
    n_evb_dihedral = 0;
    max_dihedral = 1000;
    memory->create(env_dihedral,max_dihedral,5,"EVB_List::env_dihedral");
    memory->create(evb_dihedral,max_dihedral,5,"EVB_List::evb_dihedral");
    
    n_env_improper = 0;
    n_evb_improper = 0;
    max_improper = 1000;
    memory->create(env_improper,max_improper,5,"EVB_List::env_improper");
    memory->create(evb_improper,max_improper,5,"EVB_List::evb_improper");

    indicator = -1;
}

/* ---------------------------------------------------------------------- */

EVB_List::~EVB_List()
{
    memory->sfree(evb_pair.ilist);
    memory->sfree(evb_pair.numneigh);
    memory->sfree(evb_pair.firstneigh);

    memory->sfree(env_pair.ilist);
    memory->sfree(env_pair.numneigh);
    memory->sfree(env_pair.firstneigh);

    memory->sfree(cpl_pair.ilist);
    memory->sfree(cpl_pair.numneigh);
    memory->sfree(cpl_pair.firstneigh);
   
    memory->destroy(evb_bond);
    memory->destroy(env_bond);
  
    memory->destroy(evb_angle);
    memory->destroy(env_angle);

    memory->destroy(evb_dihedral);
    memory->destroy(env_dihedral);
  
    memory->destroy(evb_improper);
    memory->destroy(env_improper);

#ifdef DLEVB_MODEL_SUPPORT
    if(evb_engine->EVB14) {
      delete [] evb_lj_pair.ilist;
      delete [] evb_lj_pair.numneigh;
      memory->destroy(evb_lj_pair.firstneigh);
    }
#endif

}

/* ---------------------------------------------------------------------- */

void EVB_List::setup()
{
    // Pair list
    
    save_pairlist = evb_engine->get_pair_list();
    sys_pair.inum = save_pairlist->inum;
    sys_pair.ilist = save_pairlist->ilist;
    sys_pair.numneigh = save_pairlist->numneigh;
    sys_pair.firstneigh = save_pairlist->firstneigh;
    
    if(sys_pair.inum > max_inum)
    {
        max_inum = sys_pair.inum;
        int size = sizeof(int)*max_inum;
        
        evb_pair.ilist = (int*)memory->srealloc(evb_pair.ilist, size,
                                                "EVB_List:evb_pair.ilist");    
        evb_pair.numneigh = (int*)memory->srealloc(evb_pair.numneigh, size,
                                                   "EVB_List:evb_pair.numneigh");
        evb_pair.firstneigh = (int**)memory->srealloc(evb_pair.firstneigh, sizeof(int*)*max_inum,
                                                      "EVB_List:evb_pair.firstneigh");

#ifdef DLEVB_MODEL_SUPPORT
	if(evb_engine->EVB14) {
	  int nall = evb_engine->natom;
	  evb_lj_pair.ilist = new int[nall];
	  evb_lj_pair.numneigh = new int[nall];
	  memory->create(evb_lj_pair.firstneigh,nall,nall,"EVB_List:evb_lj_pair.firstneigh");
	}
#endif
    
        env_pair.ilist = (int*)memory->srealloc(env_pair.ilist, size,
                                                "EVB_List:env_pair.ilist");    
        env_pair.numneigh = (int*)memory->srealloc(env_pair.numneigh, size,
                                                   "EVB_List:env_pair.numneigh");
        env_pair.firstneigh = (int**)memory->srealloc(env_pair.firstneigh, sizeof(int*)*max_inum,
                                                      "EVB_List:env_pair.firstneigh");
        if(evb_engine->ncomplex>1)
        {   
            cpl_pair.ilist = (int*)memory->srealloc(cpl_pair.ilist, size,
                                                    "EVB_List:cpl_pair.ilist");    
            cpl_pair.numneigh = (int*)memory->srealloc(cpl_pair.numneigh, size,
                                                       "EVB_List:cpl_pair.numneigh");
            cpl_pair.firstneigh = (int**)memory->srealloc(cpl_pair.firstneigh, sizeof(int*)*max_inum,
                                                          "EVB_List:cpl_pair.firstneigh");
        }    
    }

    // Bond list
    n_sys_bond = neighbor->nbondlist;
    sys_bond = neighbor->bondlist;

    if(n_sys_bond>max_bond)
    {
        max_bond = n_sys_bond;
        memory->grow(env_bond, max_bond, 3,"EVB_List:env_bond");
        memory->grow(evb_bond, max_bond, 3,"EVB_List:evb_bond");
    }

    // Angle list
    n_sys_angle = neighbor->nanglelist;
    sys_angle = neighbor->anglelist;
  
    if(n_sys_angle>max_angle)
    {
        max_angle = n_sys_angle;
        memory->grow(env_angle, max_angle, 4,"EVB_List:env_angle");
        memory->grow(evb_angle, max_angle, 4,"EVB_List:evb_angle");
    }

    // Dihedral list
    n_sys_dihedral = neighbor->ndihedrallist;
    sys_dihedral = neighbor->dihedrallist;
  
    if(n_sys_dihedral>max_dihedral)
    {
        max_dihedral = n_sys_dihedral;
        memory->grow(env_dihedral, max_dihedral, 5,"EVB_List:env_dihedral");
        memory->grow(evb_dihedral, max_dihedral, 5,"EVB_List:evb_dihedral");
    }

    // Improper list
    n_sys_improper = neighbor->nimproperlist;
    sys_improper = neighbor->improperlist;
  
    if(n_sys_improper>max_improper)
    {
        max_improper = n_sys_improper;
        memory->grow(env_improper, max_improper, 5,"EVB_List:env_improper");
        memory->grow(evb_improper, max_improper, 5,"EVB_List:evb_improper");
    }
}
  
/* ---------------------------------------------------------------------- */

void EVB_List::single_split()
{
#if defined (_OPENMP)
  single_split_omp();
  return;
#endif

  int *molecule = atom->molecule;
  int *complex_atom = evb_engine->complex_atom;
  int *kernel_atom = evb_engine->kernel_atom;
  int nall = evb_engine->natom;

  int has_cplx_atom = evb_engine->has_complex_atom;
  
  /*******************************************************************/
  /***   Spliting of pair-list   *************************************/
  /*******************************************************************/
  
  evb_pair.inum = env_pair.inum = 0;
 
  if (has_cplx_atom) {
    for(int i=0; i<sys_pair.inum; i++)
    {
        int atom_i = sys_pair.ilist[i];
        int numj = sys_pair.numneigh[atom_i];
        int* jlist = sys_pair.firstneigh[atom_i];
      
        int num_non_evb = numj;
    
        const int cplx_i = complex_atom[atom_i];
        // loop all pairs in a list
        if (cplx_i) {
          for(int j=0; j<num_non_evb; j++)
          {
            int atom_j = jlist[j] & NEIGHMASK;
	    num_non_evb--;
	    INT_EXCHG(jlist[j],jlist[num_non_evb]);
	    j--;
          }
        } else {
          for(int j=0; j<num_non_evb; j++)
          {
            int atom_j = jlist[j] & NEIGHMASK;
  	    if(complex_atom[atom_j])
	    {
	      num_non_evb--;
	      INT_EXCHG(jlist[j],jlist[num_non_evb]);
	      j--;
	    }
          }
        }
        
        // place the list
        if(num_non_evb == 0)  // EVB pair_list
        {
          evb_pair.ilist[evb_pair.inum++] = atom_i;
          evb_pair.numneigh[atom_i]=numj;
          evb_pair.firstneigh[atom_i]=jlist;
        }
        else if(num_non_evb == numj)  // non-EVB pair_list
        {
          env_pair.ilist[env_pair.inum++] = atom_i;
          env_pair.numneigh[atom_i]=numj;
          env_pair.firstneigh[atom_i]=jlist;
        }
        else  // splitted  pair_list
        {
          env_pair.ilist[env_pair.inum++] = atom_i;
          env_pair.numneigh[atom_i]=num_non_evb;
          env_pair.firstneigh[atom_i]=jlist;
      
          evb_pair.ilist[evb_pair.inum++] = atom_i;
          evb_pair.numneigh[atom_i]=numj-num_non_evb;
          evb_pair.firstneigh[atom_i]=jlist+num_non_evb;
        }

      // end of loop all lists
    }
  } else {
    // ** No complex atom ** //
    for(int i=0; i<sys_pair.inum; i++)
    {
      int atom_i = sys_pair.ilist[i];
      int numj = sys_pair.numneigh[atom_i];
      int* jlist = sys_pair.firstneigh[atom_i];
      // place the list, all non-evb
      env_pair.ilist[env_pair.inum++] = atom_i;
      env_pair.numneigh[atom_i] = numj;
      env_pair.firstneigh[atom_i] = jlist;
    }
  }

#ifdef DLEVB_MODEL_SUPPORT
  // Search for Coulomb 1-4 pairs which cross the cplx+env boundary.
  if(evb_engine->EVB14) search_14coul();
#endif
  
  /*******************************************************************/
  /***   Spliting of bond-list   *************************************/
  /*******************************************************************/

  n_env_bond = n_evb_bond = 0;
  
  if (has_cplx_atom) {
    for(int i=0; i<n_sys_bond; i++)
    {
      if(kernel_atom[sys_bond[i][0]] && complex_atom[sys_bond[i][0]])
      {
        memcpy(evb_bond[n_evb_bond],sys_bond[i],sizeof(int)*3);
        n_evb_bond++;
      }
      else
      {
        memcpy(env_bond[n_env_bond],sys_bond[i],sizeof(int)*3);
        n_env_bond++;
      }
    }
  } else {
    n_env_bond = n_sys_bond;
    for(int i=0; i<n_sys_bond; i++)
      memcpy(env_bond[i],sys_bond[i],sizeof(int)*3);
  }  

  //fprintf(screen,"sys: %d, env: %d, evb: %d\n", n_sys_bond, n_env_bond, n_evb_bond);
  //exit(0);
  
  /*******************************************************************/
  /***   Spliting of angle-list   ************************************/
  /*******************************************************************/
  
  n_env_angle = 0;
  n_evb_angle = 0;
  
  if (has_cplx_atom) {
    for(int i=0; i<n_sys_angle; i++)
    {
      if(kernel_atom[sys_angle[i][1]] && complex_atom[sys_angle[i][1]])
      {
        memcpy(evb_angle[n_evb_angle],sys_angle[i],sizeof(int)*4);
        n_evb_angle++;
      }
      else
      {
        memcpy(env_angle[n_env_angle],sys_angle[i],sizeof(int)*4);
        n_env_angle++;
      }
    }
  } else {
    n_env_angle = n_sys_angle;
    for(int i=0; i<n_sys_angle; i++)
      memcpy(env_angle[i],sys_angle[i],sizeof(int)*4);
  }
  
  //fprintf(screen,"sys: %d, env: %d, evb: %d\n", n_sys_angle, n_env_angle, n_evb_angle);

  /*******************************************************************/
  /***   Spliting of dihedral-list   *********************************/
  /*******************************************************************/
  
  n_env_dihedral = 0;
  n_evb_dihedral = 0;
  
  if (has_cplx_atom) {
    for(int i=0; i<n_sys_dihedral; i++)
    {
      if(kernel_atom[sys_dihedral[i][1]] && complex_atom[sys_dihedral[i][1]])
      {
          memcpy(evb_dihedral[n_evb_dihedral],sys_dihedral[i],sizeof(int)*5);
          n_evb_dihedral++;
      }
      else
      {
          memcpy(env_dihedral[n_env_dihedral],sys_dihedral[i],sizeof(int)*5);
          n_env_dihedral++;
      }
    }
  } else {
    n_env_dihedral = n_sys_dihedral;
    for(int i=0; i<n_sys_dihedral; i++)
      memcpy(env_dihedral[i],sys_dihedral[i],sizeof(int)*5);
  }
  
  /*******************************************************************/
  /***   Spliting of improper-list   *********************************/
  /*******************************************************************/
  
  n_env_improper = 0;
  n_evb_improper = 0;
  
  if (has_cplx_atom) {
    for(int i=0; i<n_sys_improper; i++)
    {
      if(kernel_atom[sys_improper[i][1]] && complex_atom[sys_improper[i][1]])
      {
        memcpy(evb_improper[n_evb_improper],sys_improper[i],sizeof(int)*5);
        n_evb_improper++;
      }
      else
      {
        memcpy(env_improper[n_env_improper],sys_improper[i],sizeof(int)*5);
        n_env_improper++;
      }
    }
  } else {
    n_env_improper = n_sys_improper;
    for(int i=0; i<n_sys_improper; i++)
      memcpy(env_improper[i],sys_improper[i],sizeof(int)*5);
  }
}

/* ---------------------------------------------------------------------- */

void EVB_List::single_combine()
{
    
  /***************************************************************/
  /********** Check memory ***************************************/
  /***************************************************************/
  /* YP: temporarily remove this part because I think lammps doesn't need to do this anymore 

  if(force->bond)
  {
      int nbonds = n_env_bond+n_evb_bond;
	 
      if(nbonds>=neighbor->maxbond) 
      {
          neighbor->maxbond = nbonds+100;
          memory->grow(sys_bond,neighbor->maxbond,3,"EVB_List:sys_bond");
      }
  }
  
  if(force->angle)
  {
      int nangles = n_env_angle+n_evb_angle;
      
      if(nangles>=neighbor->maxangle) 
      {
          neighbor->maxangle = nangles+100;
          memory->grow(sys_angle,neighbor->maxangle,4,"EVB_List:sys_angle");
      }
  }
  
  if(force->dihedral)
  {
      int ndihedrals = n_env_dihedral+n_evb_dihedral;
      
      if(ndihedrals>=neighbor->maxdihedral) 
      {
          neighbor->maxdihedral = ndihedrals+100;
          memory->grow(sys_dihedral,neighbor->maxdihedral,5,"EVB_List:sys_dihedral");
      }
  }
  
  if(force->improper)
  {
      int nimpropers = n_env_improper+n_evb_improper;
      
      if(nimpropers>=neighbor->maximproper) 
      {
          neighbor->maximproper = nimpropers+100;
          memory->grow(sys_improper,neighbor->maximproper,5,"EVB_List:sys_improper");
      }
  }
*/  
  /***************************************************************/
  /********** Bond  **********************************************/
  /***************************************************************/
  n_sys_bond = 0;
  
  for(int i=0; i<n_env_bond; i++)
  {
      memcpy(sys_bond[n_sys_bond],env_bond[i],sizeof(int)*3);
      n_sys_bond++;	
  }
  
  for(int i=0; i<n_evb_bond; i++)
  {
      memcpy(sys_bond[n_sys_bond],evb_bond[i],sizeof(int)*3);
      n_sys_bond++;
  }
  
  /***************************************************************/
  /********** Angle **********************************************/
  /***************************************************************/
  n_sys_angle = 0;
  
  for(int i=0; i<n_env_angle; i++)
  {
      memcpy(sys_angle[n_sys_angle],env_angle[i],sizeof(int)*4);
      n_sys_angle++;
  }
  
  for(int i=0; i<n_evb_angle; i++)
  {
      memcpy(sys_angle[n_sys_angle],evb_angle[i],sizeof(int)*4);
      n_sys_angle++;
  }
  
  /***************************************************************/
  /********** Dihedral *******************************************/
  /***************************************************************/
  n_sys_dihedral = 0;
  
  for(int i=0; i<n_env_dihedral; i++)
  {
      memcpy(sys_dihedral[n_sys_dihedral],env_dihedral[i],sizeof(int)*5);
      n_sys_dihedral++;
  }
  
  for(int i=0; i<n_evb_dihedral; i++)
  {
      memcpy(sys_dihedral[n_sys_dihedral],evb_dihedral[i],sizeof(int)*5);
      n_sys_dihedral++;
  }
  
  /***************************************************************/
  /********** Improper *******************************************/
  /***************************************************************/
  n_sys_improper = 0;
  
  for(int i=0; i<n_env_improper; i++)
  {
      memcpy(sys_improper[n_sys_improper],env_improper[i],sizeof(int)*5);
      n_sys_improper++;
  }
  
  for(int i=0; i<n_evb_improper; i++)
  {
      memcpy(sys_improper[n_sys_improper],evb_improper[i],sizeof(int)*5);
      n_sys_improper++;
  }

  // update atom information
  //atom->nbonds = n_sys_bond;
  //atom->nangles = n_sys_angle;
  //atom->ndihedrals = n_sys_dihedral;
  //atom->nimpropers = n_sys_improper;
}

/* ---------------------------------------------------------------------- */

void EVB_List::change_list(int list_index)
{
    indicator = list_index;

    switch(list_index) 
      {
      case SYS_LIST: CHANGE_LIST(sys); break;
      case ENV_LIST: CHANGE_LIST(env); break;
      case EVB_LIST: CHANGE_LIST(evb); break;
	
      case CPL_LIST: 
	save_pairlist->inum=cpl_pair.inum;           
	save_pairlist->ilist=cpl_pair.ilist;                               
	save_pairlist->numneigh=cpl_pair.numneigh;                         
	save_pairlist->firstneigh=cpl_pair.firstneigh;  
	break;
      }
}

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

#ifdef DLEVB_MODEL_SUPPORT

void EVB_List::change_pairlist(int list_index)
{
  switch(list_index)
    {
    case EVB_LIST:    CHANGE_PAIRLIST(evb); break;
    case EVB_LJ_LIST: CHANGE_PAIRLIST(evb_lj); break;
    }
}

/* ---------------------------------------------------------------------- */

void EVB_List::search_14coul()
{
  // fprintf(stdout,"Search for Coulomb 1-4 pairs\n");
    
  int i,j,jj,k,inum,jnum,jnall;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int nall = evb_engine->natom;
  int* is_cplx_atom = evb_engine->complex_atom;
  double *special_coul = force->special_coul;
  double factor_coul;
    
  int nPairs = 0;
  int nPairsMax = 100;
  int PairList14[nPairsMax][2];

//   fprintf(stdout,"\nList of Complex atoms\n");
  int n=0;
  for(int ii=0; ii<atom->nlocal; ii++) {
    if(is_cplx_atom[ii]) {
      n++;
//       fprintf(stdout,"%i: %i\n",n,atom->tag[ii]);
    }
  }
//   fprintf(stdout,"# of Complex atom = %i\n",n);
  
  inum       = evb_pair.inum;
  ilist      = evb_pair.ilist;
  numneigh   = evb_pair.numneigh;
  firstneigh = evb_pair.firstneigh;
  k = 0;
  for (int ii=0; ii<inum; ii++) {
    i     = ilist[ii];
    jlist = firstneigh[i];
    jnum  = numneigh[i];
    
//     fprintf(stdout,"%i: %i  numneigh = %i\n",ii,atom->tag[i],jnum);
    jj=0;
    while ( jj<jnum ) {
      j = jlist[jj];
      int mask_j = sbmask(j);
      factor_coul = special_coul[mask_j];
      j &= NEIGHMASK;       

//       if(ii==2) {
// 	fprintf(stdout,"jj = %i  jlist = %i  j = %i  nall = %i  jnall = %i  cplx = %i  %i\n",jj,jlist[jj],atom->tag[j],nall,jnall,is_cplx_atom[i],is_cplx_atom[j]);
//       }
      
      if( mask_j == 3 ) {
	if( (is_cplx_atom[i] && !is_cplx_atom[j]) || (is_cplx_atom[j] && !is_cplx_atom[i]) ) {
	  if(nPairs>nPairsMax) error->all(FLERR,"EVB_list::single_split()  nPairs>nPairsMax\n");
	  PairList14[nPairs][0] = i;
	  PairList14[nPairs][1] = jlist[jj];
	  nPairs++;
	  for(int kk=jj; kk<jnum-1; kk++) jlist[kk] = jlist[kk+1];
	  jnum--;
	  jj--;
	}
      }
      jj++;
    }
    evb_pair.numneigh[i]   = jnum;
    evb_pair.firstneigh[i] = jlist;
  }
  
  //Build evb_lj_pair structure
  inum = 0;
  evb_lj_pair.ilist[0] = PairList14[0][0];
  jnum = 0;
  
//    fprintf(stdout,"\nPairList14: nPairs = %i\n",nPairs);
  for(int ii=0; ii<nPairs; ii++) {
    i = PairList14[ii][0];
    j = PairList14[ii][1];
//     fprintf(stdout,"%i: %i   %i\n",ii,atom->tag[i],atom->tag[j%nall]);
    if(i != evb_lj_pair.ilist[inum]) {
      jnum = 0;
      inum++;
    }
    jnum++;
    
    if(i>max_inum) error->all(FLERR,"EVB_List:single_split()  i>max_inum\n");
    if(jnum>max_inum) error->all(FLERR,"EVB_List:single_split()  jnum>max_inum\n");
    
    evb_lj_pair.ilist[inum] = i;
    evb_lj_pair.numneigh[i] = jnum;
    evb_lj_pair.firstneigh[i][jnum-1] = j;
  }
  if(inum!=0) inum++;
  evb_lj_pair.inum = inum;    
  if(evb_lj_pair.inum>max_inum) error->all(FLERR,"EVB_List:single_split()  inum>max_inum\n");
  
  // fprintf(stdout,"\n\n# of unique pair lists = %i\n",evb_lj_pair.inum);
  // for(int ii=0; ii<evb_lj_pair.inum; ii++) {
  //   int ilist = evb_lj_pair.ilist[ii];
  //   int numneigh = evb_lj_pair.numneigh[ilist];
  //   fprintf(stdout,"%i  Pair List: %i  numneigh = %i: ",ii+1,ilist,numneigh);
  //   for(int jj=0; jj<numneigh; jj++) fprintf(stdout,"  %i",evb_lj_pair.firstneigh[ilist][jj]);
  //   fprintf(stdout,"\n");
  // }
  // fprintf(stdout,"\n");
}

/* ---------------------------------------------------------------------- */
#endif
