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

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

void EVB_List::multi_split()
{
#if defined (_OPENMP)
  multi_split_omp();
  return;
#endif

  int *molecule = atom->molecule;
  int *complex_atom = evb_engine->complex_atom;
  int *kernel_atom = evb_engine->kernel_atom;
  int nall = evb_engine->natom;
  int cplx_id = evb_complex->id;
  
  /*******************************************************************/
  /***   Spliting of pair-list   *************************************/
  /*******************************************************************/
  
  evb_pair.inum = 0;
    
  for(int i=0; i<sys_pair.inum; i++)
  {
      int atom_i = sys_pair.ilist[i];
      int numj = sys_pair.numneigh[atom_i];
      int* jlist = sys_pair.firstneigh[atom_i];
      
      if(complex_atom[atom_i]==0)
      {
          int num_evb = numj;
          
          for(int j=0; j<num_evb; j++)
          {
              int atom_j = jlist[j] & NEIGHMASK;
              if(complex_atom[atom_j]!=cplx_id)
              {
                  num_evb--;
                  INT_EXCHG(jlist[j],jlist[num_evb]);
                  j--;
              }
          }

          if(num_evb>0)
          {
              evb_pair.ilist[evb_pair.inum++] = atom_i;
              evb_pair.numneigh[atom_i] = num_evb;
              evb_pair.firstneigh[atom_i] = jlist;
          }
      }
      else if(complex_atom[atom_i]==cplx_id)
      {
          int num_evb = numj;
          
          for(int j=0; j<num_evb; j++)
          {
              int atom_j = jlist[j] & NEIGHMASK;
              if(complex_atom[atom_j]!=0 && complex_atom[atom_j]!=cplx_id)
              {
                  num_evb--;
                  INT_EXCHG(jlist[j],jlist[num_evb]);
                  j--;
              }
          }

          if(num_evb>0)
          {
              evb_pair.ilist[evb_pair.inum++] = atom_i;
              evb_pair.numneigh[atom_i] = num_evb;
              evb_pair.firstneigh[atom_i] = jlist;
          }
      }   
  }  
}

/* ---------------------------------------------------------------------- */

void EVB_List::sci_split_inter()
{
#if defined (_OPENMP)
  sci_split_inter_omp();
  return;
#endif

    int nall = evb_engine->natom;
    int *complex_atom = evb_engine->complex_atom;
    npair_cpl = 0;

    // Clear the pair-list
    cpl_pair.inum = 0;

    // Splitting of pair-list
    
    for(int i=0; i<sys_pair.inum; i++)
    {
        int atom_i = sys_pair.ilist[i];
        int jnum = sys_pair.numneigh[atom_i];
        int *jlist = sys_pair.firstneigh[atom_i];

        if(complex_atom[atom_i]==0) continue;
      
        // Re-sort the list to put env atoms at last
        int num_cpl = jnum;
             
        for(int j=0; j<num_cpl; j++)
        {
            int atom_j = jlist[j] & NEIGHMASK;
            
            if(complex_atom[atom_j]==0 || complex_atom[atom_j]==complex_atom[atom_i])
            {
                num_cpl--;
                INT_EXCHG(jlist[j],jlist[num_cpl]);
                j--;
            }
        }
            
        if(num_cpl > 0)
        {
            cpl_pair.ilist[cpl_pair.inum++] = atom_i;
            cpl_pair.numneigh[atom_i] = num_cpl;
            cpl_pair.firstneigh[atom_i] = jlist;
	    npair_cpl += num_cpl;
        }
    }
}

/* ---------------------------------------------------------------------- */

void EVB_List::sci_split_env()
{

#if defined (_OPENMP)
  sci_split_env_omp();
  return;
#endif

    int nall = evb_engine->natom;
    int *complex_atom = evb_engine->complex_atom;
    int *kernel_atom = evb_engine->kernel_atom;

    /*******************************************************************/
    /***   Spliting of pair-list   *************************************/
    /*******************************************************************/
    env_pair.inum = 0;
     
    for(int i=0; i<sys_pair.inum; i++)
    {
        int atom_i = sys_pair.ilist[i];
        int jnum = sys_pair.numneigh[atom_i];
        int *jlist = sys_pair.firstneigh[atom_i];

        if(!complex_atom[atom_i])
        {
            // Re-sort the list to put env atoms at last
            int num_env = jnum;
             
            for(int j=0; j<num_env; j++)
            {
                int atom_j = jlist[j] & NEIGHMASK;
            
                if(complex_atom[atom_j])
                {
                    num_env--;
                    INT_EXCHG(jlist[j],jlist[num_env]);
                    j--;
                }
            }
            
            if(num_env > 0)
            {
                env_pair.ilist[env_pair.inum++] = atom_i;
                env_pair.numneigh[atom_i] = num_env;
                env_pair.firstneigh[atom_i] = jlist;
            }
        }
    }

    /*******************************************************************/
    /***   Spliting of bond-list   *************************************/
    /*******************************************************************/
    n_env_bond = 0;
  
    for(int i=0; i<n_sys_bond; i++)
    {
        if(!kernel_atom[sys_bond[i][0]] || !complex_atom[sys_bond[i][0]])
        {
            memcpy(env_bond[n_env_bond],sys_bond[i],sizeof(int)*3);
            n_env_bond++;
        }
    }
  
    /*******************************************************************/
    /***   Spliting of angle-list   ************************************/
    /*******************************************************************/
    n_env_angle = 0;
  
    for(int i=0; i<n_sys_angle; i++)
    {
        if(!kernel_atom[sys_angle[i][1]] || !complex_atom[sys_angle[i][1]])
        {
            memcpy(env_angle[n_env_angle],sys_angle[i],sizeof(int)*4);
            n_env_angle++;
        }
    }
  
    /*******************************************************************/
    /***   Spliting of dihedral-list   *********************************/
    /*******************************************************************/
    n_env_dihedral = 0;
    
    for(int i=0; i<n_sys_dihedral; i++)
    {
        if(!kernel_atom[sys_dihedral[i][1]] || !complex_atom[sys_dihedral[i][1]])
        {
            memcpy(env_dihedral[n_env_dihedral],sys_dihedral[i],sizeof(int)*5);
            n_env_dihedral++;
        }
    }
  
    /*******************************************************************/
    /***   Spliting of improper-list   *********************************/
    /*******************************************************************/
    n_env_improper = 0;
    
    for(int i=0; i<n_sys_improper; i++)
    {
        if(!kernel_atom[sys_improper[i][1]] || !complex_atom[sys_improper[i][1]])
        {
            memcpy(env_improper[n_env_improper],sys_improper[i],sizeof(int)*5);
            n_env_improper++;
        }
    }
}

void EVB_List::multi_combine()
{
  // Copy ENV-list to SYS-list

  n_sys_bond = n_sys_angle = n_sys_dihedral = n_sys_improper = 0;
  for(int i=0; i<n_env_bond; i++) memcpy(sys_bond[n_sys_bond++], env_bond[i], sizeof(int)*3);
  for(int i=0; i<n_env_angle; i++) memcpy(sys_angle[n_sys_angle++], env_angle[i], sizeof(int)*4);
  for(int i=0; i<n_env_dihedral; i++) memcpy(sys_dihedral[n_sys_dihedral++], env_dihedral[i], sizeof(int)*5);
  for(int i=0; i<n_env_improper; i++) memcpy(sys_improper[n_sys_improper++], env_improper[i], sizeof(int)*5);

  // Add EVB-list  
  
  for(int i=0; i<evb_engine->ncomplex; i++)
  {
    evb_engine->all_complex[i]->update_bond_list();
    
    // check memory 
    /*
 
    if(force->bond)
    {
      int nbonds = n_sys_bond+n_evb_bond;
		 	 
      if(nbonds>=neighbor->maxbond) 
      {
        neighbor->maxbond = nbonds+100;
        memory->grow(sys_bond,neighbor->maxbond,3,"EVB_List:sys_bond");
      }
    }
       
    if(force->angle)
    {
      int nangles = n_sys_angle+n_evb_angle;
         
      if(nangles>=neighbor->maxangle) 
      {
        neighbor->maxangle = nangles+100;
        memory->grow(sys_angle,neighbor->maxangle,4,"EVB_List:sys_angle");
      }
    }
         
    if(force->dihedral)
    {
      int ndihedrals = n_sys_dihedral+n_evb_dihedral;
	           
      if(ndihedrals>=neighbor->maxdihedral) 
      {
        neighbor->maxdihedral = ndihedrals+100;
        memory->grow(sys_dihedral,neighbor->maxdihedral,5,"EVB_List:sys_dihedral");
      }
    }
	   
    if(force->improper)
    {
      int nimpropers = n_sys_improper+n_evb_improper;
         
      if(nimpropers>=neighbor->maximproper) 
      {
        neighbor->maximproper = nimpropers+100;
        memory->grow(sys_improper,neighbor->maximproper,5,"EVB_List:sys_improper");
      }
    }
    */
    // add evb bonds

    for(int i=0; i<n_evb_bond; i++) memcpy(sys_bond[n_sys_bond++], evb_bond[i], sizeof(int)*3);
    for(int i=0; i<n_evb_angle; i++) memcpy(sys_angle[n_sys_angle++], evb_angle[i], sizeof(int)*4);
    for(int i=0; i<n_evb_dihedral; i++) memcpy(sys_dihedral[n_sys_dihedral++], evb_dihedral[i], sizeof(int)*5);
    for(int i=0; i<n_evb_improper; i++) memcpy(sys_improper[n_sys_improper++], evb_improper[i], sizeof(int)*5);
  } 
}
