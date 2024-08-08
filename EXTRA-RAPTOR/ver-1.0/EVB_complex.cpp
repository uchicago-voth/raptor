/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "memory.h"
#include "error.h"
#include "update.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "domain.h"

#include "EVB_engine.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_chain.h"
#include "EVB_list.h"
#include "EVB_complex.h"
#include "EVB_reaction.h"
#include "EVB_matrix_sci.h"
#include "EVB_cec.h"
#include "EVB_cec_v2.h"
#include "EVB_offdiag.h"

#include "mp_verlet.h"

// ** AWGL ** //
#if defined (_OPENMP)
#include <omp.h>
#endif

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_Complex::EVB_Complex(LAMMPS *lmp, EVB_Engine *engine) : Pointers(lmp), EVB_Pointers(engine)
{
  nstate = natom = rc_start = 0;
  cplx_list = NULL;

  status =  NULL;
  status_nstate = status_natom = 0;
  
  max_offdiag = max_extra = 0;
  extra_i = extra_j =NULL;
  nexch_off = nexch_extra = NULL;
  iexch_off = iexch_extra = NULL;
  qexch_off = qexch_extra = NULL;

  cec = new EVB_CEC(lmp,engine,this);
  cec_v2 = new EVB_CEC_V2(lmp,engine,this);
}

/* ---------------------------------------------------------------------- */

EVB_Complex::~EVB_Complex()
{
  memory->sfree(cplx_list);
  
  memory->sfree(extra_i);
  memory->sfree(extra_j);
  memory->sfree(nexch_off);
  memory->sfree(nexch_extra);
  memory->destroy(iexch_off);
  memory->destroy(iexch_extra);
  memory->destroy(qexch_off);
  memory->destroy(qexch_extra);

  for(int i=0; i<status_nstate; i++)
  {
    memory->sfree(status[i].type);
    memory->sfree(status[i].mol_type);
    memory->sfree(status[i].mol_index);
    memory->sfree(status[i].q);
    memory->sfree(status[i].molecule);
    memory->sfree(status[i].num_bond);
    memory->sfree(status[i].num_angle);
    memory->sfree(status[i].num_dihedral);
    memory->sfree(status[i].num_improper);
    
    memory->destroy(status[i].bond_type);
    memory->destroy(status[i].bond_atom);
    
    memory->destroy(status[i].angle_type);
    memory->destroy(status[i].angle_atom1);
    memory->destroy(status[i].angle_atom2);
    memory->destroy(status[i].angle_atom3);
    
    memory->destroy(status[i].dihedral_type);
    memory->destroy(status[i].dihedral_atom1);
    memory->destroy(status[i].dihedral_atom2);
    memory->destroy(status[i].dihedral_atom3);
    memory->destroy(status[i].dihedral_atom4);
    
    memory->destroy(status[i].improper_type);
    memory->destroy(status[i].improper_atom1);
    memory->destroy(status[i].improper_atom2);
    memory->destroy(status[i].improper_atom3);
    memory->destroy(status[i].improper_atom4);
  }
  
  memory->sfree(status);

  delete cec;
  delete cec_v2;

}

/* ---------------------------------------------------------------------- */

void EVB_Complex::setup()
{
  _EVB_REFRESH_AVEC_POINTERS;
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::delete_state(int x)
{
  int is_delete[MAX_STATE];
  int head = x;
  int tail = x+1;
  state_per_shell[shell[x]-1]--;

  memset(is_delete,0, sizeof(int)*nstate);
  is_delete[x] = 1;

  for(int i=tail; i<nstate; i++) if(is_delete[parent_id[i]])
  {
      is_delete[i] = 1;
      state_per_shell[shell[i]-1]--;
  }   
  
  while (tail < nstate)
  {
      if(!is_delete[tail])
      {
	  parent_id[head] = parent_id[tail];
	  shell[head] = shell[tail];
	  molecule_A[head] = molecule_A[tail];
	  molecule_B[head] = molecule_B[tail];
	  reaction[head] = reaction[tail];
	  path[head] = path[tail];
          
	  distance[head] = distance[tail];
          
          Cs[head] = Cs[tail];
          Cs2[head] = Cs2[tail];
	  
	  for(int i=tail+1; i<nstate; i++)
              if(parent_id[i]==tail) parent_id[i]=head;
	  
	  head++;
      }
      tail++;
  }
  
  nstate = head;
}

/* ---------------------------------------------------------------------- */

#define SWAP(a,b,t) {t=a;a=b;b=t;}

void EVB_Complex::exchange_state(int x, int y)
{
  int _it; double _dt;
  
  SWAP(parent_id[x],parent_id[y],_it);
  SWAP(shell[x],shell[y],_it);
  SWAP(molecule_A[x],molecule_A[y],_it);
  SWAP(molecule_B[x],molecule_B[y],_it);
  SWAP(reaction[x],reaction[y],_it);
  SWAP(path[x],path[y],_it);
  
  SWAP(distance[x],distance[y],_dt);
  
  int i=y+1; while(shell[i]<=shell[y]) i++;
  while(i<nstate)
  {
    if (parent_id[i]==x) parent_id[i]=y;
	else if (parent_id[i]==y) parent_id[i]=x;
	
	i++;
  }
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::search_state()
{ 
  int nall = atom->nlocal + atom->nghost;
  double **x = atom->x;
  int** molecule_map = evb_engine->molecule_map;
  
  int max_shell = evb_chain->max_shell;
  memset(state_per_shell,0,sizeof(int)*max_shell);
  
  nextra_coupling = 0;
  
  // Set first state
  molecule_A[0] = -1;
  parent_id[0]=-1;
  extra_coupling[0]=-1;
  reaction[0] = -1;
  path[0] = -1;
  shell[0] = 0;
  molecule_B[0] = rc_start;

  int iatom = 1;
  while(molecule_map[rc_start][iatom]==-1) iatom++;
  rc_etype = evb_engine->mol_type[molecule_map[rc_start][iatom]];

#ifdef DLEVB_MODEL_SUPPORT
  double shell_rcut;
  switch (rc_etype) {
  case(2):              // Search starts from hydronium
    shell_rcut = 2.5;
    break;
  case(3):              // Search starts from amino acid
    shell_rcut = 3.0;
    break;
  default:
    shell_rcut = 2.5;   // Distance used in water models
    break;
  }
  shell_rcut*=shell_rcut;

  int jshell = 1;
#endif
                                
  int state_head = 0;
  int& state_tail = nstate;
  nstate = 1;
  int ishell = 1;
  
  //return;
  
  while(state_head<state_tail)
  {
      // MS-EVB2: Refine states
      if(evb_engine->bRefineStates && shell[state_head] == ishell)
      {
          refine_state (state_head);
          ishell++;
      }

#ifdef DLEVB_MODEL_SUPPORT
      if(shell[state_head] == jshell) { 
	delete_shell_states(jshell);
	jshell++;
      }
#endif
      
      // Skip it if this molecule is used before (Just for test here)   
      int t; for(t=0; t<state_head; t++) if(shell[t]<shell[state_head] && molecule_B[t]==molecule_B[state_head]) break;
      if(t<state_head) { state_head++; continue; }
     
      // Get the evb_type of the reaction center to loop its chains
      int tmp = molecule_map[molecule_B[state_head]][1];
      int type = evb_engine->mol_type[tmp];
      //fprintf(screen, "[EVB] Add state from evb_type: %d\n",type);
      
      // Loop its chains of this molecule
      int start = evb_chain->index[type-1];
      int end = start + evb_chain->count[type-1];

      for(int i=start; i<end; i++)
      {    
          // Get the host atom of this chain
          int A = molecule_map[molecule_B[state_head]][evb_chain->host[i]];
          
          // Loop all the atoms in this domain (processor)
          for(int B=0; B<nall; B++)
          {
              // Check whether this atom can be a client target
              if (    evb_engine->mol_type[B] == evb_chain->target[i]
                      && evb_engine->mol_index[B] == evb_chain->client[i]
                      && atom->molecule[B] != molecule_B[state_head])
              {
                  // Check whether target is used before

                  int bRepeat = 0;
                  int k = state_tail-1;
                  while ( molecule_A[k] == atom->molecule[A]      &&
                          reaction[k]   == evb_chain->reaction[i] &&
                          path[k]       == evb_chain->path[i] )
                  {
                      if(molecule_B[k] == atom->molecule[B])
                      {  bRepeat = 1; break; }
                      
                      k--;
                  }
                  if(bRepeat) continue;
                  
                  // Check the current state is beyond the shell limit of this chain
        	  int nshell = shell[state_head]+1;
                  if (nshell>evb_chain->shell_limit[i]) continue;
                  
                  // Check the distance between the host and client
                  double dr2,dr[3];
                  VECTOR_SUB(dr,x[A],x[B]);
                  VECTOR_PBC(dr);
                  VECTOR_R2(dr2,dr);
                  
                  //fprintf(screen,"x(%lf %lf) y(%lf %lf) z(%lf %lf) dr2 %lf\n",x[A][0],x[B][0],x[A][1],x[B][1],x[A][2],x[B][2],dr2);
                  
                  // Add a new state
#ifdef DLEVB_MODEL_SUPPORT
                  if(dr2 < shell_rcut)
                  {
                      parent_id[state_tail] = state_head;
                      shell[state_tail] = nshell;
                      molecule_A[state_tail] = atom->molecule[A];
                      molecule_B[state_tail] = atom->molecule[B];
                      reaction[state_tail] = evb_chain->reaction[i];
                      path[state_tail] = evb_chain->path[i];
           
                      distance[state_tail] = dr2;
                      extra_coupling[state_tail] = 0;
                      
                      state_tail++;
                      state_per_shell[nshell-1]++;
                  
                      if(state_tail==MAX_STATE)
                      {  nstate = state_tail; return; }
                  }
#else
                  if(dr2 < evb_chain->distance_limit[i])
                  {
                      parent_id[state_tail] = state_head;
                      shell[state_tail] = nshell;
                      molecule_A[state_tail] = atom->molecule[A];
                      molecule_B[state_tail] = atom->molecule[B];
                      reaction[state_tail] = evb_chain->reaction[i];
                      path[state_tail] = evb_chain->path[i];
           
                      distance[state_tail] = dr2;
                      extra_coupling[state_tail] = 0;
                      
                      // MS-EVB3: Add extra coupling
                      if(evb_engine->bExtraCouplings)
                      {
                          int istate = state_tail-1;
                          while(  parent_id[istate] == parent_id[state_tail] &&
                                  molecule_A[istate] == molecule_A[state_tail] && 
                                  reaction[istate] == reaction[state_tail] && 
                                  path[istate] == path[state_tail] &&
                                  molecule_B[istate] != molecule_B[state_tail] )
                          {
                              if(nextra_coupling==MAX_EXTRA) break;

                              extra_coupling[state_tail]++;
                              nextra_coupling++;
                            
                              istate--;
                          }
                      }
                      
                      state_tail++;
                      state_per_shell[nshell-1]++;
                  
                      //if(state_tail==MAX_STATE || state_tail == evb_engine->user_max_state) // AWGL
                      if(state_tail==MAX_STATE) 
                      {  nstate = state_tail; return; }
                  }
#endif
              }
          }
      }
      
      state_head++;
  }
 
  // nstate = 1;
}

/* ---------------------------------------------------------------------- */
  
/*****************************************************/
/******MS-EVB2 STATE_SEARCH       ********************/
/*****************************************************/
  
void EVB_Complex::refine_state(int head)
{ 
  int k;
  int start = head;
  
  while(head<nstate)
  {
        int id=0; 
	double min=10000.0;
	  
	for(k=head; k<nstate; k++) if (distance[k]<min)
	{  min = distance[k]; id=k; }
	  
	if(head!=id) exchange_state(head,id);
	  
	for(k=start; k<head; k++)
	{  
	  if(molecule_B[k]==molecule_B[head] && reaction[k]==reaction[head])
	  {  delete_state(head); break; }
		
	  if(molecule_A[k]==molecule_A[head] && reaction[k]==reaction[head] && path[k]==path[head])
	  {  delete_state(head); break; }
	}
	
	if(k == head) head++;
  }
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::pack_state()
{
  int pos = 0;
  
  state_buf[pos]=nstate; pos++;
  state_buf[pos]=rc_etype; pos++;
  
  int size = sizeof(int)*nstate;
  memcpy(state_buf+pos,shell,size); pos+=nstate;
  memcpy(state_buf+pos,parent_id,size); pos+=nstate;
  memcpy(state_buf+pos,molecule_A,size); pos+=nstate;
  memcpy(state_buf+pos,molecule_B,size); pos+=nstate;
  memcpy(state_buf+pos,reaction,size); pos+=nstate;
  memcpy(state_buf+pos,path,size); pos+=nstate;
  memcpy(state_buf+pos,extra_coupling,size); pos+=nstate;
  
  buf_size = pos;
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::unpack_state()
{
  int pos = 0;
  
  nstate = state_buf[pos]; pos++;
  rc_etype = state_buf[pos]; pos++;
  
  int size = sizeof(int)*nstate;
  memcpy(shell,state_buf+pos,size); pos+=nstate;
  memcpy(parent_id,state_buf+pos,size); pos+=nstate;
  memcpy(molecule_A,state_buf+pos,size); pos+=nstate;
  memcpy(molecule_B,state_buf+pos,size); pos+=nstate;
  memcpy(reaction,state_buf+pos,size); pos+=nstate;
  memcpy(path,state_buf+pos,size); pos+=nstate;
  memcpy(extra_coupling,state_buf+pos,size); pos+=nstate;
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::compute_qsqsum()
{
    if(!evb_kspace) return;

    double *_qsqsum = evb_type->qsqsum; 

    qsqsum = _qsqsum[rc_etype-1];

    for(int i=1; i<nstate; i++)
    {
        bool flag = true;

        for(int j=0; j<i; j++) 
	  if (molecule_B[i]==molecule_B[j]) 
	  {
	    flag = false;
	    break;
	  }

	if(flag)
	{
	  int etypeB = evb_reaction->reactant_B[reaction[i]-1];
          qsqsum += _qsqsum[etypeB-1];
	}
    }
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::build_cplx_map()
{
  int* complex = evb_engine->complex_molecule;
  int** map = evb_engine->molecule_map;
  int* atoms =evb_engine->complex_atom;
  
  int nmolecule = evb_engine->nmolecule;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  
  int* kernels = evb_engine->kernel_atom;
  int* is_kernel = evb_type->is_kernel;
  
  if(nall>natom)
  {
    natom = nall;
    cplx_list = (int*) memory->srealloc(cplx_list,natom*sizeof(int),"EVB_Complex:cplx_list");
  }
  
  memset(complex, 0 , nmolecule*sizeof(int));   
  memset(atoms, 0 , nall*sizeof(int));
  memset(kernels, 0 , nall*sizeof(int));
  natom_cplx = nghost_cplx = 0;
  
  for(int i=0; i<nstate; i++)
  {
      int mol_id = molecule_B[i];
      complex[mol_id] = id;
      
      int j;
      for(j=0; j<i; j++) if (molecule_B[j]==molecule_B[i]) break;
      if(j<i) continue;

      int natm = map[mol_id][0];
      int *iatm = map[mol_id]+1;

      for(j=0; j<natm; j++)
	  if(iatm[j]>=nlocal) nghost_cplx++;
  }
  
  int *mol_type = evb_engine->mol_type;
  int *mol_index = evb_engine->mol_index;

  for(int i=0; i<nall; i++)
  {	  
    if(mol_type[i]>0)
    {
      int start = evb_type->type_index[mol_type[i]-1];
      if( is_kernel[start+mol_index[i]-1] == 1) kernels[i] = id;
    }
	
    int molecule_id = atom->molecule[i];
    if(complex[molecule_id]) 
    { 
      atoms[i] = id; 
      cplx_list[natom_cplx++]=i;
    }
  }
  for(nlocal_cplx=0; nlocal_cplx<natom_cplx; nlocal_cplx++)
    if(cplx_list[nlocal_cplx]>=nlocal) break;

  //fprintf(screen,"natom_cplx = %d(local) + %d(ghost) at rank %d\n",nlocal_cplx,nghost_cplx,evb_engine->me);
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::setup_avec()
{
  int old_status_nstate = status_nstate;
  
  if(nstate>status_nstate)
  {
    status = (STATE_INFO*) memory->srealloc(status,sizeof(STATE_INFO)*nstate,"EVB_Complex:status");
    status_nstate = nstate;
  }
  
  if(natom_cplx>status_natom)
  {
      status_natom = natom_cplx;
      
      for(int i=0; i<old_status_nstate; i++)
      {
        status[i].type = (int*) memory->srealloc(status[i].type,sizeof(int)*status_natom,"EVB_Complex:status");
        status[i].molecule = (int*) memory->srealloc(status[i].molecule,sizeof(int)*status_natom,"EVB_Complex:status");
        status[i].mol_type = (int*) memory->srealloc(status[i].mol_type,sizeof(int)*status_natom,"EVB_Complex:status");
        status[i].mol_index = (int*) memory->srealloc(status[i].mol_index,sizeof(int)*status_natom,"EVB_Complex:status");
        status[i].num_bond = (int*) memory->srealloc(status[i].num_bond,sizeof(int)*status_natom,"EVB_Complex:status");
        status[i].num_angle = (int*) memory->srealloc(status[i].num_angle,sizeof(int)*status_natom,"EVB_Complex:status");
        status[i].num_dihedral = (int*) memory->srealloc(status[i].num_dihedral,sizeof(int)*status_natom,"EVB_Complex:status");
        status[i].num_improper = (int*) memory->srealloc(status[i].num_improper,sizeof(int)*status_natom,"EVB_Complex:status");
        status[i].q = (double*) memory->srealloc(status[i].q,sizeof(double)*status_natom,"EVB_Complex:status");
        
	memory->grow(status[i].bond_type,status_natom,atom->bond_per_atom,"EVB_Complex:status");
	memory->grow(status[i].bond_atom,status_natom,atom->bond_per_atom,"EVB_Complex:status");
        
	memory->grow(status[i].angle_type,status_natom,atom->angle_per_atom,"EVB_Complex:status");
	memory->grow(status[i].angle_atom1,status_natom,atom->angle_per_atom,"EVB_Complex:status");
	memory->grow(status[i].angle_atom2,status_natom,atom->angle_per_atom,"EVB_Complex:status");
	memory->grow(status[i].angle_atom3,status_natom,atom->angle_per_atom,"EVB_Complex:status");
        
	memory->grow(status[i].dihedral_type,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");
	memory->grow(status[i].dihedral_atom1,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");
	memory->grow(status[i].dihedral_atom2,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");
	memory->grow(status[i].dihedral_atom3,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");
	memory->grow(status[i].dihedral_atom4,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");

	memory->grow(status[i].improper_type,status_natom,atom->improper_per_atom,"EVB_Complex:status");
	memory->grow(status[i].improper_atom1,status_natom,atom->improper_per_atom,"EVB_Complex:status");
	memory->grow(status[i].improper_atom2,status_natom,atom->improper_per_atom,"EVB_Complex:status");
	memory->grow(status[i].improper_atom3,status_natom,atom->improper_per_atom,"EVB_Complex:status");
	memory->grow(status[i].improper_atom4,status_natom,atom->improper_per_atom,"EVB_Complex:status");
      }
    
  }
  
  for(int i=old_status_nstate; i<status_nstate; i++)
  {
    status[i].type = (int*) memory->smalloc(sizeof(int)*status_natom,"EVB_Complex:status");
    status[i].molecule = (int*) memory->smalloc(sizeof(int)*status_natom,"EVB_Complex:status");
    status[i].mol_type = (int*) memory->smalloc(sizeof(int)*status_natom,"EVB_Complex:status");
    status[i].mol_index = (int*) memory->smalloc(sizeof(int)*status_natom,"EVB_Complex:status");
    status[i].num_bond = (int*) memory->smalloc(sizeof(int)*status_natom,"EVB_Complex:status");
    status[i].num_angle = (int*) memory->smalloc(sizeof(int)*status_natom,"EVB_Complex:status");
    status[i].num_dihedral = (int*) memory->smalloc(sizeof(int)*status_natom,"EVB_Complex:status");
    status[i].num_improper = (int*) memory->smalloc(sizeof(int)*status_natom,"EVB_Complex:status");
    status[i].q = (double*) memory->smalloc(sizeof(double)*status_natom,"EVB_Complex:status");
    
    memory->create(status[i].bond_type,status_natom,atom->bond_per_atom,"EVB_Complex:status");
    memory->create(status[i].bond_atom,status_natom,atom->bond_per_atom,"EVB_Complex:status");
    
    memory->create(status[i].angle_type ,status_natom,atom->angle_per_atom,"EVB_Complex:status");
    memory->create(status[i].angle_atom1,status_natom,atom->angle_per_atom,"EVB_Complex:status");
    memory->create(status[i].angle_atom2,status_natom,atom->angle_per_atom,"EVB_Complex:status");
    memory->create(status[i].angle_atom3,status_natom,atom->angle_per_atom,"EVB_Complex:status");
    
    memory->create(status[i].dihedral_type ,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");
    memory->create(status[i].dihedral_atom1,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");
    memory->create(status[i].dihedral_atom2,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");
    memory->create(status[i].dihedral_atom3,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");
    memory->create(status[i].dihedral_atom4,status_natom,atom->dihedral_per_atom,"EVB_Complex:status");
    
    memory->create(status[i].improper_type ,status_natom,atom->improper_per_atom,"EVB_Complex:status");
    memory->create(status[i].improper_atom1,status_natom,atom->improper_per_atom,"EVB_Complex:status");
    memory->create(status[i].improper_atom2,status_natom,atom->improper_per_atom,"EVB_Complex:status");
    memory->create(status[i].improper_atom3,status_natom,atom->improper_per_atom,"EVB_Complex:status");
    memory->create(status[i].improper_atom4,status_natom,atom->improper_per_atom,"EVB_Complex:status");
  }
  
  current_status = 0;
}

void EVB_Complex::setup_offdiag()
{ 
  int noffdiag = nstate-1;
  
  if(noffdiag > max_offdiag)
  {
    max_offdiag = noffdiag;

    if(EVB_OffDiag::max_nexch)
    {
        nexch_off = (int*) memory->srealloc(nexch_off,sizeof(int)*max_offdiag,"EVB_Complex:nexch_off");
        memory->grow(iexch_off,max_offdiag,EVB_OffDiag::max_nexch*27,"EVB_Complex:iexch_off");
        memory->grow(qexch_off,max_offdiag,EVB_OffDiag::max_nexch*27,"EVB_Complex:qexch_off");
    }

  }

  if(nextra_coupling > max_extra)
  {
    max_extra = nextra_coupling;
    extra_i = (int*) memory->srealloc(extra_i,sizeof(int)*max_extra,"EVB_Complex:extra_i");
    extra_j = (int*) memory->srealloc(extra_j,sizeof(int)*max_extra,"EVB_Complex:extra_j");

    if(EVB_OffDiag::max_nexch)
    {        
        nexch_extra = (int*) memory->srealloc(nexch_extra,sizeof(int)*max_extra,"EVB_Complex:nexch_extra");
        memory->grow(iexch_extra,max_extra,EVB_OffDiag::max_nexch*27,"EVB_Complex:iexch_extra");
        memory->grow(qexch_extra,max_extra,EVB_OffDiag::max_nexch*27,"EVB_Complex:qexch_extra");
    }
  }
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::build_state(int index)
{
  if(current_status!=parent_id[index]) load_avec(parent_id[index]);

  /*********************************************************************/
  /******* Deal with the atom_vec             **************************/
  /*********************************************************************/

  int **map = evb_engine->molecule_map;
  int *kernel_atom = evb_engine->kernel_atom;
  
  // update the atom_vec information
  
  // Get general information
  int ma = molecule_A[index];
  int mb = molecule_B[index];
  int re = reaction[index]-1;
  int pa = path[index]-1;
  //fprintf(screen,"%d %d %d %d\n",ma,mb,re,pa);
  int ta = evb_reaction->product_A[re];
  int tb = evb_reaction->product_B[re];
  
  int* count = (evb_reaction->Path)[re][pa].atom_count;
  int* mp = (evb_reaction->Path)[re][pa].moving_part;
  int* fp = (evb_reaction->Path)[re][pa].first_part;
  int* sp = (evb_reaction->Path)[re][pa].second_part;
  
  int m1, m2, t2;
  if(evb_reaction->backward[re]) { m1 = mb; m2 = ma; t2=ta;}
  else { m1=ma; m2=mb; t2=tb; }

  for(int i=0; i<natom_cplx; i++)
  {    
    int id = cplx_list[i];
    bool change_flag = false;
    
	if(kernel_atom[id]) num_bond[id] = num_angle[id] = num_dihedral[id] = num_improper[id] = 0;
    
    // Deal with moving part
    
    for(int j=0; j<count[0]; j++)
    {
      int index = mp[j*2];
      int target = mp[j*2+1];

      if(m1 == molecule[id] && index == mol_index[id])
      {//fprintf(screen,"m %d",m1);
        molecule[id] = m2;
        evb_reaction->change_atom(id,t2,target);
        change_flag = true;        
      }      
      if(change_flag) break;
    }    
    if(change_flag) continue;
    
    // Deal with rest part
    
    for(int j=0; j<count[1]; j++)
    {
      int index = fp[j*2];
      int target = fp[j*2+1];
      
      if(ma == molecule[id] && index == mol_index[id])
      {//fprintf(screen,"r ");
        evb_reaction->change_atom(id,ta,target);
        change_flag = true;
      }      
      if(change_flag) break;
    }    
    if(change_flag) continue;
    
    // Deal with new part
    
    for(int j=0; j<count[2]; j++)
    {
      int index = sp[j*2];
      int target = sp[j*2+1];
      
      if(mb == molecule[id] && index == mol_index[id])
      {//fprintf(screen,"n ");
        evb_reaction->change_atom(id,tb,target);
        change_flag = true;
      }      
      if(change_flag) break;
    }
  }
    
  update_mol_map();
  update_list();
  
  /*********************************************************************/
  /******* EVB KSpace                   ********************************/
  /*********************************************************************/
  
  int rA = evb_reaction->reactant_A[re];
  int rB = evb_reaction->reactant_B[re];

  if(evb_kspace)
  {
    double chg;
    double *_qsqsum = evb_type->qsqsum;
    chg = _qsqsum[ta-1]+_qsqsum[tb-1]-_qsqsum[rA-1]-_qsqsum[rB-1];
    qsqsum += chg;
  }

  /***********************************/
  /****** Atoms struct ***************/
  /***********************************/

  int *narray;
  narray = evb_type->nbonds;
  atom->nbonds    += (narray[ta-1] + narray[tb-1] - narray[rA-1] - narray[rB-1]);
  narray = evb_type->nangles;
  atom->nangles   += (narray[ta-1] + narray[tb-1] - narray[rA-1] - narray[rB-1]);
  narray = evb_type->ndihedrals;
  atom->ndihedrals+= (narray[ta-1] + narray[tb-1] - narray[rA-1] - narray[rB-1]);
  narray = evb_type->nimpropers;
  atom->nimpropers+= (narray[ta-1] + narray[tb-1] - narray[rA-1] - narray[rB-1]);

  // set ID
  current_status = index;
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::update_mol_map()
{
#if defined (_OPENMP)
  update_mol_map_omp();
  return;
#endif
  
  const int nmolecule = evb_engine->nmolecule;
  const int * const cplx_mol = evb_engine->complex_molecule;
  const int apm = evb_engine->atoms_per_molecule;
  int ** map = evb_engine->molecule_map;
  
  for(int i=0; i<nmolecule; i++) {
    if(evb_engine->complex_molecule[i]==id) {
      memset(map[i], -1, sizeof(int)*(apm));
      map[i][0]=0;
    }
  }
  
  for(int i=0; i<natom_cplx; i++) {
    const int id = cplx_list[i];
    const int moleculeid = molecule[id];
    const int molindexid = mol_index[id];
    if(map[moleculeid][molindexid]==-1) {
      map[moleculeid][0]++;
      map[moleculeid][molindexid]=id;
    }
  }
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::update_mol_map_omp()
{
  const int nmolecule = evb_engine->nmolecule;
  const int * const cplx_mol = evb_engine->complex_molecule;
  const int apm = evb_engine->atoms_per_molecule;
  int ** map = evb_engine->molecule_map;

  const int chunksize = 50;

#if defined (_OPENMP)
#pragma omp parallel default(none) shared(map)
  {
#pragma omp for schedule(static,chunksize)
#endif
    for(int i=0; i<nmolecule; i++) {
      if(evb_engine->complex_molecule[i]==id) {
	memset(map[i], -1, sizeof(int)*(apm));
	map[i][0]=0;
      }
    }
#if defined (_OPENMP)
  }
#endif
  
  for(int i=0; i<natom_cplx; i++) {
    const int id = cplx_list[i];
    const int moleculeid = molecule[id];
    const int molindexid = mol_index[id];
    if(map[moleculeid][molindexid]==-1) {
      map[moleculeid][0]++;
      map[moleculeid][molindexid]=id;
    }
  }
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::update_pair_list()
{
  // No need to edit pair lists when screening states
  if(evb_engine->engine_indicator == ENGINE_INDICATOR_SCREEN) return;

  #if defined (_OPENMP)
  // ** AWGL ** //
  update_pair_list_omp();
  return;
  #endif

  NeighList *list = evb_engine->get_pair_list();
  int inum = list->inum;
  int* ilist = list->ilist;
  int* numneigh = list->numneigh;
  int** firstneigh = list->firstneigh;
  int* mol_type = evb_engine->mol_type;
  int* kernel_atom = evb_engine->kernel_atom;
  int nall = atom->nlocal+atom->nghost;

  
  for(int i=0; i<inum; i++)
  {
      int atomi = ilist[i];
      int numj = numneigh[atomi];
      int *jlist = firstneigh[atomi];
  
      if (mol_type[atomi]==0) continue;

      for(int j=0; j<numj; j++)
      {
          int atomj = jlist[j];
          atomj &= NEIGHMASK;
          
	  if(mol_type[atomj]==0) continue;

	  if(molecule[atomi]!=molecule[atomj])
	  {
	    jlist[j]=atomj;
	    continue;
	  }

          int **bond_map = evb_type->bond_map[mol_type[atomi]-1];
          int bond_info = bond_map[mol_index[atomi]][mol_index[atomj]];
          if(bond_info)
          {
              jlist[j]=atomj ^ (bond_info << SBBITS);
              continue;
          }
          
          if(kernel_atom[atomi]==0 || kernel_atom[atomj]==0) continue;
          else jlist[j]=atomj;
      }
  }

}

/* ---------------------------------------------------------------------- */

void EVB_Complex::update_pair_list_omp()
{
  // ** AWGL : OpenMP version of update_pair_list ** //
  
  NeighList *list = evb_engine->get_pair_list();
  const int inum = list->inum;
  const int * const ilist = list->ilist;
  const int * const numneigh = list->numneigh;
  int * const * const firstneigh = list->firstneigh;
  const int * const mol_type = evb_engine->mol_type;
  const int * const kernel_atom = evb_engine->kernel_atom;
  const int nall = atom->nlocal+atom->nghost;

  int atomj;
  bool bool_kernel_atom_I;

#if defined (_OPENMP)
#pragma omp parallel default(none) \
  private(atomj, bool_kernel_atom_I)
  {
#pragma omp for schedule(dynamic)
#endif
    for(int i=0; i<inum; ++i) {
      const int atomi = ilist[i];
      const int jnum = numneigh[atomi];
      int *jlist = firstneigh[atomi];
      
      if(mol_type[atomi]==0) continue;

      const int molecule_I = molecule[atomi];
      const int mol_index_I = mol_index[atomi];
      int * const * const bond_map = evb_type->bond_map[ mol_type[atomi]-1 ];
      if(kernel_atom[atomi] == 0) bool_kernel_atom_I = true;
      else bool_kernel_atom_I = false;
      
      for(int j=0; j<jnum; ++j) {
	atomj = jlist[j];
	atomj &= NEIGHMASK;
	
	if(mol_type[atomj]==0) continue;
	
	if(molecule_I != molecule[atomj]) {
	  jlist[j] = atomj;
	  continue;
	}
	
	const int bond_info = bond_map[ mol_index_I ][ mol_index[atomj] ];
	if(bond_info) {
	  jlist[j]=atomj ^ (bond_info << SBBITS);
	  continue;
	}
	
	if(bool_kernel_atom_I) continue;
	if(kernel_atom[atomj]==0) continue;
	
	jlist[j]=atomj;
      }
    }   
#if defined (_OPENMP)
  } //close OpenMP bracket
#endif
  
}
/* ---------------------------------------------------------------------- */

void EVB_Complex::update_list()
{
  if(evb_engine->mp_verlet && evb_engine->mp_verlet->is_master==0) return;
  
  update_pair_list();
  update_bond_list();
}


void EVB_Complex::update_bond_list()
{
  /*********************************************************************/
  /******* Deal with the Pair interaction ******************************/
  /*********************************************************************/

  int **map = evb_engine->molecule_map;
  int nall = atom->nlocal+atom->nghost;
  int *kernel_atom = evb_engine->kernel_atom;
 
  /*********************************************************************/
  /******* Deal with the bond structure ********************************/
  /*********************************************************************/
    
  int bond_size = evb_list->max_bond;
  int **bond = neighbor->bondlist;
  int nbond = 0;
 
  for(int i=0; i<natom_cplx; i++)
  {
    int atom_id = cplx_list[i];
    if (!kernel_atom[atom_id]) continue;
    if (atom_id>=atom->nlocal) break;
   
    int index = evb_type->type_index[mol_type[atom_id]-1]+mol_index[atom_id]-1;
    num_bond[atom_id] = evb_type->num_bond[index];

    for(int j=0; j<num_bond[atom_id]; j++)
    {
      int ba = map[molecule[atom_id]][evb_type->bond_atom[index][j]];
      if(ba==-1)  
      {
	fprintf(screen,"status %d\n",current_status);
	fprintf(screen,"atom_id= %d  molecule= %d  index= %d  bond_atom= %d\n",atom_id,molecule[atom_id],index,evb_type->bond_atom[index][j]);
	
	   if(screen)
           { 
             {
               fprintf(screen,"    state parent  shell  mol_A  mol_B  react   path  coupling\n");
               for(int kkk2=0; kkk2<nstate; kkk2++)
                 fprintf(screen,"   %6d %6d %6d %6d %6d %6d %6d %6d\n",kkk2,evb_complex->parent_id[kkk2],
                         evb_complex->shell[kkk2],evb_complex->molecule_A[kkk2],evb_complex->molecule_B[kkk2],
                         evb_complex->reaction[kkk2],evb_complex->path[kkk2],evb_complex->extra_coupling[kkk2]);
	     }
	   }

	   // for(int k=1; k<evb_engine->nmolecule; k++)
	   //   fprintf(screen,"mol %d(%d-%d): %d %d %d %d\n", k, map[k][0],mol_type[map[k][1]],
	   // 	 map[k][1],map[k][2],map[k][3],map[k][4]);

	   int k = molecule[atom_id];
	   fprintf(screen,"mol %d(%d-%d): %d %d %d %d\n", k, map[k][0],mol_type[map[k][1]],
		   map[k][1],map[k][2],map[k][3],map[k][4]);
	   
	error->one(FLERR,"[EVB] Missed bond_atom!");
      }

      bond_atom[atom_id][j]=tag[ba];
      bond_type[atom_id][j]=evb_type->bond_type[index][j];
      
      bond[nbond][0]=atom_id;
      bond[nbond][1]=domain->closest_image(atom_id, ba);
      bond[nbond][2]=bond_type[atom_id][j]; 
      // printf("BOND TYPE %d %d %d\n", atom->tag[bond[nbond][0]], atom->tag[bond[nbond][1]], bond[nbond][2]);
      
      nbond++;
      
      if(nbond>=bond_size) error->one(FLERR,"[EVB] Overflow the size of bonds!");
    }
  }
  neighbor->nbondlist = evb_list->n_evb_bond = nbond;

  /*********************************************************************/
  /******* Deal with the angle structure *******************************/
  /*********************************************************************/
  
  int angle_size = evb_list->max_angle;
  int **angle = neighbor->anglelist;
  int nangle = 0;
  //fprintf(screen,"angle_atom3 -> %ld %ld\n",(long)angle_atom3[23],(long)(atom->angle_atom3[23]));

  //for(int i=0; i<neighbor->nanglelist; i++)
  //  fprintf(screen,"angle %d: %d %d %d\n",i,angle[i][0],angle[i][1],angle[i][2]);
  
  for(int i=0; i<natom_cplx; i++)
  {
    int atom_id = cplx_list[i];
	if (!kernel_atom[atom_id]) continue;
    if (atom_id>=atom->nlocal) break;
    
    int index = evb_type->type_index[mol_type[atom_id]-1]+mol_index[atom_id]-1;
    num_angle[atom_id] = evb_type->num_angle[index];
    
    for(int j=0; j<num_angle[atom_id]; j++)
    {
      int a1 = map[molecule[atom_id]][evb_type->angle_atom1[index][j]];
      int a2 = map[molecule[atom_id]][evb_type->angle_atom2[index][j]];
      int a3 = map[molecule[atom_id]][evb_type->angle_atom3[index][j]];
      if(a1==-1 || a2==-1 || a3==-1)  error->one(FLERR,"[EVB] Missed angle_atom!");
      
      angle_atom1[atom_id][j]=tag[a1];
      angle_atom2[atom_id][j]=tag[a2];
      angle_atom3[atom_id][j]=tag[a3];
      angle_type[atom_id][j]=evb_type->angle_type[index][j];
      
      angle[nangle][0]=domain->closest_image(a2, a1);
      angle[nangle][1]=a2;
      angle[nangle][2]=domain->closest_image(a2, a3);
      angle[nangle][3]=angle_type[atom_id][j];
      
      //fprintf(screen,"angle %d: %d %d\n",nangle,atom_id,angle_type[atom_id][j]);
      nangle++;
      
      if(nangle>=angle_size) error->one(FLERR,"[EVB] Overflow the size of angles!");
    }
  }
  
  neighbor->nanglelist = evb_list->n_evb_angle = nangle;  
  
  /*********************************************************************/
  /******* Deal with the dihedral structure ****************************/
  /*********************************************************************/
  
  int dihedral_size = evb_list->max_dihedral;
  int **dihedral = neighbor->dihedrallist;
  int ndihedral = 0;
  
  //for(int i=0; i<neighbor->ndihedrallist; i++)
  //  fprintf(screen,"dihedral %d: %d %d %d\n",i,dihedral[i][0],dihedral[i][1],dihedral[i][2]);
  
  for(int i=0; i<natom_cplx; i++)
  {
    int atom_id = cplx_list[i];
	if (!kernel_atom[atom_id]) continue;
    if (atom_id>=atom->nlocal) break;
    
    int index = evb_type->type_index[mol_type[atom_id]-1]+mol_index[atom_id]-1;
    num_dihedral[atom_id] = evb_type->num_dihedral[index];
    
    for(int j=0; j<num_dihedral[atom_id]; j++)
    {
      int a1 = map[molecule[atom_id]][evb_type->dihedral_atom1[index][j]];
      int a2 = map[molecule[atom_id]][evb_type->dihedral_atom2[index][j]];
      int a3 = map[molecule[atom_id]][evb_type->dihedral_atom3[index][j]];
      int a4 = map[molecule[atom_id]][evb_type->dihedral_atom4[index][j]];
      if(a1==-1 || a2==-1 || a3==-1 || a4==-1)  error->one(FLERR,"[EVB] Missed dihedral_atom!");
      
      dihedral_atom1[atom_id][j]=tag[a1];
      dihedral_atom2[atom_id][j]=tag[a2];
      dihedral_atom3[atom_id][j]=tag[a3];
      dihedral_atom4[atom_id][j]=tag[a4];
      dihedral_type[atom_id][j]=evb_type->dihedral_type[index][j];
      
      dihedral[ndihedral][0]=domain->closest_image(a2, a1);
      dihedral[ndihedral][1]=a2;
      dihedral[ndihedral][2]=domain->closest_image(a2, a3);
      dihedral[ndihedral][3]=domain->closest_image(a2, a4);
      dihedral[ndihedral][4]=dihedral_type[atom_id][j];
      
      //fprintf(screen,"dihedral %d: %d %d %d\n",ndihedral,atom_id,ba,dihedral_type[atom_id][j]);
      ndihedral++;
      
      if(ndihedral>=dihedral_size) error->one(FLERR,"[EVB] Overflow the size of dihedrals!");
    }
  }
  
  neighbor->ndihedrallist = evb_list->n_evb_dihedral = ndihedral;  
  
  
  /*********************************************************************/
  /******* Deal with the improper structure ****************************/
  /*********************************************************************/
  
  int improper_size = evb_list->max_improper;
  int **improper = neighbor->improperlist;
  int nimproper = 0;
  
  //for(int i=0; i<neighbor->nimproperlist; i++)
  //  fprintf(screen,"improper %d: %d %d %d\n",i,improper[i][0],improper[i][1],improper[i][2]);
  
  for(int i=0; i<natom_cplx; i++)
  {
    int atom_id = cplx_list[i];
	if (!kernel_atom[atom_id]) continue;
    if (atom_id>=atom->nlocal) break;
    
    int index = evb_type->type_index[mol_type[atom_id]-1]+mol_index[atom_id]-1;
    num_improper[atom_id] = evb_type->num_improper[index];
    
    for(int j=0; j<num_improper[atom_id]; j++)
    {
      int a1 = map[molecule[atom_id]][evb_type->improper_atom1[index][j]];
      int a2 = map[molecule[atom_id]][evb_type->improper_atom2[index][j]];
      int a3 = map[molecule[atom_id]][evb_type->improper_atom3[index][j]];
      int a4 = map[molecule[atom_id]][evb_type->improper_atom4[index][j]];
      if(a1==-1 || a2==-1 || a3==-1 || a4==-1)  error->one(FLERR,"[EVB] Missed improper_atom!");
      
      improper_atom1[atom_id][j]=tag[a1];
      improper_atom2[atom_id][j]=tag[a2];
      improper_atom3[atom_id][j]=tag[a3];
      improper_atom4[atom_id][j]=tag[a4];
      improper_type[atom_id][j]=evb_type->improper_type[index][j];
      
      improper[nimproper][0]=domain->closest_image(a2, a1);
      improper[nimproper][1]=a2;
      improper[nimproper][2]=domain->closest_image(a2, a3);
      improper[nimproper][3]=domain->closest_image(a2, a4);
      improper[nimproper][4]=improper_type[atom_id][j];
      
      //fprintf(screen,"improper %d: %d %d %d %d -> %d\n",nimproper,atom->tag[a1],atom->tag[a2],
	  //atom->tag[a3],atom->tag[a4],improper[nimproper][4]);
      
	  nimproper++;
      
      if(nimproper>=improper_size) error->one(FLERR,"[EVB] Overflow the size of impropers!");
    }
  }
  
  neighbor->nimproperlist = evb_list->n_evb_improper = nimproper;  
  
  /*********************************************************************/
  /******* Finish                       ********************************/
  /*********************************************************************/
  
  /*********************************************************************/
  /******* Just for test                ********************************/
  /*********************************************************************
  fprintf(screen,"Testing ...\n");
  fprintf(screen,"[avec] ...\n");
  for(int i=0; i<nall; i++)
    fprintf(screen, "tag %d   atp %d   q %8lf   molecule %d   evb_type %d   evb_index %d\n", tag[i], type[i], q[i], molecule[i], evb_itype[i], evb_index[i]);
  
  fprintf(screen,"[bond] ...\n");
  for(int i=0; i<neighbor->nbondlist; i++)
    fprintf(screen, "%d %d %d\n",neighbor->bondlist[i][0],neighbor->bondlist[i][1],neighbor->bondlist[i][2]);
  
  fprintf(screen,"[angle] ...\n");
  for(int i=0; i<neighbor->nanglelist; i++)
    fprintf(screen, "%d %d %d %d\n",neighbor->anglelist[i][0],neighbor->anglelist[i][1],
	                                neighbor->anglelist[i][2],neighbor->anglelist[i][3]);

  fprintf(screen,"[pair list] ...\n");
  for(int i=0; i<inum; i++)
  {
    fprintf(screen,"%d->",ilist[i]);
	for(int j=0; j<numneigh[ilist[i]]; j++) fprintf(screen," %d",firstneigh[ilist[i]][j]);
	fprintf(screen,"\n");
  }
  /*********************************************************************/
}


/* ---------------------------------------------------------------------- */

void EVB_Complex::save_avec(int index)
{
  STATE_INFO* state = status+index;
  
  for(int i=0; i<natom_cplx; i++)
  {
    int id=cplx_list[i];

    state->type[i]=type[id];
    state->mol_type[i]=mol_type[id];
    state->mol_index[i]=mol_index[id];
    state->molecule[i]=molecule[id];
    state->q[i]=q[id];
 
    if (id>=atom->nlocal) 
    {
        state->num_bond[i]=0;
        state->num_angle[i]=0;
        state->num_dihedral[i]=0;
        state->num_improper[i]=0;
        continue;
    }
	
    if(evb_engine->kernel_atom[id]==0) continue;
	
    state->num_bond[i]=num_bond[id];
    state->num_angle[i]=num_angle[id];
    state->num_dihedral[i]=num_dihedral[id];
    state->num_improper[i]=num_improper[id];
    
    for(int j=0;j<num_bond[id]; j++)
    {
      state->bond_type[i][j]=bond_type[id][j];
      state->bond_atom[i][j]=bond_atom[id][j];
    }
    
    for(int j=0;j<num_angle[id]; j++)
    {
      state->angle_type[i][j]=angle_type[id][j];
      state->angle_atom1[i][j]=angle_atom1[id][j];
      state->angle_atom2[i][j]=angle_atom2[id][j];
      state->angle_atom3[i][j]=angle_atom3[id][j];
    }
    
    for(int j=0;j<num_dihedral[id]; j++)
    {
      state->dihedral_type[i][j]=dihedral_type[id][j];
      state->dihedral_atom1[i][j]=dihedral_atom1[id][j];
      state->dihedral_atom2[i][j]=dihedral_atom2[id][j];
      state->dihedral_atom3[i][j]=dihedral_atom3[id][j];
      state->dihedral_atom4[i][j]=dihedral_atom4[id][j];
    }
    
    for(int j=0;j<num_improper[id]; j++)
    {
      state->improper_type[i][j]=improper_type[id][j];
      state->improper_atom1[i][j]=improper_atom1[id][j];
      state->improper_atom2[i][j]=improper_atom2[id][j];
      state->improper_atom3[i][j]=improper_atom3[id][j];
      state->improper_atom4[i][j]=improper_atom4[id][j];
    }
  }
  
  if(evb_kspace) state->qsqsum_cplx = qsqsum;

  state->nbonds     = atom->nbonds;
  state->nangles    = atom->nangles;
  state->ndihedrals = atom->ndihedrals;
  state->nimpropers = atom->nimpropers;
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::load_avec(int index)
{
  STATE_INFO* state = status+index;
 
  for(int i=0; i<natom_cplx; i++)
  {
    int id=cplx_list[i]; 

    //if(atom->tag[id]==254) fprintf(screen,"#### %d %lf\n",index,state->q[i]);
    type[id]=state->type[i];
    mol_type[id]=state->mol_type[i];
    mol_index[id]=state->mol_index[i];
    molecule[id]=state->molecule[i];
    q[id]=state->q[i];

    if (id>=atom->nlocal) 
	{
	  num_bond[id]=0;
      num_angle[id]=0;
      num_dihedral[id]=0;
      num_improper[id]=0;
	  continue;
    }
	
	if(evb_engine->kernel_atom[id]==0) continue;
	
    num_bond[id]=state->num_bond[i];
    num_angle[id]=state->num_angle[i];
    num_dihedral[id]=state->num_dihedral[i];
    num_improper[id]=state->num_improper[i];
    
    for(int j=0;j<num_bond[id]; j++)
    {
      bond_type[id][j]=state->bond_type[i][j];
      bond_atom[id][j]=state->bond_atom[i][j];
    }
    
    for(int j=0;j<num_angle[id]; j++)
    {
      angle_type[id][j]=state->angle_type[i][j];
      angle_atom1[id][j]=state->angle_atom1[i][j];
      angle_atom2[id][j]=state->angle_atom2[i][j];
      angle_atom3[id][j]=state->angle_atom3[i][j];
    }
    
    for(int j=0;j<num_dihedral[id]; j++)
    {
      dihedral_type[id][j]=state->dihedral_type[i][j];
      dihedral_atom1[id][j]=state->dihedral_atom1[i][j];
      dihedral_atom2[id][j]=state->dihedral_atom2[i][j];
      dihedral_atom3[id][j]=state->dihedral_atom3[i][j];
      dihedral_atom4[id][j]=state->dihedral_atom4[i][j];
    }
	
    for(int j=0;j<num_improper[id]; j++)
    {
      improper_type[id][j]=state->improper_type[i][j];
      improper_atom1[id][j]=state->improper_atom1[i][j];
      improper_atom2[id][j]=state->improper_atom2[i][j];
      improper_atom3[id][j]=state->improper_atom3[i][j];
      improper_atom4[id][j]=state->improper_atom4[i][j];
    }
  }
  
  if(evb_kspace) qsqsum = state->qsqsum_cplx;
  
  atom->nbonds      = state->nbonds;
  atom->nangles     = state->nangles;
  atom->ndihedrals  = state->ndihedrals;
  atom->nimpropers  = state->nimpropers;

  current_status = index;
}

/* ---------------------------------------------------------------------- */

#ifdef DLEVB_MODEL_SUPPORT

void EVB_Complex::delete_shell_states(int ishell)
{
  // The following section would be given as input in the evb.cfg file.
  int max_states_shell[MAX_SHELL];
  switch (rc_etype) {
  case(2):                     // Search starts from hydronium
    max_states_shell[1] =  3;
    max_states_shell[2] =  6;
    max_states_shell[3] = 12;
    break;
  case(3):                     // Search starts from amino acid
    max_states_shell[1] =  1;
    max_states_shell[2] =  4;
    max_states_shell[3] =  8;
    break;
  default:                     // MS-EVB2 rules by default
    max_states_shell[1] =  3;
    max_states_shell[2] =  6;
    max_states_shell[3] = 12;
    break;
  }  

  // Find which states are in this shell
  int start = 0;
  for (int i=1; i<nstate; i++) if(start==0 && shell[i] == ishell) start = i;

  int nstate_shell = nstate - start;
  int nExtra = nstate_shell - max_states_shell[ishell];  

  // Remove extra states if present
  if(nExtra>0) { 
    int indx_list[MAX_STATE];
    int included[MAX_STATE];
    int indx;
    for(int i=0; i<MAX_STATE; i++) included[i] = 0; //FALSE
    
    // Sort distances from greatest to smallest
    for (int i=start; i<nstate; i++) {
      double max = 0.0;
      indx = 0;
      for (int j=start; j<nstate; j++) {
	if(!included[j] && (distance[j] > max)) {
	  max = distance[j];
	  indx = j;
	}
      }
      indx_list[i-start] = indx;
      included[indx] = 1;
    }
    
    // Sort state indices for nExtra largest distances
    int indx_list2[MAX_STATE];
    for(int i=0; i<MAX_STATE; i++) included[i] = 0; //FALSE
    int k = 0;
    for (int i=0; i<nExtra; i++) {
      indx = 0;
      int max = 0;
      for (int j=0; j<nExtra; j++) {
	if(!included[j] && (indx_list[j] > max)) {
          indx = j;
          max = indx_list[j];
        }
      }
      indx_list2[k] = indx;
      included[indx] = 1;
      k++;
    }
    
    // Remove extra states
    for (int i=0; i<nExtra; i++) delete_state(indx_list[ indx_list2[i] ]);

  }
}
#endif

/* ---------------------------------------------------------------------- */

int EVB_Complex::ss_do_extra  = 0;
int EVB_Complex::ss_do_refine = 0;

int EVB_Complex::set(char *buf, int *offset, int iword, int ntypes)
{
  iword++;

  if(strcmp(buf+offset[iword],"EXTRA_COUPLINGS") == 0) {
    ss_do_extra = atoi(buf+offset[iword+1]);
    iword+= 2;
  } else return -1;

  if(strcmp(buf+offset[iword],"REFINE") == 0) {
    ss_do_refine = atoi(buf+offset[iword+1]);
    iword+= 2;
  } else return -2;

  return iword;
}

/* ---------------------------------------------------------------------- */

void EVB_Complex::sci_build_state(int index)
{
  if(current_status!=parent_id[index]) load_avec(parent_id[index]);

  /*********************************************************************/
  /******* Deal with the atom_vec             **************************/
  /*********************************************************************/

  int **map = evb_engine->molecule_map;
  int *kernel_atom = evb_engine->kernel_atom;
  
  // update the atom_vec information
  
  // Get general information
  int ma = molecule_A[index];
  int mb = molecule_B[index];
  int re = reaction[index]-1;
  int pa = path[index]-1;
  //fprintf(screen,"%d %d %d %d\n",ma,mb,re,pa);
  int ta = evb_reaction->product_A[re];
  int tb = evb_reaction->product_B[re];
  
  int* count = (evb_reaction->Path)[re][pa].atom_count;
  int* mp = (evb_reaction->Path)[re][pa].moving_part;
  int* fp = (evb_reaction->Path)[re][pa].first_part;
  int* sp = (evb_reaction->Path)[re][pa].second_part;
  
  int m1, m2, t2;
  if(evb_reaction->backward[re]) { m1 = mb; m2 = ma; t2=ta;}
  else { m1=ma; m2=mb; t2=tb; }

  for(int i=0; i<natom_cplx; i++) {    
    int id = cplx_list[i];
    bool change_flag = false;
    
    if(kernel_atom[id]) num_bond[id] = num_angle[id] = num_dihedral[id] = num_improper[id] = 0;
    
    // Deal with moving part
    
    for(int j=0; j<count[0]; j++) {
      int index = mp[j*2];
      int target = mp[j*2+1];
      
      if(m1 == molecule[id] && index == mol_index[id]) {
        molecule[id] = m2;
        evb_reaction->change_atom(id,t2,target);
        change_flag = true;        
      }      
      if(change_flag) break;
    }    
    if(change_flag) continue;
    
    // Deal with rest part
    
    for(int j=0; j<count[1]; j++) {
      int index = fp[j*2];
      int target = fp[j*2+1];
      
      if(ma == molecule[id] && index == mol_index[id]){
	evb_reaction->change_atom(id,ta,target);
	change_flag = true;
      }      
      if(change_flag) break;
    }    
    if(change_flag) continue;
    
    // Deal with new part
    
    for(int j=0; j<count[2]; j++) {
      int index = sp[j*2];
      int target = sp[j*2+1];
      
      if(mb == molecule[id] && index == mol_index[id]) {
        evb_reaction->change_atom(id,tb,target);
        change_flag = true;
      }      
      if(change_flag) break;
    }
  }
    
  update_mol_map();
  update_bond_list();
  
  /*********************************************************************/
  /******* EVB KSpace                   ********************************/
  /*********************************************************************/
  
  int rA = evb_reaction->reactant_A[re];
  int rB = evb_reaction->reactant_B[re];

  if(evb_kspace) {
    double chg;
    double *_qsqsum = evb_type->qsqsum;
    chg = _qsqsum[ta-1]+_qsqsum[tb-1]-_qsqsum[rA-1]-_qsqsum[rB-1];
    qsqsum += chg;
  }

  /***********************************/
  /****** Atoms struct ***************/
  /***********************************/

  int *narray;
  narray = evb_type->nbonds;
  atom->nbonds    += (narray[ta-1] + narray[tb-1] - narray[rA-1] - narray[rB-1]);
  narray = evb_type->nangles;
  atom->nangles   += (narray[ta-1] + narray[tb-1] - narray[rA-1] - narray[rB-1]);
  narray = evb_type->ndihedrals;
  atom->ndihedrals+= (narray[ta-1] + narray[tb-1] - narray[rA-1] - narray[rB-1]);
  narray = evb_type->nimpropers;
  atom->nimpropers+= (narray[ta-1] + narray[tb-1] - narray[rA-1] - narray[rB-1]);

  // set ID
  current_status = index;
}
