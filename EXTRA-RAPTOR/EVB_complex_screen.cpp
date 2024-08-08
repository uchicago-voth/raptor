/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Steve Tse
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

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

// The elements of del_list are expected to be in ascending order
/* ---------------------------------------------------------------------- */
void EVB_Complex::delete_multiplestates(int del_list[],int del_size)
{
  int is_delete[MAX_STATE];  
  int head = del_list[0];
  int next = head+1;

  int count=0;
  /*
  fprintf(screen, "Before \n ");
  for (int i=0;i<nstate;i++) {
  fprintf(screen, "%d ",molecule_B[i]);  
  count++;
  if(count%10==0)
  fprintf(screen, "\n ");
  }
  fprintf(screen, "\n ");  
  */
  
  memset(is_delete,0, sizeof(int)*nstate);
  for (int i=0;i<del_size;i++) {
    is_delete[del_list[i]] = 1;
    state_per_shell[shell[del_list[i]]-1]--;
  }
  
  for(int i=next; i<nstate; i++) 
    if(is_delete[parent_id[i]]) {
      if(is_delete[i]!=1) {
        is_delete[i] = 1;
        state_per_shell[shell[i]-1]--;
      }
    }   
  
//
//  fprintf(screen, "\nKeeping \n ");
//  for (int i=0;i<nstate;i++) {
//      if(is_delete[i]==0) {
//         fprintf(screen, "%d , %.10f\n ",i,Cs2[i]);                    
//      }
//  }
//  fprintf(screen, "\n ");
  
  
  while (next < nstate) {
    if(!is_delete[next]) {
      
      parent_id[head] = parent_id[next];
      shell[head] = shell[next];
      //  fprintf(screen, "\ndeleting %d \n",molecule_B[head]);                    
      
      
      molecule_A[head] = molecule_A[next];
      molecule_B[head] = molecule_B[next];
      extra_coupling[head]=extra_coupling[next];
      reaction[head] = reaction[next];
      path[head] = path[next];
      
      distance[head] = distance[next];
      
      Cs[head] = Cs[next];
      Cs2[head] = Cs2[next];
      
      for(int i=next+1; i<nstate; i++) 
	if(parent_id[i]==next) parent_id[i]=head;         
      
      head++;
      /*
	count=0;
	fprintf(screen, "Current \n ");
	for (int i=0;i<nstate;i++) {
	fprintf(screen, "%d ",molecule_B[i]);  
	count++;
	if(count%10==0)
	fprintf(screen, "\n ");
	} */       
      
    }      
    next++;
  }
  
  nstate = head;

  // Update # of extra couplings for remaining states
  nextra_coupling = 0;
  for(int i=0; i<nstate; i++) nextra_coupling+= extra_coupling[i];

  /*
    count=0;
    fprintf(screen, "\nafter deletion \n ");
    for (int i=0;i<nstate;i++) {
    fprintf(screen, "%d ",molecule_B[i]);  
    count++;
    if(count%10==0)
    fprintf(screen, "\n ");
    }
    fprintf(screen, "\nFlushing \n ");
  */
}
/*---------------------------------------------------------------------*/
