/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "universe.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "atom.h"
#include "atom_vec.h"

#include "EVB_engine.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_chain.h"
#include "EVB_reaction.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_Reaction::EVB_Reaction(LAMMPS *lmp, EVB_Engine *engine) : Pointers(lmp), EVB_Pointers(engine)
{
  backward = reactant_A = product_A = reactant_B = product_B = nPath = NULL;
  name = NULL;
  Path = NULL;
}

/* ---------------------------------------------------------------------- */

EVB_Reaction::~EVB_Reaction()
{
  for(int i=0; i<nPair; i++)
  {
    for(int j=0; j<nPath[i]; j++)
      if(Path[i][j].moving_part) delete [] Path[i][j].moving_part;
    
    if(Path[i]) delete [] Path[i];
    memory->sfree(name[i]);
  }
  delete [] Path;
  memory->sfree(name);
  
  memory->sfree(backward);
  memory->sfree(reactant_A);
  memory->sfree(reactant_B);
  memory->sfree(product_A);
  memory->sfree(product_B);
  memory->sfree(nPath);
}

/* ---------------------------------------------------------------------- */

int EVB_Reaction::data_reaction(char* buf, int* offset, int start, int end)
{
    char name_line[1000];
    char errline[255];
    
    FILE * fp = evb_engine->fp_cfg_out;
    if(universe->me == 0) fprintf(fp,"\n\nStarting to process reaction templates.\n");

    if(start==end) error->all(FLERR,"[EVB] Unexpected end of file.");
    
    if(strstr(buf+offset[start],"segment.reaction")==0)
    {
        sprintf(errline,"[EVB] Cannot find key word [segment.reaction] at: \"%s\"",buf+offset[start]);
        error->all(FLERR,errline);
    }
    
    int t,count1=0, count2=0;
    for(t=start+1; t<end; t++)
    {
        if(strstr(buf+offset[t],"reaction.start"))
        {
            if(count1>count2) error->all(FLERR,"[EVB] Expecting key word [reaction.end].");
            count1++;
        }
        
        else if(strstr(buf+offset[t],"reaction.end"))
        {
            count2++;
            if(count2>count1) error->all(FLERR,"[EVB] Unexpected key word [reaction.end].");
        }
    }
    if(count1>count2) error->all(FLERR,"[EVB] Expecting key word [reaction.end].");
    nPair = count1;

    if(universe->me == 0) fprintf(fp,"Allocating memory for %i reaction templates.\n",count1);

    name = (char**) memory->smalloc(nPair*sizeof(char*),"EVB_Reaction:name");
    backward = (int*) memory->srealloc(backward, nPair*sizeof(int), "EVB_Reaction:forward");
    reactant_A = (int*) memory->srealloc(reactant_A, nPair*sizeof(int), "EVB_Reaction:reactant_A");
    reactant_B = (int*) memory->srealloc(reactant_B, nPair*sizeof(int), "EVB_Reaction:reactant_B");
    product_A = (int*) memory->srealloc(product_A, nPair*sizeof(int), "EVB_Reaction:product_A");
    product_B = (int*) memory->srealloc(product_B, nPair*sizeof(int), "EVB_Reaction:product_B");
    nPath = (int*) memory->srealloc(nPath, nPair*sizeof(int), "EVB_Reaction:forward");
    Path = new EVB_Path*[nPair];  

    t = start+1;
    
    for(int i=0; i<nPair; i++)
    {
        sprintf(errline,"[EVB] Wrong format exists after %s.",buf+offset[t]);
        int _end = t;
        while(strstr(buf+offset[_end],"reaction.end")==0) _end++;

        char *pp = strstr(buf+offset[t],"reaction.start");
        if(pp==0)
        {
            sprintf(errline,"[EVB] Expecting [reaction.start] at: \"%s\"",buf+offset[t]);
            error->all(FLERR,errline);
        }
        pp += 15;
        char *ppp=name_line;
        while(*pp && *pp!=']')
        {
            *ppp=*pp;
            pp++;
            ppp++;
        }
        *ppp = 0;
        name[i] = (char*) memory->smalloc(sizeof(char)*(strlen(name_line)+1),"EVB_Reaction:name[i]");
        strcpy(name[i],name_line);    
        t++;
        
	if(universe->me == 0) fprintf(fp,"\n%i: name= %s\n",i,name[i]);

        backward[i] = atoi(buf+offset[t++]);
        
        reactant_A[i] = evb_type->get_type(buf+offset[t++]);
        if(reactant_A[i]==-1)
        {
            sprintf(errline,"[EVB] Undefined molecule_type [%s].", buf+offset[t-1]);
            error->all(FLERR,errline);
        }
        
        product_A[i] = evb_type->get_type(buf+offset[t++]);
        if(product_A[i]==-1)
        {
            sprintf(errline,"[EVB] Undefined molecule_type [%s].", buf+offset[t-1]);
            error->all(FLERR,errline);
        }

        reactant_B[i] = evb_type->get_type(buf+offset[t++]);
        if(reactant_B[i]==-1)
        {
            sprintf(errline,"[EVB] Undefined molecule_type [%s].", buf+offset[t-1]);
            error->all(FLERR,errline);
        }

        product_B[i] = evb_type->get_type(buf+offset[t++]);
        if(product_B[i]==-1)
        {
            sprintf(errline,"[EVB] Undefined molecule_type [%s].", buf+offset[t-1]);
            error->all(FLERR,errline);
        }
        
        nPath[i]   = atoi(buf+offset[t++]);

	if(universe->me == 0) {
	  fprintf(fp,"   Direction of particle migration: %i",backward[i]);
	  if(backward[i]) fprintf(fp," (particles in %s transferring to %s).\n",evb_type->name[reactant_B[i]-1],evb_type->name[reactant_A[i]-1]);
	  else fprintf(fp," (particles in %s transferring to %s).\n",evb_type->name[reactant_A[i]-1],evb_type->name[reactant_B[i]-1]);
	  fprintf(fp,"   Molecule A transformation %s <--> %s.\n",evb_type->name[reactant_A[i]-1],evb_type->name[product_A[i]-1]);
	  fprintf(fp,"   Molecule B transformation %s <--> %s.\n",evb_type->name[reactant_B[i]-1],evb_type->name[product_B[i]-1]);
	  fprintf(fp,"   Number of ways reaction can be attempted: nPath= %i.\n",nPath[i]);
	}

        Path[i] = new EVB_Path[nPath[i]];
        for(int j=0; j<nPath[i]; j++)
        {
            EVB_Path* path = Path[i]+j;
            path->atom_count[0] = atoi(buf+offset[t++]);
            path->atom_count[1] = atoi(buf+offset[t++]);
            path->atom_count[2] = atoi(buf+offset[t++]);
            
            int total_atom = path->atom_count[0] + path->atom_count[1] + path->atom_count[2];
            //fprintf(screen,"total %d\n",total_atom);
            path->moving_part = new int [total_atom*2];
            path->first_part  = path->moving_part + path->atom_count[0]*2;
            path->second_part = path->first_part + path->atom_count[1]*2;
            
            for(int k=0; k<total_atom; k++)
            {
                path->moving_part[k*2] = atoi(buf+offset[t++]);
                path->moving_part[k*2+1] = atoi(buf+offset[t++]);
            }

	    if(universe->me == 0) {
	      fprintf(fp,"\n   Path %i:\n",j);
	      fprintf(fp,"   +++++++++++++++++++++++++++++++\n");

	      fprintf(fp,"   # of particles transferred= %i.\n",path->atom_count[0]);
	      for(int k=0; k<path->atom_count[0]; k++) fprintf(fp,"      %i --> %i\n",path->moving_part[k*2],path->moving_part[k*2+1]);

	      fprintf(fp,"   # of particles remaining in molecule A= %i.\n",path->atom_count[1]);
	      for(int k=0; k<path->atom_count[1]; k++) fprintf(fp,"      %i --> %i\n",path->first_part[k*2],path->first_part[k*2+1]);

	      fprintf(fp,"   # of particles remaining in molecule B= %i.\n",path->atom_count[2]);
	      for(int k=0; k<path->atom_count[2]; k++) fprintf(fp,"      %i --> %i\n",path->second_part[k*2],path->second_part[k*2+1]);
	    }
        }

        if(t!=_end) error->all(FLERR,errline);
        t++;
	
	if(universe->me == 0) fprintf(fp,"\n   Processing reaction template complete.\n");
    }

    if(strstr(buf+offset[t],"segment.end")==0) error->all(FLERR,"[EVB] Expecting [segment.end] for [segment.reaction]");
    t++;

    if(universe->me == 0) {
      fprintf(fp,"\nProcessing ALL reaction templates complete.\n");
      fprintf(fp,"\n==================================\n");
    }
    
    return t;
}

/* ---------------------------------------------------------------------- */

void EVB_Reaction::setup()
{
  _EVB_REFRESH_AVEC_POINTERS;
}

/* ---------------------------------------------------------------------- */

void EVB_Reaction::change_atom(int id, int itype, int index)
{
  //if(itype>2) fprintf(screen,"id: %d itype: %d index: %d\n",atom->tag[id],itype,index);

  int n = evb_type->type_index[itype-1]+index-1;
  mol_type[id] = itype;
  mol_index[id] = index;
  
  q[id] = evb_type->atom_q[n];
  type[id] = evb_type->atom_type[n];
  molecule[id] = atom->molecule[id];
}

/* ---------------------------------------------------------------------- */

int EVB_Reaction::get_reaction(char* s)
{
    int i;
    for(i=0; i<nPair; i++) if(strcmp(name[i],s)==0) break;
    if(i<nPair) return i+1;
    else return -1;
}
