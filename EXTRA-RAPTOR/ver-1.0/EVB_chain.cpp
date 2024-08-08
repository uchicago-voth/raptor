/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "universe.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "atom.h"
#include "atom_vec.h"

#include "EVB_engine.h"
#include "EVB_complex.h"
#include "EVB_source.h"
#include "EVB_chain.h"
#include "EVB_type.h"
#include "EVB_reaction.h"

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_Chain::EVB_Chain(LAMMPS* lmp, EVB_Engine* engine) : Pointers(lmp), EVB_Pointers(engine)
{
  shell_limit = host = target = client = reaction = path = NULL;
  distance_limit = NULL;
  
  type_count = evb_type->type_count; chain_total =0;
  
  index = (int*) memory->smalloc((type_count+1)*sizeof(int),"EVB_Chain:index");
  count = (int*) memory->smalloc((type_count+1)*sizeof(int),"EVB_Chain:count");
  
  max_shell =0;
}

/* ---------------------------------------------------------------------- */

EVB_Chain::~EVB_Chain()
{
  memory->sfree(index);
  memory->sfree(count);
  memory->sfree(host);
  memory->sfree(target);
  memory->sfree(client);
  memory->sfree(shell_limit);
  memory->sfree(distance_limit);
  memory->sfree(reaction);
  memory->sfree(path);
}

/* ---------------------------------------------------------------------- */

void EVB_Chain::grow_chain(int n)
{
  chain_total+=n;
  host = (int*) memory->srealloc(host,chain_total*sizeof(int),"EVB_Chain:index");
  client = (int*) memory->srealloc(client,chain_total*sizeof(int),"EVB_Chain:index");
  target = (int*) memory->srealloc(target,chain_total*sizeof(int),"EVB_Chain:target");
  shell_limit = (int*) memory->srealloc(shell_limit,chain_total*sizeof(int),"EVB_Chain:shell_limit");
  distance_limit = (double*) memory->srealloc(distance_limit,chain_total*sizeof(double),"EVB_Chain:distance_limit");
  reaction = (int*) memory->srealloc(reaction,chain_total*sizeof(int),"EVB_Chain:reaction");
  path = (int*) memory->srealloc(path,chain_total*sizeof(int),"EVB_Chain:path");
}

/* ---------------------------------------------------------------------- */

int EVB_Chain::data_chain(char* buf, int* offset, int start, int end)
{
  int total=0;  
  char errline[255];
  char name_line[1000];
  
  FILE * fp = evb_engine->fp_cfg_out;
  if(universe->me == 0) fprintf(fp,"\n\nStarting to process reaction chains.\n");

  if(strstr(buf+offset[start],"segment.state_search")==0)
  {
      sprintf(errline,"[EVB] Expecting [segment.state_search] at: \"%s\"",buf+offset[start]);
      error->all(FLERR,errline);
  }
  
  int _end;
  for(_end=start+1;_end<end; _end++) if(strstr(buf+offset[_end],"segment.end")) break;
  if(_end==end) error->all(FLERR,"[EVB] Expecting [segment.end] for [segment.state_search].");
  int t = start+1;
  if(t==_end) error->all(FLERR,"[EVB] Wrong format in [segment.state_search].");
  evb_engine->bRefineStates = atoi(buf+offset[t++]);

  if(evb_engine->bRefineStates) evb_engine->bExtraCouplings = 0;
  else evb_engine->bExtraCouplings = 1;

  if(universe->me == 0) {
    fprintf(fp,"   EVB3 --> EVB2 refinement: bRefineStates= %i\n",evb_engine->bRefineStates);
    fprintf(fp,"   Extra off-diagonal couplings: bExtraCouplings= %i\n",evb_engine->bExtraCouplings);
  }
  
  for(int i=0; i<type_count; i++)
  {
      char *pp = strstr(buf+offset[t],"state_search.start");
      if(pp==0)
      {
          sprintf(errline,"[EVB] Expecting key word [state_search.start] at: \"%s\"",buf+offset[t]);
          error->all(FLERR,errline);
      }

      pp += 19;
      char *ppp=name_line;
      while(*pp && *pp!=']')
      {
          *ppp=*pp;
          pp++;
          ppp++;
      }
      *ppp = 0;

      int type = evb_type->get_type(name_line);
      if(type==-1)
      {
          sprintf(errline,"[EVB] Undefined molecule_type [%s].", name_line);
          error->all(FLERR,errline);
      }
      
      t++;

      if(universe->me == 0) fprintf(fp,"   \n\nChain definitions for host molecule: %s.\n",name_line);

      int tail;
      for(tail=t; tail<_end; tail++) if(strstr(buf+offset[tail],"state_search.end")) break;
      if(tail==_end)
      {
           sprintf(errline,"[EVB] lack of definition for [state_seach].");
           error->all(FLERR,errline);
      }

      int m = tail-t;

      if(m%7!=0)
      {
          sprintf(errline,"[EVB] Wrong format at: %s\n",buf+offset[t-1]);
          error->all(FLERR,errline);
      }

      int n = m/7;
      type--;
      index[type]=total;
      count[type]=n;
      total+=n;
      grow_chain(n);
      
      for(int j=index[type]; j<n+index[type]; j++)
      {
          host[j] = atoi(buf+offset[t++]);

          target[j] = evb_type->get_type(buf+offset[t++]);
          if(target[j]==-1)
          {
              sprintf(errline,"[EVB] Undefined molecule_type [%s].", buf+offset[t-1]);
              error->all(FLERR,errline);
          }
          
          client[j] = atoi(buf+offset[t++]);
          shell_limit[j] = atoi(buf+offset[t++]);
          distance_limit[j] = atof(buf+offset[t++]);
          distance_limit[j] = distance_limit[j]*distance_limit[j];    

          reaction[j] = evb_reaction->get_reaction(buf+offset[t++]);
          if(reaction[j]==-1)
          {
              sprintf(errline,"[EVB] Undefined reaction_type [%s].", buf+offset[t-1]);
              error->all(FLERR,errline);
          }
          
          path[j] = atoi(buf+offset[t++]);
          
          if (shell_limit[j]>max_shell) max_shell = shell_limit[j];
	  
	  if(universe->me == 0) {
	    fprintf(fp,"\n   Chain %i:\n",j-index[type]);
	    fprintf(fp,"   +++++++++++++++++++++++++++++++\n");
	    fprintf(fp,"   Index of atom searching for reactant: host= %i.\n",host[j]);
	    fprintf(fp,"   Target molecule: target= %s.\n",evb_type->name[target[j]-1]);
	    fprintf(fp,"   Index of atom in target molecule: client= %i.\n",client[j]);
	    fprintf(fp,"   Maximum number of reaction hops attempted: shell_limit= %i.\n",shell_limit[j]);
	    fprintf(fp,"   Maximum allowed distance to target atom: distance_limit= %f.\n",sqrt(distance_limit[j]));
	    fprintf(fp,"   Reaction template used to update molecules: reaction= %i.\n",reaction[j]);
	    fprintf(fp,"   Reaction path used to update particles: path= %i.\n",path[j]);
	  }
      }

      if(universe->me == 0) fprintf(fp,"\n   Processing chain definitions complete.\n");
  
      t++;
  }
  
  /*****************************************
  fprintf(screen,"\n");
  for(int i=0;i<chain_total; i++)
    fprintf(screen,"%d %d %d %d %d %d\n",host[i],target[i],client[i], shell_limit[i], reaction[i],path[i]);
  exit(0);
  /*****************************************/

  if(strstr(buf+offset[t],"segment.end")==0)
  {
      sprintf(errline,"[EVB] Expecting [segment.end] at: \"%s\"",buf+offset[t]);
      error->all(FLERR,errline);
  }
  t++;
  
  if(universe->me == 0) {
    fprintf(fp,"\nProcessing ALL chain definitions complete.\n");
    fprintf(fp,"\n==================================\n");
  }

  return t;
}

/* ---------------------------------------------------------------------- */
