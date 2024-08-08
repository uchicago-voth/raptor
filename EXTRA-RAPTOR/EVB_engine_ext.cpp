#include "EVB_engine.h"
#include "EVB_cec_v2.h"
#include "EVB_complex.h"

#include "universe.h"
#include "error.h"
#include "atom.h"
#include "comm.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace LAMMPS_NS;

int EVB_Engine::data_extension(char* buf, int* offset, int start, int end)
{
  if(start==end) return end;

  if(universe->me == 0) fprintf(fp_cfg_out,"\n\nStarting to process extended segments (a.k.a controls for add-on features).\n");

  int iword = start;
  if(iword==end || strstr(buf+offset[iword],"segment.extension")==0)
    error->all(FLERR,"[EVB] Expecting key word [segment.extension]");
  
  iword++;
  
  while(true)
  {
    if(iword==end) error->all(FLERR,"[EVB] Expecting key word [segment.end]");

    if(strstr(buf+offset[iword],"segment.end")) break;

    #define EXTENSION_BEGIN(v) else if(strstr(buf+offset[iword],#v".start")) {
    #define EXTENSION_END(v) if(strstr(buf+offset[iword++],#v".end")==0) \
      error->all(FLERR,"[EVB] Expecting key word ["#v".end]"); \
    }
    
    // ====================================================================  
    
    EXTENSION_BEGIN(cec_v2)
    
      if(universe->me == 0) {
	fprintf(fp_cfg_out,"\n   Starting to process segment: cec_v2.\n");
	fprintf(fp_cfg_out,"   +++++++++++++++++++++++++++++++\n");
      }

      iword = EVB_CEC_V2::set(buf, offset, iword, atom->ntypes);
    
      if(iword==-1) error->all(FLERR,"[EVB] Expecting key word \"COORDINATORS\".");
      if(iword==-2) error->all(FLERR,"[EVB] Expecting key word \"HYDROGENS\".");
      
      if(universe->me == 0) {
	fprintf(fp_cfg_out,"   RSW= %f  DWS= %f\n",EVB_CEC_V2::RSW,EVB_CEC_V2::DSW);
	for(int i=0; i<atom->ntypes; i++) fprintf(fp_cfg_out,"   i= %i  type= %i  weight= %f\n",i,EVB_CEC_V2::type[i],EVB_CEC_V2::weight[i]);
	fprintf(fp_cfg_out,"\n   Processing segment complete.\n");
      }

    EXTENSION_END(cec_v2)
    
    // ====================================================================  

    EXTENSION_BEGIN(state_search)
    
      if(universe->me == 0) {
	fprintf(fp_cfg_out,"\n   Starting to process segment: state_search.\n");
	fprintf(fp_cfg_out,"   +++++++++++++++++++++++++++++++\n");
      }

      iword = EVB_Complex::set(buf, offset, iword, atom->ntypes);
    
      if(iword==-1) error->all(FLERR,"[EVB] Expecting key word \"EXTRA_COUPLINGS\".");
      if(iword==-2) error->all(FLERR,"[EVB] Expecting key word \"REFINE\".");

      bRefineStates   = evb_complex->ss_do_refine;
      bExtraCouplings = evb_complex->ss_do_extra;

      if(universe->me == 0) {
	fprintf(fp_cfg_out,"   *** These overide settings from above ***\n");
	fprintf(fp_cfg_out,"   EVB3 --> EVB2 refinement: bRefineStates= %i\n",bRefineStates);
	fprintf(fp_cfg_out,"   Extra off-diagonal couplings: bExtraCouplings= %i\n",bExtraCouplings);
	fprintf(fp_cfg_out,"\n   Processing segment complete.\n");
      }
      
    EXTENSION_END(state_search)
    
    // ====================================================================  
    
    EXTENSION_BEGIN(screen_hamiltonian)

      if(universe->me == 0) {
	fprintf(fp_cfg_out,"\n   Starting to process segment: screen_hamiltonian.\n");
	fprintf(fp_cfg_out,"   +++++++++++++++++++++++++++++++\n");
      }

      iword++;
      if(strcmp(buf+offset[iword],"MINP") == 0) screen_minP = atof(buf+offset[iword+1]);
      else error->all(FLERR,"[EVB] Expecting key word \"MINP\".");
      iword+=2;

      if(strcmp(buf+offset[iword],"MINP2") == 0) screen_minP2 = atof(buf+offset[iword+1]);
      else error->all(FLERR,"[EVB] Expecting key word \"MINP2\".");
      iword+=2;

      bscreen_hamiltonian = true;

      if(universe->me == 0) {
	fprintf(fp_cfg_out,"   Tolerance for eigenvector coefficient: screen_minP= %e.\n",screen_minP);
	fprintf(fp_cfg_out,"   Tolerance for off-diagonal coefficient pairs: screen_minP= %e.\n",screen_minP2);
	fprintf(fp_cfg_out,"\n   Processing segment complete.\n");
      }

      if(comm->me == 0 && screen) {
	fprintf(screen,"[EVB] Hamiltonian screening activated.\n");
	fprintf(screen,"      Only states with P > %g and off-diagonals with P2 > %g retained.\n",screen_minP,screen_minP2);
      }
      
    EXTENSION_END(screen_hamiltonian)

    /*
    
    EXTENSTION_BEGIN(your_key_word)
    
      Put your input operations here between the two macros.
      Macros take care of the [.start] and [.end] words,
      so you only needs to write things between them.
      Maintain the proper behaviour of the variable "iword" here".     
      Be careful with putting the "iword++" at the end.
       
    EXTENSTION_END(your_key_word)
    
    */
    
    // ====================================================================  
    
    else 
    {
      char errline[200];
      sprintf(errline,"[EVB] Unexpected key word \"%s\".", buf+offset[iword]);
      error->all(FLERR, errline);
    }
  }
  
  if(universe->me == 0) {
    fprintf(fp_cfg_out,"\nProcessing ALL extensions complete.\n");
    fprintf(fp_cfg_out,"\n==================================\n");
  }

  return iword+1;
}
