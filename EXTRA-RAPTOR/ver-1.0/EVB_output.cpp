/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "memory.h"
#include "error.h"
#include "update.h"
#include "atom.h"
#include "output.h"

#include "EVB_source.h"
#include "EVB_output.h"
#include "EVB_engine.h"
#include "EVB_complex.h"
#include "EVB_matrix.h"
#include "EVB_matrix_full.h"
#include "EVB_matrix_sci.h"
#include "EVB_chain.h"
#include "EVB_cec.h"
#include "EVB_cec_v2.h"
#include "EVB_timer.h"
#include "integrate.h"
#include "mp_verlet_sci.h"

#include "comm.h"
#include "universe.h"

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

EVB_Output::EVB_Output(LAMMPS *lmp, EVB_Engine *engine, char* fout_name) : Pointers(lmp), EVB_Pointers(engine)
{
  bool open_output = true;
  if(universe->iworld != 0) open_output = false;

  if(open_output) {
    fp = fopen(fout_name,"w");
    if(!fp) error->one(FLERR,"[EVB] out_file is unavaliable when constructing EVB_Engine.");
  }
  
  freq = ifreq = react = bin = forceflush = 0;
  bCenter = bNState = bStates = bEnergy = bEneDecompose = 0;
}

EVB_Output::~EVB_Output()
{
  if(universe->iworld == 0) fclose(fp);
}

int EVB_Output::data_output(char *buf, int *offset, int start, int end)
{
  int t = start;  
  if(t==end || strstr(buf+offset[t],"segment.output")==0)
        error->all(FLERR,"[EVB] Expecting key word [segment.output]");
  t++;

  FILE * fp = evb_engine->fp_cfg_out;
  if(universe->me == 0) fprintf(fp,"\n\nStarting to process OUTPUT.\n");

  freq = atoi(buf+offset[t++]); ifreq = freq-1;
  react = atoi(buf+offset[t++]);
  bin = atoi(buf+offset[t++]);
  forceflush = atoi(buf+offset[t++]);
  
  bCenter = atoi(buf+offset[t++]);
  bNState = atoi(buf+offset[t++]);
  bStates = atoi(buf+offset[t++]);
  bEnergy = atoi(buf+offset[t++]);
  bEneDecompose = atoi(buf+offset[t++]);

  if(t==end || strstr(buf+offset[t],"segment.end")==0)
        error->all(FLERR,"[EVB] Expecting key word [segment.end]");
  t++;

  if(universe->me == 0) {
    fprintf(fp,"Processing OUTPUT complete.\n");
    fprintf(fp,"   Output Frequency:   freq= %i\n",freq);
    fprintf(fp,"   Output if reaction: react= %i\n",react);
    fprintf(fp,"   Binary output:      bin= %i\n",bin);
    fprintf(fp,"   Flush every output: forceflush= %i\n\n",forceflush);
    fprintf(fp,"   Write CEC coordinates: bCenter= %i\n",bCenter);
    fprintf(fp,"   Write # of states:     bNState= %i\n",bNState);
    fprintf(fp,"   Write state info:      bStates= %i\n",bStates);
    fprintf(fp,"   Write total energies:  bEnergy= %i\n",bEnergy);
    fprintf(fp,"   Write state energies:  bEneDecompose= %i\n",bEneDecompose);
    fprintf(fp,"\n==================================\n");
  }
  
  return t;
}

void EVB_Output::print(const char *str)
{
  if(logfile) fprintf(logfile,"%s",str);
}

void EVB_Output::execute()
{  
  int flag = true;
  
  if(freq && (++ifreq==freq)) { flag=false; ifreq=0; }
  else if(freq==0 && update->ntimestep == output->next) { flag=false; }
  else if(react && evb_engine->nreact) { flag=false; }
  if(flag) return;
  
  if(bin) write_bin(); 
  else write_txt();
}

void EVB_Output::write_bin()
{

}

void EVB_Output::write_txt()
{
  TIMER_STAMP(EVB_OUTPUT, write_txt);

    double *arr;
    
    fprintf(fp,"\n******************************************************\n");
    
    /************ Timestep *****************/
    fprintf(fp,"\nTIMESTEP "BIGINT_FORMAT"\n",update->ntimestep);

    /************ NCOMPLEX *****************/
    fprintf(fp,"COMPLEX_COUNT %d\n",evb_engine->ncomplex);
    
    /************ center location *****************/
    if(bCenter)
    {
        fprintf(fp,"REACTION_CENTER_LOCATION [ center_id | molecule_id | location_rank ]\n");
        for(int i=0; i<evb_engine->ncomplex; i++)
            fprintf(fp,"%d   %d   %d\n", i+1, evb_engine->rc_molecule[i],evb_engine->rc_rank[i]);
    }
    
    /************ energy summary *****************/
    if(bEnergy)
    {
      fprintf(fp,"ENERGY_SUMMARY\n");
      if(evb_engine->ncenter>1 || evb_engine->mp_verlet_sci) arr=evb_engine->full_matrix->sci_e_env;
      else arr = evb_engine->full_matrix->e_env;
      fprintf(fp,"ENE_ENVIRONMENT   %16lf\n", arr[EDIAG_POT]);
      
      if(evb_engine->ncomplex==1) fprintf(fp,"ENE_COMPLEX       %16lf\n",evb_engine->full_matrix->ground_state_energy);
      else
      {
        for(int i=0; i<evb_engine->ncomplex; i++)
          fprintf(fp,"ENE_COMPLEX%-4d   %16lf\n",i+1,evb_engine->all_matrix[i]->ground_state_energy);
      
        fprintf(fp,"ENE_COMPLEXES     %16lf\n",evb_engine->cplx_energy);
        fprintf(fp,"ENE_INTER_CPLX    %16lf\n",evb_engine->inter_energy);
      }
      fprintf(fp,"ENE_TOTAL         %16lf\n",evb_engine->energy);
      if(evb_engine->efieldz) fprintf(fp,"ENE_EFIELD_ENV    %16lf\n",evb_engine->efield_energy_env);
    }
    
    /************ environment *****************/
    if(bEneDecompose)
    {
      fprintf(fp,"DECOMPOSED_ENV_ENERGY \n");
      fprintf(fp,"ENVIRONMENT [vdw|coul|bond|angle|dihedral|improper|kspace]\n");
      if(evb_engine->ncenter>1 || evb_engine->mp_verlet_sci) arr = evb_engine->full_matrix->sci_e_env;
      else arr = evb_engine->full_matrix->e_env;
      fprintf(fp," %12lf %12lf %12lf %12lf %12lf %12lf %12lf\n",
                       arr[EDIAG_VDW],
                       arr[EDIAG_COUL],
                       arr[EDIAG_BOND],
                       arr[EDIAG_ANGLE],
                       arr[EDIAG_DIHEDRAL],
                       arr[EDIAG_IMPROPER],
                       arr[EDIAG_KSPACE]);
    }

#ifdef RELAMBDA
    // AWGL : Print out lambda scaling information if on 
    if (evb_engine->lambda_flag) {
      fprintf(fp, "LAMBDA = %f\n", evb_engine->offdiag_lambda);
    }
#endif
    
    /************ loop all_complex *************/
    fprintf(fp,"LOOP_ALL_COMPLEX\n");

    for(int icplx=0; icplx<evb_engine->ncomplex; icplx++)
    {
        fprintf(fp,"START_OF_COMPLEX %d\n",icplx+1);
        
        EVB_Complex* cplx = evb_engine->all_complex[icplx];
        EVB_Matrix* mtx ;
        if(evb_engine->ncomplex==1) mtx = (EVB_Matrix*)(evb_engine->full_matrix);
        else mtx = (EVB_Matrix*)(evb_engine->all_matrix[icplx]);
        
        /************ state search *****************/
        if(bNState)
        {
            fprintf(fp, "STATE_SEARCH\n" );
            
            fprintf(fp, "COMPLEX %d: %d state(s) = 1", icplx+1, cplx->nstate);
            for(int kkk=0; kkk<evb_chain->max_shell; kkk++) fprintf(fp," + %d",cplx->state_per_shell[kkk]);
            fprintf(fp,"\n");
            
            if(bStates)
            {
                fprintf(fp,"STATES [ id | parent | shell | mol_A | mol_B | react | path | extra_cpl ]\n");
                
                for(int j=0; j<cplx->nstate; j++)
                    fprintf(fp,"       %6d %6d %6d %6d %6d %6d %6d %6d\n",j,cplx->parent_id[j],
                            cplx->shell[j],cplx->molecule_A[j],cplx->molecule_B[j],
                            cplx->reaction[j],cplx->path[j],cplx->extra_coupling[j]);
            }
            
        }
        
	/************ Decomposed Energy *****************/
	if(bEneDecompose)
	{
            fprintf(fp,"DECOMPOSED_COMPLEX_ENERGY\n");
            
            fprintf(fp,"DIAGONAL [id|total|vdw|coul|bond|angle|dihedral|improper|kspace|repulsive]\n");            
            for(int i=0; i<cplx->nstate; i++) fprintf(fp," %8d %12lf %12lf %12lf %12lf %12lf %12lf %12lf %12lf %12lf\n", i,
                       mtx->e_diagonal[i][EDIAG_POT],
                       mtx->e_diagonal[i][EDIAG_VDW],
                       mtx->e_diagonal[i][EDIAG_COUL],
                       mtx->e_diagonal[i][EDIAG_BOND],
                       mtx->e_diagonal[i][EDIAG_ANGLE],
                       mtx->e_diagonal[i][EDIAG_DIHEDRAL],
                       mtx->e_diagonal[i][EDIAG_IMPROPER],
                       mtx->e_diagonal[i][EDIAG_KSPACE],
                       mtx->e_repulsive[i]); 
                       
            fprintf(fp,"OFF-DIAGONAL [ id | energy |  A_Rq  | Vij_const |   Vij   ]\n");
            for(int i=0; i<cplx->nstate-1; i++)
                fprintf(fp,"%12d %12lf %12lf %12lf %12lf\n",i+1,mtx->e_offdiag[i][EOFF_ENE],mtx->e_offdiag[i][EOFF_ARQ],mtx->e_offdiag[i][EOFF_VIJ_CONST],mtx->e_offdiag[i][EOFF_VIJ]);
            
            fprintf(fp,"EXTRA-COUPLING %d [ I | J | energy ]\n",cplx->nextra_coupling);
            for(int i=0; i<cplx->nextra_coupling; i++)
                fprintf(fp,"                  %3d  %3d   %lf\n",cplx->extra_i[i],cplx->extra_j[i],mtx->e_extra[i][EOFF_ENE]);
	}
	
	/************ Diagonalization *****************/
	fprintf(fp, "DIAGONALIZATION\n");
	fprintf(fp, "EIGEN_VECTOR\n");
	
	int ground_state = mtx->ground_state;
        for(int i=0; i<cplx->nstate; i++)
            fprintf(fp,"%0.4lE ",cplx->Cs[i]);
        fprintf(fp,"\n");
        
	fprintf(fp, "NEXT_PIVOT_STATE %d\n", mtx->pivot_state);
    
    
        fprintf(fp, "CEC_COORDINATE\n");
  
        fprintf(fp,"%lf %lf %lf\n", cplx->cec->r_cec[0],
                cplx->cec->r_cec[1],  cplx->cec->r_cec[2]);

        fprintf(fp, "CEC_V2_COORDINATE\n");

        fprintf(fp,"%lf %lf %lf\n", cplx->cec_v2->r_cec[0],
                cplx->cec_v2->r_cec[1],  cplx->cec_v2->r_cec[2]);
          
        /*
          for(int j=0; j<cplx->nstate;j++)
          fprintf(fp,"---> %lf %lf %lf\n",cplx->r_coc[j][0], cplx->r_coc[j][1], cplx->r_coc[j][2]);
        */
        
        fprintf(fp,"END_OF_COMPLEX %d\n",icplx+1);
    }
    
    if(forceflush) fflush(fp);   

    TIMER_CLICK(EVB_OUTPUT, write_txt);
}
