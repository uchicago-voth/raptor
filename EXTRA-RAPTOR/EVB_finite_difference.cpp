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
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "neighbor.h"
#include "comm.h"
#include "domain.h"
#include "universe.h"

#include "EVB_engine.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_chain.h"
#include "EVB_reaction.h"
#include "EVB_list.h"
#include "EVB_matrix.h"
#include "EVB_matrix_full.h"
#include "EVB_complex.h"
#include "EVB_cec.h"
#include "EVB_rep_vii.h"
#include "EVB_offdiag_pt.h"
#include "EVB_kspace.h"
#include "EVB_output.h"

#define MAXLENGTH 2000

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/*   reset the system since the PBC cell changes                          */
/* ---------------------------------------------------------------------- */

#define CHANGE_SIZE(di,delta,n) domain->x2lamda(n);   \
    domain->boxhi[di]+delta;                                    \
    domain->set_global_box();                                   \
    domain->set_local_box();                                    \
    domain->lamda2x(n);                                    \
    if (force->kspace) force->kspace->setup();

/* ---------------------------------------------------------------------- */

void EVB_Engine::finite_difference_virial()
{
    double half_delta = 1.0;
    double delta = half_delta*2;
    double delta2 = -delta;
    int nall=atom->nlocal+atom->nghost;
    
    double dudc[6];
    
    execute(true); 
    MPI_Allreduce(virial,dudc,6,MPI_DOUBLE,MPI_SUM,world);

    // diagonal element of analytical dU/dC

    dudc[0] /= domain->xprd;
    dudc[1] /= domain->yprd;
    dudc[2] /= domain->zprd;
    
    if (comm->me == 0) {
        fprintf(screen,"\nfinite difference on cell force:");
        fprintf(screen,"\nDrt:    Analytical:     Numerical: \n");
    }
    
    for (int i = 0; i < 3; i++) {
        
        CHANGE_SIZE(i,half_delta,nall);
        execute(true); 
        double pe1 = full_matrix->ground_state_energy+full_matrix->e_env[EDIAG_POT];
        
        CHANGE_SIZE(i,delta2,nall);
        execute(true); 
        double pe2 = full_matrix->ground_state_energy+full_matrix->e_env[EDIAG_POT];
        
        CHANGE_SIZE(i,half_delta,nall);
        double fd = (pe1-pe2)/delta;
        
        if (comm->me == 0)
            fprintf(screen,"%2d      %5.5f       %5.5f \n",i+1,dudc[i],fd);
    }

    exit(0);
}


void EVB_Engine::finite_difference_force()
{

  if (comm->me == 0) {
    fprintf(screen,"*********************************************************\n");
    fprintf(screen,"        This is the finite difference test :\n");
    fprintf(screen,"*********************************************************\n");
  }

  if(comm->nprocs > 1) error->all(FLERR,"FDM=force can only use 1 processor per partition.");

  double half_delta = 0.001;
  double delta = half_delta*2;
  
  for(int i=0; i<atom->nlocal+atom->nghost; i++)
    for(int j=0; j<3; j++)
	  atom->f[i][j]=0.0;
  
  execute(false);
  comm->reverse_comm();

  double **f_save;
  memory->create(f_save,natom,3,"EVB_Engine::finite_difference_force()");

  for(int i=0; i<atom->nlocal; i++) {
    f_save[i][0] = atom->f[i][0];
    f_save[i][1] = atom->f[i][1];
    f_save[i][2] = atom->f[i][2];
  }

  if (comm->me == 0)
      fprintf(screen,"\nfinite difference on atomic force:\n");

  /***********************************************************************
  for(int i=0; i<atom->nlocal; i++)
    fprintf(screen,"atom->f[%d] %lf %lf %lf\n",atom->tag[i], f[i][0], f[i][1], f[i][2]);
  exit(0);
  ************************************************************************/

  double *xx = &(atom->x[0][0]);
  double * ff = &(f_save[0][0]);

  int count[4];
  count[0] = 0;
  count[1] = 0;
  count[2] = 0;
  count[3] = 0;

  double error_sq = 0.0;

  double * error_sq_list;
  memory->create(error_sq_list,ncomplex+1,"EVB_Engine::error_sq_list");
  for(int i=0; i<ncomplex+1; i++) error_sq_list[i] = 0.0;

  double * error_sq_max_list;
  memory->create(error_sq_max_list,ncomplex+1,"EVB_Engine::error_sq_max_list");
  for(int i=0; i<ncomplex+1; i++) error_sq_max_list[i] = 0.0;

  int ii = 0;
  double tstart = MPI_Wtime();
  for (int j=0; j<atom->nlocal; j++) {
    for(int i=0; i<atom->nlocal*3; i++) {
      if(atom->tag[i/3] != j+1) continue;
      
      // central difference
#if 1
      xx[i]+=half_delta; 
      comm->forward_comm();
      execute(false); 
      double pre_ene = energy;
      
      xx[i]-=delta;
      comm->forward_comm();
      execute(false);
      double force = -(pre_ene-energy)/delta;

      xx[i]+=half_delta;
#endif

      // 5-pt stencil
#if 0
      xx[i]+=delta; 
      comm->forward_comm();
      execute(false); 
      const double ene_p2 = -energy;
      
      xx[i]-=half_delta;
      comm->forward_comm();
      execute(false);
      const double ene_p1 = 8.0 * energy;

      xx[i]-=delta;
      comm->forward_comm();
      execute(false);
      const double ene_m1 = -8.0 * energy;
      
      xx[i]-=half_delta;
      comm->forward_comm();
      execute(false);
      const double ene_m2 = energy;

      const double force = -(ene_p2 + ene_p1 + ene_m1 + ene_m2) / (12 * half_delta);

      xx[i]+=delta;
#endif
      
      double error_f = ff[i] - force;
      double abs_error = fabs(error_f);
      if(abs_error >  0.01) count[0]++;
      if(abs_error >  0.1 ) count[1]++;
      if(abs_error >  1.0 ) count[2]++;
      if(abs_error > 10.0 ) count[3]++;

      const int cplx_id = complex_atom[i/3];
      const double esq = error_f * error_f;
      error_sq += esq;
      error_sq_list[cplx_id] += esq;
      if(esq > error_sq_max_list[cplx_id]) error_sq_max_list[cplx_id] = esq;

      // How much time remaining?
      ii++;
      double time_remaining = ((MPI_Wtime() - tstart) / ii) * (3*atom->nlocal - ii);
      
      fprintf(screen,"%3d [%d] type = %d  cplx_id = %d\tanalytic=:%12lf\tnumeric=%12lf\terror=%12lf\ttime=%15lf",
	      j,i%3,atom->type[i/3],cplx_id,ff[i],force,error_f,time_remaining);
      if(abs_error > 1.0) fprintf(stdout,"  *****\n");
      else fprintf(stdout,"\n");
      fprintf(logfile,"%3d [%d] type = %d  cplx_id = %d\tanalytic=:%12lf\tnumeric=%12lf\terror=%12lf\ttime=%15lf\n",
	      j,i%3,atom->type[i/3],cplx_id,ff[i],force,error_f,time_remaining);
    }
  }

  double scale = 100.0 / (atom->nlocal * 3.0);
  fprintf(screen,"\n# of forces larger than 0.01 = %d(%4.2f%%)  0.1 = %d(%4.2f%%)  1.0 = %d(%4.2f%%)  10.0 = %d(%4.2f%%)  error_sq= %f.\n",
	  count[0],double (count[0]) * scale,
	  count[1],double (count[1]) * scale,
	  count[2],double (count[2]) * scale,
	  count[3],double (count[3]) * scale,
	  error_sq);
  
  fprintf(screen,"\nENV:    error_sq= %f  error_sq_max= %f.\n",error_sq_list[0],error_sq_max_list[0]);
  for(int i=1; i<ncomplex+1; i++) fprintf(screen,"CPLX %i: error_sq= %f  error_sq_max= %f.\n",i,error_sq_list[i],error_sq_max_list[i]);

  fprintf(logfile,"\n# of forces larger than 0.01 = %d(%4.2f%%)  0.1 = %d(%4.2f%%)  1.0 = %d(%4.2f%%)  10.0 = %d(%4.2f%%)  error_sq= %f.\n",
	  count[0],double (count[0]) * scale,
	  count[1],double (count[1]) * scale,
	  count[2],double (count[2]) * scale,
	  count[3],double (count[3]) * scale,
	  error_sq);

  fprintf(logfile,"\nENV:    error_sq= %f  error_sq_max= %f.\n",error_sq_list[0],error_sq_max_list[0]);
  for(int i=1; i<ncomplex+1; i++) fprintf(logfile,"CPLX %i: error_sq= %f  error_sq_max= %f.\n",i,error_sq_list[i],error_sq_max_list[i]);

  memory->destroy(f_save);
  memory->destroy(error_sq_list);
  exit(0);
}

/* ---------------------------------------------------------------------- */

void EVB_Engine::finite_difference_cec()
{
  if (comm->me == 0) {
    fprintf(screen,"*********************************************************\n");
    fprintf(screen,"        This is the finite difference test on cec bias:\n");
    fprintf(screen,"*********************************************************\n");
  }

  double half_delta = 0.0001;
  double delta = half_delta*2;
  
  double k = 1000.0;
  
  if (comm->me == 0)
      fprintf(screen,"\nfinite difference on atomic force:\n");
  
  double *xx = &(atom->x[0][0]);

  for(int i=0; i<atom->nlocal*3; i++)
  {
      xx[i]+=half_delta; 
      comm->forward_comm();
      execute(false); 
      
      double cec1[3];
      memcpy(cec1, all_complex[0]->cec->r_cec, sizeof(double)*3);
      double U1=(cec1[0]-2.0)*(cec1[0]-2.0)+(cec1[1]-2.0)*(cec1[1]-2.0)+(cec1[2]-2.0)*(cec1[2]-2.0);
      
      xx[i]-=delta;
      comm->forward_comm();
      execute(false);
      
      double cec2[3];
      memcpy(cec2, all_complex[0]->cec->r_cec, sizeof(double)*3);
      double U2=(cec2[0]-2.0)*(cec2[0]-2.0)+(cec2[1]-2.0)*(cec2[1]-2.0)+(cec2[2]-2.0)*(cec2[2]-2.0);
      
      double dU = 0.5*k*(U2-U1)/delta;
          
      if (i%3==0) fprintf(screen,"[%d]  %lf", atom->tag[i/3],dU);
      if (i%3==1) fprintf(screen,"  %lf",dU);
      if (i%3==2) fprintf(screen,"  %lf\n",dU);
      xx[i]+=half_delta;
  }

  exit(0);
}

/* ---------------------------------------------------------------------- */
void EVB_Engine::finite_difference_amplitude()
{
  if (comm->me == 0) {
    fprintf(screen,"*********************************************************\n");
    fprintf(screen," This is the finite difference test for EVB amplitude:\n");
    fprintf(screen,"*********************************************************\n");
  }

  double half_delta = 0.0002;
  double delta = half_delta*2;
  
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  if(nall==0) return;

  double **x = atom->x;
  int *tag = atom->tag;
  
  int istart = 1967;
  int iend = 1970;

  int ground;
  int pivot;

  for(int itag=istart; itag<=iend; itag++)
  {
      double dev[3];
      
      for(int d=0; d<3; d++)
      {
          double C1,C2;

          for(int i=0; i<nall; i++) if(tag[i]==itag)
              x[i][d]-=half_delta;

          execute(false);

          ground = evb_matrix->ground_state;
          pivot = evb_matrix->pivot_state;
          C1 = evb_matrix->unitary[pivot][ground];
          
          
          for(int i=0; i<nall; i++) if(tag[i]==itag)
              x[i][d]+=delta;
          
          execute(false);

          ground = evb_matrix->ground_state;
          pivot = evb_matrix->pivot_state;
          C2 = evb_matrix->unitary[pivot][ground];

          for(int i=0; i<nall; i++) if(tag[i]==itag)
              x[i][d]-=half_delta;

          dev[d]=(C2-C1)/delta;
      }

      if(comm->me==0) fprintf(screen,"X[%d]=%lf\tY[%d]=%lf\tZ[%d]=%lf\n",
                              itag,dev[0],itag,dev[1],itag,dev[2]);
      
  }
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag::finite_difference_test()
{
  double half_delta = 0.001;
  double delta = half_delta*2;
  
  int nall = atom->nlocal+atom->nghost;
  if(nall==0) return;
  
  double **f;
  memory->create(f,nall,3,"EVB_Rep_Hydrolium::finite_difference_test()");
  memset(&(f[0][0]),0,sizeof(double)*nall*3);
  
  double **lmp_f = atom->f;
  atom->f = f;
  compute(false);
  atom->f = lmp_f;
  
  double *xx = &(atom->x[0][0]);
  double *ff = &(f[0][0]);
  
  fprintf(screen,"*********************************************************\n");
  fprintf(screen," This is the finite difference test for OffDiag terms:\n");
  fprintf(screen,"*********************************************************\n");
  
  for(int i=0; i<nall*3; i++) if(ff[i]!=0)
  {
    xx[i]+=half_delta;
    compute(false);
    double pre_ene = energy;
    xx[i]-=delta;
    compute(false);
    double force = -(pre_ene-energy)/delta;
    fprintf(screen,"%3d[%d]:\tanalytic=:%12lf\tnumeric=%12lf\terror=%12lf\n",atom->tag[i/3],i%3,ff[i],force,ff[i]-force);
    xx[i]+=half_delta;
  }
  
  memory->destroy(f);
  exit(0);
}

/* ----------------------------------------------------------------------*/


void EVB_Repulsive::finite_difference_test()
{
  double half_delta = 0.00001;
  double delta = half_delta*2;
  
  int nall = atom->nlocal+atom->nghost;
  if(nall==0) return;
  
  double **f;
  memory->create(f,nall,3,"EVB_Rep_Hydrolium::finite_difference_test()");
  memset(&(f[0][0]),0,sizeof(double)*nall*3);
  
  double **lmp_f = atom->f;
  atom->f = f;
  compute(false);
  atom->f = lmp_f;
  
  double *xx = &(atom->x[0][0]);
  double *ff = &(f[0][0]);
  
  fprintf(screen,"*********************************************************\n");
  fprintf(screen," This is the finite difference test for repulsive terms:\n");
  fprintf(screen,"*********************************************************\n");
  
  for(int i=0; i<nall*3; i++) if(ff[i]!=0)
  {
    xx[i]+=half_delta;
	compute(false);
	double pre_ene = energy;
	xx[i]-=delta;
	compute(false);
	double force = -(pre_ene-energy)/delta;
	
	fprintf(screen,"%3d[%d]:\tanalytic=:%12lf\tnumeric=%12lf\terror=%12lf\n",i/3,i%3,ff[i],force,ff[i]-force);
	xx[i]+=half_delta;
  }
  
  memory->destroy(f);
  exit(0);
}
