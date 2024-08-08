/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "EVB_engine.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_complex.h"
#include "EVB_cec.h"
#include "EVB_matrix.h"
#include "EVB_kspace.h"
#include "EVB_effpair.h"

#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "error.h"

// ** AWGL ** //
#if defined (_OPENMP)
#include <omp.h>
#endif
#include "memory.h"

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_CEC::EVB_CEC(LAMMPS *lmp, EVB_Engine *engine, EVB_Complex *complex)
    : Pointers(lmp), EVB_Pointers(engine)
{
  cplx = complex;

  for(int i=0; i<MAX_STATE; i++) 
  {
    qi_coc[i] = new double[evb_engine->atoms_per_molecule];
    id_coc[i] = new int[evb_engine->atoms_per_molecule];
  }
} 

/* ---------------------------------------------------------------------- */

EVB_CEC::~EVB_CEC()
{
  for(int i=0; i<MAX_STATE; i++) delete [] qi_coc[i];
  for(int i=0; i<MAX_STATE; i++) delete [] id_coc[i];
}

/* ---------------------------------------------------------------------- */

void EVB_CEC::clear()
{
  if(evb_engine->rc_rank[cplx->id-1]!=comm->me) return;
  
  /********************************************/
  /*** Alloc the memory for CEC calculation ***/
  /********************************************/
  memset(r_cec,0,sizeof(double)*3);
  memset(r_coc[0],0,sizeof(double)*cplx->nstate*3);
}

/* ---------------------------------------------------------------------- */

void EVB_CEC::compute_coc()
{
  if(evb_engine->rc_rank[cplx->id-1]!=comm->me) return;
  
  /********************************************/
  /*** Assign COC information  ****************/
  /********************************************/
  int **molecule_map = evb_engine->molecule_map;
  double *q = atom->q;
  int i = cplx->current_status;
  
  // Get molecule ID and number of atoms
  int mol_id = cplx->molecule_B[i];
  int mol_type = evb_engine->mol_type[molecule_map[mol_id][1]];
  int *index = evb_type->iCOC[mol_type-1];
  natom_coc[i] = evb_type->nCOC[mol_type-1];
  qsum_coc[i] = 0.0;
	
  // Get atom ID and charge
  for(int j=0; j<natom_coc[i]; j++)
  {
    id_coc[i][j] = molecule_map[mol_id][index[j]];
    qi_coc[i][j] = fabs(q[id_coc[i][j]]);
    qsum_coc[i]+=qi_coc[i][j];
  }
	
  // Reset the qi
  for(int j=0; j<natom_coc[i]; j++) qi_coc[i][j]/=qsum_coc[i];
  
  // Calculate COC
  double **x = atom->x;
  
  ref[0] = x[id_coc[i][0]][0];
  ref[1] = x[id_coc[i][0]][1];
  ref[2] = x[id_coc[i][0]][2];
	
  for(int j=1; j<natom_coc[i]; j++)
  {
      double dr[3];
      VECTOR_SUB(dr,x[id_coc[i][j]],ref);
      VECTOR_PBC(dr);
      VECTOR_SCALE(dr,qi_coc[i][j]);
      VECTOR_ADD(r_coc[i],r_coc[i],dr);
  }
  VECTOR_ADD(r_coc[i],r_coc[i],ref);
  
  //fprintf(screen,"%d %d %lf %lf %lf\n", cplx->id, i, r_coc[i][0],r_coc[i][1],r_coc[i][2]);
}

/* ---------------------------------------------------------------------- */

void EVB_CEC::compute()
{
  r_cec[0]=r_cec[1]=r_cec[2]=0.0;
  if(evb_engine->rc_rank[cplx->id-1]==comm->me) {

  /********************************************/
  /*** Calculate CEC  *************************/
  /********************************************/
  
  double *C2 = cplx->Cs2;
  double dr[3];
  
  ref[0] = r_coc[0][0];
  ref[1] = r_coc[0][1];
  ref[2] = r_coc[0][2];
  
  for(int i=1; i<cplx->nstate; i++)
  {
    VECTOR_SUB(dr,r_coc[i],ref);
    VECTOR_PBC(dr);
    VECTOR_SCALE_ADD(r_cec,dr,C2[i]); 
  }
  
  VECTOR_ADD(r_cec,r_cec,ref);

  /********************************************/
  /**** redo pbc for coc reffered to cec ******/
  /********************************************/

  for(int i=0; i<cplx->nstate; i++)
  {
      VECTOR_SUB(dr,r_coc[i],r_cec);
      VECTOR_PBC(dr);
      VECTOR_ADD(r_coc[i],r_cec,dr);
  }
 
  }
 
  //fprintf(screen,"cec %lf %lf %lf\n",r_cec[0],r_cec[1],r_cec[2]);
  broadcast();
}

/* ---------------------------------------------------------------------- */

void EVB_CEC::broadcast()
{
  MPI_Bcast(r_cec,3,MPI_DOUBLE,evb_engine->rc_rank[cplx->id-1],world);
  MPI_Bcast(r_coc,cplx->nstate*3,MPI_DOUBLE,evb_engine->rc_rank[cplx->id-1],world);
}

/* ---------------------------------------------------------------------- */

void EVB_CEC::decompose_force(double* force)
{
  // fprintf(screen,"CEC BIAS FORCE   %lf %lf %lf\n", force[0],force[1],force[2]);

  double **f = atom->f;
  double* C2 = cplx->Cs2;
  
  EVB_Matrix* matrix;
  if(evb_engine->ncomplex==1) matrix = (EVB_Matrix*)(evb_engine->full_matrix);
  else matrix = (EVB_Matrix*)(evb_engine->all_matrix[cplx->id-1]);

  /******************************************************************/
  /*** Calculate derivitive of [qCOC(i)] ****************************/
  /******************************************************************/
  
  if(evb_engine->rc_rank[cplx->id-1] == comm->me)
  {     
      for(int i=0; i<cplx->nstate; i++)
      {
          for(int j=0; j<natom_coc[i]; j++)
	  {
              int aid = id_coc[i][j];
              for(int k=0; k<3; k++) 
              {
		  f[aid][k]+=force[k]*C2[i]*qi_coc[i][j];
		  //if(atom->tag[aid]==3793) printf("FORCE %lf\n", force[k]*C2[i]*qi_coc[i][j]);
              }
	  } // loop all the atoms in the coc
      } // loop all the coc(s)
  }
  
  /******************************************************************/
  /*** Calculate derivitive of [C(i)^2]  ****************************/
  /******************************************************************/

  partial_C_N3(force); 
}

/* ---------------------------------------------------------------------- */

void EVB_CEC::partial_C_N3(double *force)
{
#if defined (_OPENMP) 
  partial_C_N3_omp(force); // ** AWGL ** //
  return;
#endif
  /******************************************************************/
  /*** JPCB, 112, 2349 Eq 21-23 *************************************/
  /******************************************************************/

  double **f = atom->f;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;

  EVB_Matrix* matrix = evb_matrix; 
  int ground = matrix->ground_state;
  double **C = matrix->unitary;
  double *E = matrix->eigen_value;
  
  double ***diagonal = matrix->f_diagonal;
  double ***off_diagonal = matrix->f_off_diagonal;
  double ***extra_coupl = matrix->f_extra_coupling;
  
  int *parent = cplx->parent_id;
  int *nextra = cplx->extra_coupling;
  int nstate = cplx->nstate;

  double **f_tmp = NULL;
#ifdef STATE_DECOMP
  // For multistate partitioning, need a temporary force holder 
  // to be later reduced over partitions.
  if (evb_engine->flag_mp_state > 2) {
    memory->create(f_tmp, nall, 3, "f_tmp");
    memcpy(&(f_tmp[0][0]), &(f[0][0]), sizeof(double)*nall*3);
    memset(&(f[0][0]), 0.0, sizeof(double)*nall*3);
  }
#endif
  
  double factor, factor1, factor2;
  
  // Pre-calculation
  for(int i=0; i<nstate; i++)
  {
    deltaE[i] = E[ground]-E[i];
    
    x_factor[i] =
      force[0]*r_coc[i][0]*C[i][ground]*2.0 +
      force[1]*r_coc[i][1]*C[i][ground]*2.0 +
      force[2]*r_coc[i][2]*C[i][ground]*2.0;
  }
  
  // Loop all atoms
  
  for(int i=0; i<nall; i++) 
  { 
    //double fff[3]; fff[0]=fff[1]=fff[2]=0.0;
    
    memset(array1, 0, sizeof(double)*nstate*3);
    memset(array2, 0, sizeof(double)*nstate*3);

    /************ Eq. 21 ************************/
    
    for(int m=0; m<nstate; m++)
    {
      factor = C[m][ground];
      
      array1[m][0] -= factor*diagonal[m][i][0];
      array1[m][1] -= factor*diagonal[m][i][1];
      array1[m][2] -= factor*diagonal[m][i][2];

      if(m>0)
      {
        int l = parent[m];

        factor1 = C[m][ground];
        factor2 = C[l][ground];
        
        array1[m][0] -= factor2*off_diagonal[m-1][i][0];
        array1[l][0] -= factor1*off_diagonal[m-1][i][0];
        array1[m][1] -= factor2*off_diagonal[m-1][i][1];
        array1[l][1] -= factor1*off_diagonal[m-1][i][1];
        array1[m][2] -= factor2*off_diagonal[m-1][i][2];
        array1[l][2] -= factor1*off_diagonal[m-1][i][2];
      }
    }
    
    for(int k=0; k<cplx->nextra_coupling; k++)
    {
      int l = cplx->extra_i[k];
      int m = cplx->extra_j[k];

      factor1 = C[m][ground];
      factor2 = C[l][ground];
      
      array1[m][0] -= factor2*off_diagonal[m-1][i][0];
      array1[l][0] -= factor1*off_diagonal[m-1][i][0];
      array1[m][1] -= factor2*off_diagonal[m-1][i][1];
      array1[l][1] -= factor1*off_diagonal[m-1][i][1];
      array1[m][2] -= factor2*off_diagonal[m-1][i][2];
      array1[l][2] -= factor1*off_diagonal[m-1][i][2];
    }

    for(int j=0; j<nstate; j++)
    {
      if(j==ground) continue;
      
      for(int l=0; l<nstate; l++)
      {
        factor = C[l][j];
        
        array2[j][0] += array1[l][0]*factor;
        array2[j][1] += array1[l][1]*factor;
        array2[j][2] += array1[l][2]*factor;
      }
    }
    
    /************ Eq. 22 ************************/

    for(int j=0; j<nstate; j++)
    {
      if(j==ground) continue;

      array1[j][0] = array2[j][0]/deltaE[j];
      array1[j][1] = array2[j][1]/deltaE[j];
      array1[j][2] = array2[j][2]/deltaE[j];
    }

    memset(array2,0,sizeof(double)*nstate*3);
    
    for(int icoc=0; icoc<nstate; icoc++)
    {   
      for(int j=0; j<nstate; j++)
      {
        if(j==ground) continue;

        factor = C[icoc][j];
        
        array2[icoc][0] += array1[j][0]*factor;
        array2[icoc][1] += array1[j][1]*factor;
        array2[icoc][2] += array1[j][2]*factor;
      }
    }

    /************ Eq. 23 ************************/
    
    for(int icoc=0; icoc<nstate; icoc++)
    {
      f[i][0]+=x_factor[icoc]*array2[icoc][0];
      f[i][1]+=x_factor[icoc]*array2[icoc][1];
      f[i][2]+=x_factor[icoc]*array2[icoc][2];
      
      //fff[0]+=x_factor[icoc]*array2[icoc][0];
      //fff[1]+=x_factor[icoc]*array2[icoc][1];
      //fff[2]+=x_factor[icoc]*array2[icoc][2];
    }
    
    //fprintf(screen,"%d %lf %lf %lf\n",atom->tag[i],fff[0],fff[1],fff[2]);
  }

#ifdef STATE_DECOMP
  if (evb_engine->flag_mp_state > 2) {
    // ** For state partitioning, handle the force reduction ** //
    evb_engine->Communicate_Force_Between_Partitions(f);
    // Now add into the usual force array
    for(int i=0; i<nall; ++i) { 
      f[i][0] += f_tmp[i][0];
      f[i][1] += f_tmp[i][1];
      f[i][2] += f_tmp[i][2];
    }
    memory->destroy(f_tmp);
  }
#endif

  /******************** For effective charge ************************/
  if(evb_engine->bEffKSpace)
  {
    int nlocal_cplx = cplx->nlocal_cplx;
    int* cplx_list = cplx->cplx_list;

    double *q = evb_effpair->q;

    GET_OFFDIAG_EXCH(cplx);
    
    for(int i=0; i<nlocal_cplx; i++)
    {
      int id = cplx_list[i];
      memset(q_array1, 0, sizeof(double)*nstate);
      memset(q_array2, 0, sizeof(double)*nstate);

      /************ Eq. 21 ************************/
    
      for(int m=0; m<nstate; m++) q_array1[m] -= C[m][ground]*cplx->status[m].q[i];

      if(!evb_engine->flag_DIAG_QEFF) {
	for(int m=1; m<nstate; m++)
	  {       
	    for(int n=0; n<nexch_off[m-1]; n++) if(iexch_off[m-1][n]==id)
              {
		int l = parent[m];
		
		q_array1[m] -= C[l][ground]*qexch_off[m-1][n];
		q_array1[l] -= C[m][ground]*qexch_off[m-1][n];
		break;
	      }
	  }
	
	for(int k=0; k<cplx->nextra_coupling; k++)
	  for(int n=0; n<nexch_extra[k]; n++) if(iexch_extra[k][n]==id)
            {
	      int l = extra_i[k];
	      int m = extra_j[k];
	      
	      q_array1[m] -= C[l][ground]*qexch_extra[k][n];
	      q_array1[l] -= C[m][ground]*qexch_extra[k][n];
	      break;
	    }	
      }

      for(int j=0; j<nstate; j++)
      {
        if(j==ground) continue;
        for(int l=0; l<nstate; l++) q_array2[j] += q_array1[l]*C[j][j];
      }
    
      /************ Eq. 22 ************************/

      for(int j=0; j<nstate; j++)
      {
        if(j==ground) continue;
        q_array1[j] = q_array2[j]/deltaE[j];
      }

      memset(q_array2,0,sizeof(double)*nstate);
    
      for(int icoc=0; icoc<nstate; icoc++)
      {   
        for(int j=0; j<nstate; j++)
        {
          if(j==ground) continue;
          q_array2[icoc] += array1[j][0]*C[icoc][j];
        }
      }

      /************ Eq. 23 ************************/
    
      for(int icoc=0; icoc<nstate; icoc++)
        q[id]+=x_factor[icoc]*q_array2[icoc];
    }

    double *q_save = atom->q;
    atom->q = evb_effpair->q;
    evb_kspace->compute_eff(false);
    atom->q = q_save;
  }
}

/* ---------------------------------------------------------------------- */

void EVB_CEC::partial_C_N2(double *force)
{
    /******************************************************************/
    /*** JPCB, 112, 2349 Eq 24-26 *************************************/
    /******************************************************************/

    double **f = atom->f;
    int nall = atom->nlocal + atom->nghost;
    EVB_Matrix* matrix = evb_matrix; 
    int ground = matrix->ground_state;
    double **C = matrix->unitary;
    double *E = matrix->eigen_value;
    int *parent = cplx->parent_id;
    double ***diagonal = matrix->f_diagonal;
    double ***off_diagonal = matrix->f_off_diagonal;
    double ***extra_coupl = matrix->f_extra_coupling;
    int nstate = cplx->nstate;
    double factor;

    memset(array1, 0, sizeof(double)*nstate*3);
    memset(array2, 0, sizeof(double)*nstate*3);
    for(int i=0; i<nstate; i++) deltaE[i] = E[ground]-E[i];

    /********** Eq. 24 ***************/
    
    for(int j=0; j<nstate; j++)
    {
        if(j==ground) continue;
        
        for(int i=0; i<nstate; i++)
        {
            factor = C[i][j]*C[i][ground];
             array1[j][0] += 2*force[0]*(factor*r_coc[i][0]);
             array1[j][1] += 2*force[1]*(factor*r_coc[i][1]);
             array1[j][2] += 2*force[2]*(factor*r_coc[i][2]);
        }
    }

    /********** Eq. 25 ***************/
    
    for(int l=0; l<nstate; l++)
    {
        for(int j=0; j<nstate; j++)
        {
            if(j==ground) continue;
            
            factor = C[l][j]/deltaE[j];
             array2[l][0] += factor*array1[j][0];
             array2[l][1] += factor*array1[j][1];
             array2[l][2] += factor*array1[j][2];
        }
        
    }

    /*********** Eq. 26 ******************/
    
    for(int l=0; l<nstate; l++)
    {
        double prefactor[3];
        prefactor[0] = C[l][ground]*array2[l][0];
        prefactor[1] = C[l][ground]*array2[l][1];
        prefactor[2] = C[l][ground]*array2[l][2];
        
        for(int i=0; i<nall; i++)
        {
             f[i][0] -= (diagonal[l][i][0])*prefactor[0];
             f[i][1] -= (diagonal[l][i][1])*prefactor[1];
             f[i][2] -= (diagonal[l][i][2])*prefactor[2];
        }
    }

    for(int k=1; k<nstate; k++)
    {
        int l = k;
        int m = parent[k];
        
        double prefactor[3];
        prefactor[0] = C[m][ground]*array2[l][0]+C[l][ground]*array2[m][0];
        prefactor[1] = C[m][ground]*array2[l][1]+C[l][ground]*array2[m][1];
        prefactor[2] = C[m][ground]*array2[l][2]+C[l][ground]*array2[m][2];
        
        for(int i=0; i<nall; i++)
        {
             f[i][0] -= off_diagonal[k-1][i][0]*prefactor[0];
             f[i][1] -= off_diagonal[k-1][i][1]*prefactor[1];
             f[i][2] -= off_diagonal[k-1][i][2]*prefactor[2];
        }
    }
   
    for(int k=0; k<cplx->nextra_coupling; k++)
    {
        int l = cplx->extra_i[k];
        int m = cplx->extra_j[k];
        
        double prefactor[3];
        prefactor[0] = C[m][ground]*array2[l][0]+C[l][ground]*array2[m][0];
        prefactor[1] = C[m][ground]*array2[l][1]+C[l][ground]*array2[m][1];
        prefactor[2] = C[m][ground]*array2[l][2]+C[l][ground]*array2[m][2];
        
        for(int i=0; i<nall; i++)
        {
             f[i][0] -= extra_coupl[k][i][0]*prefactor[0];
             f[i][1] -= extra_coupl[k][i][1]*prefactor[1];
             f[i][2] -= extra_coupl[k][i][2]*prefactor[2];
        }
    }
}

/* ---------------------------------------------------------------------- */

void EVB_CEC::partial_C_N3_omp(double *force)
{
 
  // ** AWGL : OpenMP threaded version ** //

  /******************************************************************/
  /*** JPCB, 112, 2349 Eq 21-23 *************************************/
  /******************************************************************/

  double **f = atom->f;
  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;

  const EVB_Matrix * matrix = evb_matrix; 
  const int ground = matrix->ground_state;
  const double * const * const C = matrix->unitary;
  const double * const E = matrix->eigen_value;
  
  const double * const * const * const diagonal = matrix->f_diagonal;
  const double * const * const * const off_diagonal = matrix->f_off_diagonal;
  const double * const * const * const extra_coupl = matrix->f_extra_coupling;
  
  const int * const parent = cplx->parent_id;
  const int * const nextra = cplx->extra_coupling;
  const int nstate = cplx->nstate;

#if defined (_OPENMP)
  int nthreads = comm->nthreads;
#else
  int nthreads = 1;
#endif

  double * parray1_omp;
  double * parray2_omp;
  double * inv_deltaE;
  memory->create(parray1_omp, nthreads*nstate*3, "parray1");
  memory->create(parray2_omp, nthreads*nstate*3, "parray2");
  memory->create(inv_deltaE, nstate, "inv_deltaE");

  double **f_tmp = NULL;
#ifdef STATE_DECOMP
  // For multistate partitioning, need a temporary force holder 
  // to be later reduced over partitions.
  if (evb_engine->flag_mp_state > 2) {
    memory->create(f_tmp, nall, 3, "f_tmp");
    memcpy(&(f_tmp[0][0]), &(f[0][0]), sizeof(double)*nall*3);
    memset(&(f[0][0]), 0.0, sizeof(double)*nall*3);
  }
#endif

  int i;
#if defined (_OPENMP)
 #pragma omp parallel default(none)\
 shared(parray1_omp, parray2_omp, inv_deltaE, force, f)\
 private(i)
 {
#endif

#if defined (_OPENMP)
  int tid = omp_get_thread_num();
#else
  int tid = 0;
#endif

  double * parray1 = parray1_omp + 3*nstate*tid;
  double * parray2 = parray2_omp + 3*nstate*tid;
  
  // Pre-calculation
#if defined (_OPENMP)
  #pragma omp for
#endif
  for(i=0; i<nstate; ++i)
  {
    deltaE[i] = E[ground]-E[i];
    if (i != ground) inv_deltaE[i] = 1.0/deltaE[i]; 
    else             inv_deltaE[ground] = 0.0; // need no for ifs below with this 
    x_factor[i] = 2.0 * C[i][ground] * (
      force[0]*r_coc[i][0] +
      force[1]*r_coc[i][1] +
      force[2]*r_coc[i][2]);
  }
  
  // Loop all atoms
#if defined (_OPENMP)
  #pragma omp for
#endif
  for(i=0; i<nall; ++i) 
  { 
    /************ Eq. 21 ************************/
    const double factorc = C[0][ground];
    parray1[0] = -factorc*diagonal[0][i][0];
    parray1[1] = -factorc*diagonal[0][i][1];
    parray1[2] = -factorc*diagonal[0][i][2];
    
    for(int m=1; m<nstate; m++)
    {
      const int l = parent[m];
      const double factor1 = C[m][ground];
      const double factor2 = C[l][ground];
      parray1[3*m  ] = -factor1*diagonal[m][i][0];
      parray1[3*m+1] = -factor1*diagonal[m][i][1];
      parray1[3*m+2] = -factor1*diagonal[m][i][2];
      parray1[3*l  ] -= factor1*off_diagonal[m-1][i][0];
      parray1[3*l+1] -= factor1*off_diagonal[m-1][i][1];
      parray1[3*l+2] -= factor1*off_diagonal[m-1][i][2];
      parray1[3*m  ] -= factor2*off_diagonal[m-1][i][0];
      parray1[3*m+1] -= factor2*off_diagonal[m-1][i][1];
      parray1[3*m+2] -= factor2*off_diagonal[m-1][i][2];
    }
    
    for(int k=0; k<cplx->nextra_coupling; k++)
    {
      const int l = cplx->extra_i[k];
      const int m = cplx->extra_j[k];
      const double factor1 = C[m][ground];
      const double factor2 = C[l][ground];
      parray1[3*l  ] -= factor1*off_diagonal[m-1][i][0];
      parray1[3*l+1] -= factor1*off_diagonal[m-1][i][1];
      parray1[3*l+2] -= factor1*off_diagonal[m-1][i][2];
      parray1[3*m  ] -= factor2*off_diagonal[m-1][i][0];
      parray1[3*m+1] -= factor2*off_diagonal[m-1][i][1];
      parray1[3*m+2] -= factor2*off_diagonal[m-1][i][2];
    }

    for(int j=0; j<nstate; j++)
    {
      parray2[3*j] = parray2[3*j+1] = parray2[3*j+2] = 0.0;
      if(j==ground) continue;
      for(int l=0; l<nstate; l++)
      {
        const double factor = C[l][j];
        parray2[3*j  ] += parray1[3*l  ]*factor;
        parray2[3*j+1] += parray1[3*l+1]*factor;
        parray2[3*j+2] += parray1[3*l+2]*factor;
      }
    }
    
    /************ Eq. 22 ************************/

    for(int j=0; j<nstate; j++)
    {
      parray1[3*j  ] = parray2[3*j  ]*inv_deltaE[j];
      parray1[3*j+1] = parray2[3*j+1]*inv_deltaE[j];
      parray1[3*j+2] = parray2[3*j+2]*inv_deltaE[j];
    }

    for(int j=0; j<nstate; j++)
      parray2[3*j] = parray2[3*j+1] = parray2[3*j+2] = 0.0;
    
    for(int icoc=0; icoc<nstate; icoc++)
    {   
      for(int j=0; j<nstate; j++)
      {
        const double factor = C[icoc][j];
        parray2[3*icoc  ] += parray1[3*j  ]*factor;
        parray2[3*icoc+1] += parray1[3*j+1]*factor;
        parray2[3*icoc+2] += parray1[3*j+2]*factor;
      }
    }

    /************ Eq. 23 ************************/
    
    for(int icoc=0; icoc<nstate; icoc++)
    {
      const double factor = x_factor[icoc];
      f[i][0] += factor*parray2[3*icoc ];
      f[i][1] += factor*parray2[3*icoc+1];
      f[i][2] += factor*parray2[3*icoc+2];
    }
  }

#if defined (_OPENMP)
 }
#endif


#ifdef STATE_DECOMP
  if (evb_engine->flag_mp_state > 2) {
    // ** For state partitioning, handle the force reduction ** //
    evb_engine->Communicate_Force_Between_Partitions(f);
    // Now add into the usual force array
#if defined (_OPENMP)
    #pragma omp parallel for\
    default(none) shared(f, f_tmp) private(i)
#endif
    for(i=0; i<nall; ++i) { 
      f[i][0] += f_tmp[i][0];
      f[i][1] += f_tmp[i][1];
      f[i][2] += f_tmp[i][2];
    }
    memory->destroy(f_tmp);
  }
#endif

  memory->destroy(parray1_omp);
  memory->destroy(parray2_omp);
  memory->destroy(inv_deltaE);


  /******************** For effective charge ************************/
  if(evb_engine->bEffKSpace)
  {
    int nlocal_cplx = cplx->nlocal_cplx;
    int* cplx_list = cplx->cplx_list;

    double *q = evb_effpair->q;

    GET_OFFDIAG_EXCH(cplx);
    
    for(int i=0; i<nlocal_cplx; i++)
    {
      int id = cplx_list[i];
      memset(q_array1, 0, sizeof(double)*nstate);
      memset(q_array2, 0, sizeof(double)*nstate);

      /************ Eq. 21 ************************/
    
      for(int m=0; m<nstate; m++) q_array1[m] -= C[m][ground]*cplx->status[m].q[i];

      if(!evb_engine->flag_DIAG_QEFF) {
	for(int m=1; m<nstate; m++)
	  {       
	    for(int n=0; n<nexch_off[m-1]; n++) if(iexch_off[m-1][n]==id)
              {
		int l = parent[m];
		
		q_array1[m] -= C[l][ground]*qexch_off[m-1][n];
		q_array1[l] -= C[m][ground]*qexch_off[m-1][n];
		break;
	      }
	  }
	
	for(int k=0; k<cplx->nextra_coupling; k++)
	  for(int n=0; n<nexch_extra[k]; n++) if(iexch_extra[k][n]==id)
            {
	      int l = extra_i[k];
	      int m = extra_j[k];
	      
	      q_array1[m] -= C[l][ground]*qexch_extra[k][n];
	      q_array1[l] -= C[m][ground]*qexch_extra[k][n];
	      break;
	    }	
      }

      for(int j=0; j<nstate; j++)
      {
        if(j==ground) continue;
        for(int l=0; l<nstate; l++) q_array2[j] += q_array1[l]*C[j][j];
      }
    
      /************ Eq. 22 ************************/

      for(int j=0; j<nstate; j++)
      {
        if(j==ground) continue;
        q_array1[j] = q_array2[j]/deltaE[j];
      }

      memset(q_array2,0,sizeof(double)*nstate);
    
      for(int icoc=0; icoc<nstate; icoc++)
      {   
        for(int j=0; j<nstate; j++)
        {
          if(j==ground) continue;
          q_array2[icoc] += array1[j][0]*C[icoc][j];
        }
      }

      /************ Eq. 23 ************************/
    
      for(int icoc=0; icoc<nstate; icoc++)
        q[id]+=x_factor[icoc]*q_array2[icoc];
    }

    double *q_save = atom->q;
    atom->q = evb_effpair->q;
    evb_kspace->compute_eff(false);
    atom->q = q_save;
  }

}
