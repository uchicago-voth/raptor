/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Wim R. Cardoen and Yuxing Peng
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
#include "neigh_list.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "comm.h"

#include "EVB_engine.h"
#include "EVB_source.h"
#include "EVB_type.h"
#include "EVB_chain.h"
#include "EVB_complex.h"
#include "EVB_reaction.h"
#include "EVB_list.h"
#include "EVB_matrix_full.h"
#include "EVB_repul.h"
#include "EVB_offdiag.h"
#include "EVB_kspace.h"
#include "mp_verlet.h"

#define ROTATE(a,i,j,k,l) g=a[i][j];h=a[k][l];a[i][j]=g-s*(h+g*tau);\
    a[k][l]=h+s*(g-h*tau);


/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

EVB_Matrix::EVB_Matrix(LAMMPS *lmp, EVB_Engine *engine) : Pointers(lmp), EVB_Pointers(engine)
{
  max_state = max_atom = max_extra = 1;
  natom = nstate = nextra = 0;
  size_e = size_ediag = 0;

  f_env=NULL;
  f_diagonal = f_off_diagonal = f_extra_coupling = NULL;
  
  e_env = energy;
  e_diag = energy+EDIAG_NITEM;
  
  memory->create(hamilton,MAX_STATE,MAX_STATE, "EVB_Matrix:hamilton");
  memory->create(unitary ,MAX_STATE,MAX_STATE,"EVB_Matrix:unitary"); 
}

/* ---------------------------------------------------------------------- */

EVB_Matrix::~EVB_Matrix()
{
  memory->destroy(f_env);
  memory->destroy(f_diagonal);
  memory->destroy(f_off_diagonal);
  memory->destroy(f_extra_coupling);

  memory->destroy(hamilton);
  memory->destroy(unitary);
}

/* ---------------------------------------------------------------------- */

void EVB_Matrix::save_ev_diag(int index, bool vflag)
{
  double* etarget, *vtarget;
  
  if(index==-1) {
    etarget = e_env;
    if(vflag) vtarget = v_env;
  } else {
    etarget = e_diagonal[index];
    if(vflag) vtarget = (&v_diagonal[0][0])+index*6;
  }
  
  double ene_rep = 0.0;
  if(evb_repulsive) ene_rep = e_repulsive[index] = evb_repulsive->energy;
  
  if(force->pair) etarget[EDIAG_VDW] = force->pair->eng_vdwl;
  if(force->pair) etarget[EDIAG_COUL] = force->pair->eng_coul;
  if(force->bond) etarget[EDIAG_BOND] = force->bond->energy;
  if(force->angle) etarget[EDIAG_ANGLE] = force->angle->energy;
  if(force->dihedral) etarget[EDIAG_DIHEDRAL] = force->dihedral->energy;
  if(force->improper) etarget[EDIAG_IMPROPER] = force->improper->energy;  
  if(evb_kspace) etarget[EDIAG_KSPACE] = force->kspace->energy;
  
  etarget[EDIAG_POT] = etarget[EDIAG_VDW]
    + etarget[EDIAG_COUL]
    + etarget[EDIAG_BOND]
    + etarget[EDIAG_ANGLE]
    + etarget[EDIAG_DIHEDRAL]
    + etarget[EDIAG_IMPROPER]
    + etarget[EDIAG_KSPACE]
    + ene_rep;
  
  if(vflag) for (int i = 0; i < 6; i++) {    
      vtarget[i] = 0.0;
      if (force->pair) vtarget[i] += force->pair->virial[i];
      if (force->bond) vtarget[i] += force->bond->virial[i];
      if (force->angle) vtarget[i] += force->angle->virial[i];
      if (force->dihedral) vtarget[i] += force->dihedral->virial[i];
      if (force->improper) vtarget[i] += force->improper->virial[i];
      if (evb_repulsive) vtarget[i] += evb_repulsive->virial[i];
      
      // devide kspace virial by the number of process for the global sum later
      
      if (force->kspace) vtarget[i] += force->kspace->virial[i] / comm->nprocs;
    }
}

/* ---------------------------------------------------------------------- */

void EVB_Matrix::save_ev_offdiag(bool is_extra, int index, bool vflag)
{
  double *eoff, *voff;
    
  if(is_extra) {
    eoff = (&e_extra[0][0])+index*EOFF_NITEM;
    if(vflag) voff = (&v_extra[0][0])+index*6;
  } else {
    eoff = (&e_offdiag[0][0])+index*EOFF_NITEM;
    if(vflag) voff = (&v_offdiag[0][0])+index*6;
  }
  
  eoff[EOFF_ENE] = evb_offdiag->energy;
  eoff[EOFF_ARQ] = evb_offdiag->A_Rq;
  eoff[EOFF_VIJ] = evb_offdiag->Vij;
  eoff[EOFF_VIJ_CONST] = evb_offdiag->Vij_const;
  
  if(vflag) for(int i=0; i<6; i++) voff[i] = evb_offdiag->virial[i];
}

/* ---------------------------------------------------------------------- */

void EVB_Matrix::total_energy()
{
  MPI_Allreduce(energy,energy_allreduce,size_e,MPI_DOUBLE,MPI_SUM,world);
  memcpy(energy,energy_allreduce,sizeof(double)*size_e);
  
  if(evb_engine->mp_verlet) {
    double ek[MAX_STATE*2+1];
    int n=0;
    
    if(evb_engine->mp_verlet->is_master==0) {      
      ek[n++] = e_env[EDIAG_KSPACE];
      for(int i=0; i<evb_complex->nstate; i++) ek[n++] = e_diagonal[i][EDIAG_KSPACE];
      for(int i=0; i<evb_complex->nstate-1; i++) ek[n++] = e_offdiag[i][EOFF_VIJ];
      for(int i=0; i<evb_complex->nextra_coupling; i++) ek[n++] = e_extra[i][EOFF_VIJ];
    } else { 
      n = evb_complex->nstate * 2 + evb_complex->nextra_coupling; 
    }
    
    MPI_Bcast(ek, n, MPI_DOUBLE, 0, evb_engine->mp_verlet->block);
    
    if(evb_engine->mp_verlet->is_master==1) {
      n=0;
      e_env[EDIAG_KSPACE] = ek[n]; e_env[EDIAG_POT] += ek[n++];
      
      for(int i=0; i<evb_complex->nstate; i++) {
        e_diagonal[i][EDIAG_KSPACE] = ek[n];
        e_diagonal[i][EDIAG_POT] += ek[n++];
      }
      
      for(int i=0; i<evb_complex->nstate-1; i++) {
        e_offdiag[i][EOFF_VIJ_LONG] = ek[n++] - e_offdiag[i][EOFF_VIJ_CONST];
        e_offdiag[i][EOFF_VIJ] += e_offdiag[i][EOFF_VIJ_LONG];
        e_offdiag[i][EOFF_ENE] = e_offdiag[i][EOFF_ARQ] * e_offdiag[i][EOFF_VIJ];
      }
      
      for(int i=0; i<evb_complex->nextra_coupling; i++) {
        e_extra[i][EOFF_VIJ_LONG] = ek[n++] - e_extra[i][EOFF_VIJ_CONST];
        e_extra[i][EOFF_VIJ] += e_extra[i][EOFF_VIJ_LONG];
        e_extra[i][EOFF_ENE] = e_extra[i][EOFF_ARQ] * e_extra[i][EOFF_VIJ];
      }
    }
    
    // A_Rq
    
    n = 0;
    
    if(evb_engine->mp_verlet->rank_block==1) {      
      for(int i=0; i<evb_complex->nstate-1; i++) ek[n++] = e_offdiag[i][EOFF_ARQ];
      for(int i=0; i<evb_complex->nextra_coupling; i++) ek[n++] = e_extra[i][EOFF_ARQ];
      MPI_Send(ek, n, MPI_DOUBLE, 0, 0, evb_engine->mp_verlet->block);
    } else if(evb_engine->mp_verlet->rank_block==0) {
      n = evb_complex->nstate + evb_complex->nextra_coupling - 1;
      MPI_Status mpi_status;
      MPI_Recv(ek, n, MPI_DOUBLE, 1, 0, evb_engine->mp_verlet->block, &mpi_status);
      n = 0;
      for(int i=0; i<evb_complex->nstate-1; i++) e_offdiag[i][EOFF_ARQ] = ek[n++];
      for(int i=0; i<evb_complex->nextra_coupling; i++) e_extra[i][EOFF_ARQ] = ek[n++];
    }
    
    // Three body force from off-diagonals
    
    if(evb_engine->mp_verlet->is_master==1) {
      double **f_save = atom->f;
      int iextra = 0;
      
      for(int i=1; i<evb_complex->nstate; i++) {
        EVB_OffDiag* evb_offdiag = evb_engine->all_offdiag[evb_complex->reaction[i]-1];
        evb_offdiag->index = ndx_offdiag+10*(i-1);
        evb_offdiag->Vij = e_offdiag[i-1][EOFF_VIJ_LONG];
        atom->f = f_off_diagonal[i-1];
        evb_offdiag->mp_post_compute(1);
        
        for(int j=0; j<evb_complex->extra_coupling[i]; j++) {
          evb_offdiag->index = ndx_extra+iextra*10;
          evb_offdiag->Vij = e_extra[i-1][EOFF_VIJ_LONG];
          atom->f = f_extra_coupling[iextra];
          evb_offdiag->mp_post_compute(1);
          iextra++;
        }
      }
      
      atom->f = f_save;
    }
  }
}

/* ---------------------------------------------------------------------- */

void EVB_Matrix::diagonalize()
{
  nstate = evb_complex->nstate;
  nextra = evb_complex->nextra_coupling;
  int *parent_id = evb_complex->parent_id;
  double *Cs = evb_complex->Cs;
  double *Cs2 = evb_complex->Cs2;

  for(int i=0; i<nstate; i++) for(int j=0; j<nstate; j++) hamilton[i][j]=0.0;

  hamilton[0][0]=e_diagonal[0][EDIAG_POT];
  
  for(int i=1; i<nstate; i++) {
    hamilton[i][i] = e_diagonal[i][EDIAG_POT];
    hamilton[parent_id[i]][i] = e_offdiag[i-1][EOFF_ENE];
    
    // For a symmetric matrix, we don't need another triangle half  
    // hamilton[i][parent_id[i]] = e_offdiag[i-1][EOFF_ENE];
  }
  
  for(int i=0; i<nextra; i++)
    hamilton[evb_complex->extra_i[i]][evb_complex->extra_j[i]]=e_extra[i][EOFF_ENE];
  
/* Output the Matrix

  if(comm->me==0)
  {
    fprintf(screen,"The Harmiltonian Matrix:\n");
    for(int i=0; i<nstate; i++)
    {
      for(int j=0; j<nstate; j++)  fprintf(screen,"%8.2lf ",hamilton[i][j]);
	  fprintf(screen,"\n");
    }
  }
*/ 
  
  if(nstate>1) jacobi(hamilton,nstate,eigen_value,unitary,&num_rot);
  else {
    eigen_value[0]=hamilton[0][0];
    unitary[0][0]=1.0;
  }
  
  ground_state_energy=eigen_value[0];
  ground_state=0;
  
  for(int i=1; i<nstate; i++) if(eigen_value[i]<ground_state_energy) {
      ground_state_energy = eigen_value[i];
      ground_state = i;
    }
  
  double max_c = fabs(unitary[0][ground_state]);
  pivot_state = 0;
  
  for(int i=1; i<nstate; i++) if(fabs(unitary[i][ground_state])>max_c) {
      max_c = fabs(unitary[i][ground_state]);
      pivot_state = i;
    }  
  
  for(int i=0; i<nstate; i++) {
    Cs[i]= unitary[i][ground_state];
    Cs2[i] = Cs[i]*Cs[i];
  }
  
  if(evb_engine->mp_verlet) {
    if(evb_engine->mp_verlet->rank_block==1) {
      MPI_Send(Cs, nstate, MPI_DOUBLE, 0, 0, evb_engine->mp_verlet->block);
      MPI_Send(Cs2, nstate, MPI_DOUBLE, 0, 0, evb_engine->mp_verlet->block);
      MPI_Send(&pivot_state, 1, MPI_INT, 0, 0, evb_engine->mp_verlet->block);
    } else if(evb_engine->mp_verlet->rank_block==0) {
      MPI_Status mpi_status;
      MPI_Recv(Cs, nstate, MPI_DOUBLE, 1, 0, evb_engine->mp_verlet->block, &mpi_status);
      MPI_Recv(Cs2, nstate, MPI_DOUBLE, 1, 0, evb_engine->mp_verlet->block, &mpi_status);
      MPI_Recv(&pivot_state, 1, MPI_INT, 1, 0, evb_engine->mp_verlet->block, &mpi_status);
    }
  }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   The JACOBI diagonalization is written by Wim R. Cardoen 08/20/2009
------------------------------------------------------------------------- */

void EVB_Matrix::jacobi(double **a, int n, double *d, double **v, int *nrot)
{
   int j,iq,ip,i;
   double tresh,theta,tau,t,sm,s,h,g,c,*b,*z;

   b=(double *)malloc(sizeof(double)*n);
   z=(double *)malloc(sizeof(double)*n); 
   if((b==NULL) || (v==NULL))
   {  
      fprintf(stderr,"jacobi: Allocation error ...\n");
      fprintf(stderr,"...now exiting to system...\n");
      exit(-1);
   }

   for(ip=0;ip<n;ip++)        // Initialize to the identity matrix.
   {  
       for(iq=0;iq<n;iq++) 
           v[ip][iq]=0.0;
       v[ip][ip]=1.0;
   }

   for(ip=0;ip<n;ip++)        // Initialize b and d to the diagonal of a 
   { 
       b[ip]=d[ip]=a[ip][ip]; 
       z[ip]=0.0; 
   }
   *nrot=0;

   for(i=1;i<=50;i++) 
   {
       sm=0.0;
       for(ip=0;ip<n-1;ip++)           // Sum off-diagonal elements. 
       {
           for(iq=ip+1;iq<n;iq++)
               sm += fabs(a[ip][iq]);
       }
       if(sm==0.0) 
       {                               // The normal return, which relies on quadratic convergence to machine underflow.
          free(z);
          free(b);
          return;
       }
       if(i<4)
          tresh=0.2*sm/(n*n);         // ...on the first three sweeps.
       else
          tresh=0.0;                  // ...thereafter.


       for(ip=0;ip<n-1;ip++) 
       {
           for(iq=ip+1;iq<n;iq++) 
           {
               g=100.0*fabs(a[ip][iq]);
               // After four sweeps, skip the rotation if the oðiagonal element is small.
               if(i > 4 && (double)(fabs(d[ip])+g) == (double)fabs(d[ip]) && (double)(fabs(d[iq])+g) == (double)fabs(d[iq]))
                  a[ip][iq]=0.0;

               else if(fabs(a[ip][iq]) > tresh) 
               {
                  h=d[iq]-d[ip];
                  if((double)(fabs(h)+g) == (double)fabs(h))
                     t=(a[ip][iq])/h;          // t = 1/(2\theta)
                  else 
                  {
                     theta=0.5*h/(a[ip][iq]);  
                     t=1.0/(fabs(theta)+sqrt(1.0+theta*theta));
                     if(theta < 0.0) t = -t;
                  }
                  c=1.0/sqrt(1+t*t);
                  s=t*c;
                  tau=s/(1.0+c);
                  h=t*a[ip][iq];
                  z[ip] -= h;
                  z[iq] += h;
                  d[ip] -= h;
                  d[iq] += h;
                  a[ip][iq]=0.0;

      	          for(j=0;j<=ip-1;j++)        
                  {                       // Case of rotations 1 <= j < p.
                      ROTATE(a,j,ip,j,iq)
                  }

                  for(j=ip+1;j<=iq-1;j++) 
                  {                       // Case of rotations p < j < q.
                      ROTATE(a,ip,j,j,iq)
                  }

	          for(j=iq+1;j<n;j++) 
                  {                       // Case of rotations q < j n.
                      ROTATE(a,ip,j,iq,j)
                  }

                  for(j=0;j<n;j++) 
                  {
                      ROTATE(v,j,ip,j,iq)
                  }
                  ++(*nrot);
              }   
           }
       }
       for(ip=0;ip<n;ip++) 
       {
           b[ip] += z[ip];
           d[ip]=b[ip];   // Update d with the sum of tapq,
           z[ip]=0.0;     // and reinitialize z.
       }
   }
   
   for(int i=0; i<n; i++)
   {
     for(int j=0; j<n; j++) fprintf(screen,"%lf ",a[i][j]);
     fprintf(screen,"\n");
   }
   
   fprintf(stderr,"Too many iterations in routine jacobi ...\n");
   fprintf(stderr,"...now exiting to system...\n");
   exit(-1);
}
