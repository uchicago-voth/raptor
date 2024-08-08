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
#include "EVB_cec_v2.h"
#include "EVB_matrix.h"
#include "EVB_kspace.h"
#include "EVB_effpair.h"

#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "error.h"

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

int* EVB_CEC_V2::type = NULL;
double* EVB_CEC_V2::weight = NULL;
double EVB_CEC_V2::RSW=1.40;
double EVB_CEC_V2::DSW=0.05;

EVB_CEC_V2::EVB_CEC_V2(LAMMPS *lmp, EVB_Engine *engine, EVB_Complex *complex)
    : Pointers(lmp), EVB_Pointers(engine)
{
  cplx = complex;
} 

int EVB_CEC_V2::set(char* buf, int* offset, int iword, int ntypes)
{
  iword++;

  if(type) { delete [] type; delete [] weight; }
  type = new int [ntypes+1];
  weight = new double [ntypes+1];
  memset(type,0,sizeof(int)*(ntypes+1));
  memset(weight,0,sizeof(double)*(ntypes+1));
  
  if(strcmp(buf+offset[iword],"COEFF")==0)
  {
    RSW = atof(buf+offset[iword+1]);
    DSW = atof(buf+offset[iword+2]);
    iword+=3;
  }
  
  if(strcmp(buf+offset[iword++],"COORDINATORS")==0)
  {
    while(true)
    {
      int itype = atoi(buf+offset[iword]);
      if(itype<1) break;
            
      type[itype] = 1;
      weight[itype] = atof(buf+offset[iword+1]);
      iword+=2;
    }
    
    if(strcmp(buf+offset[iword++],"HYDROGENS")==0)
    {
      while(true)
      {
        int itype = atoi(buf+offset[iword]);
        if(itype<1) break;
            
        type[itype] = 2;
	iword++;
      }
    }
    else return -2;
    
  }
  else return -1;

  return iword;
}

/* ---------------------------------------------------------------------- */

EVB_CEC_V2::~EVB_CEC_V2()
{
  
}

/* ---------------------------------------------------------------------- */

void EVB_CEC_V2::clear()
{
  
}

/* ---------------------------------------------------------------------- */

void EVB_CEC_V2::compute_coc()
{ 

}

/* ---------------------------------------------------------------------- */

void EVB_CEC_V2::compute()
{
  r_cec[0] = r_cec[1] = r_cec[2] = 0.0;
  numX = numH = 0;
  
  if(!type) return;
  
  if(evb_engine->rc_rank[cplx->id-1]==comm->me) {
  
  typedef double matrix[3];
  matrix *devx, *devh;
  
  // find all atoms to be considered
  
  int *map = evb_engine->complex_atom;
  int *atp = atom->type;
  int nlocal = atom->nlocal;
  
  for(int i=0; i<evb_engine->complex_atom_size; i++) if(map[i])
  {
    if(i>=nlocal && atom->map(atom->tag[i])!=i) continue; // skip the atom has been added
    
    if(type[atp[i]]==1) XAtom[numX++] = i;
    else if(type[atp[i]]==2) HAtom[numH++] = i;
  }
  
  // Prepare the derivatives arrays
  
  memset(dev_xatom,0,sizeof(double)*9*numX);
  memset(dev_hatom,0,sizeof(double)*9*numH);
  
  // Set-up PBC
  
  double xorg[3];
  double **x = atom->x;  
  
  int iorg = evb_engine->molecule_map[evb_complex->molecule_B[0]][1];
  xorg[0] = x[iorg][0]; xorg[1] =x[iorg][1]; xorg[2] = x[iorg][2];
  
  // Coordinators
  
  for(int i=0; i<numX; i++)
  {
    int id = XAtom[i];
    double W = weight[atp[id]];
    
    double dx = x[id][0] - xorg[0];
    double dy = x[id][1] - xorg[1];
    double dz = x[id][2] - xorg[2];
    domain->minimum_image(dx,dy,dz);
    
    r_cec[0] -= W * dx;
    r_cec[1] -= W * dy;
    r_cec[2] -= W * dz;
    
    devx = dev_xatom[i];
    devx[0][0] -= W;
    devx[1][1] -= W;
    devx[2][2] -= W;
  }
    
  // hydrogens
  
  for(int i=0; i<numH; i++)
  {    
    int id = HAtom[i];
    double dx = x[id][0] - xorg[0];
    double dy = x[id][1] - xorg[1];
    double dz = x[id][2] - xorg[2];
    domain->minimum_image(dx,dy,dz);
    
    r_cec[0] += dx;
    r_cec[1] += dy;
    r_cec[2] += dz;
    
    devh = dev_hatom[i];
    devh[0][0] += 1.0;
    devh[1][1] += 1.0;
    devh[2][2] += 1.0;
  }

  // cross-terms

  for(int i=0; i<numX; i++) for(int j=0; j<numH; j++)
  {
    int ii = XAtom[i]; int jj = HAtom[j];
    double dx = x[jj][0] - x[ii][0];
    double dy = x[jj][1] - x[ii][1];
    double dz = x[jj][2] - x[ii][2];
    domain->minimum_image(dx,dy,dz);
    
    double dr = sqrt (dx*dx + dy*dy + dz*dz);
    double fsw = 1.0 / ( 1.0 + exp((dr-RSW)/DSW) );
    if(fsw<1E-5) continue;
    
    r_cec[0] -= fsw * dx;
    r_cec[1] -= fsw * dy;
    r_cec[2] -= fsw * dz;
  
    double t = exp(RSW/DSW) + exp(dr/DSW);
    double  dfdr = - exp((dr + RSW)/DSW) / ( DSW * t * t);
    
    double dfdxi = dfdr * dx/dr;
    double dfdyi = dfdr * dy/dr;
    double dfdzi = dfdr * dz/dr;
    
    double ff;
    devx = dev_xatom[i];
    devh = dev_hatom[j];
    
    ff = fsw + dfdxi * dx; devh[0][0] -= ff; devx[0][0] += ff;
    ff = dfdxi * dy; devh[1][0] -= ff; devx[1][0] +=ff;
    ff = dfdxi * dz; devh[2][0] -= ff; devx[2][0] +=ff;

    ff = fsw + dfdyi * dx; devh[0][1] -= ff; devx[0][1] += ff;
    ff = dfdyi * dy; devh[1][1] -= ff; devx[1][1] +=ff;
    ff = dfdyi * dz; devh[2][1] -= ff; devx[2][1] +=ff;
    
    ff = fsw + dfdzi * dx; devh[0][2] -= ff; devx[0][2] += ff;
    ff = dfdzi * dy; devh[1][2] -= ff; devx[1][2] +=ff;
    ff = dfdzi * dz; devh[2][2] -= ff; devx[2][2] +=ff;

  }

  // change back to PBC
  
  r_cec[0] += xorg[0]; r_cec[1] += xorg[1]; r_cec[2] += xorg[2]; 
  
  }
  
  // broadcast
  
  MPI_Bcast(r_cec,3,MPI_DOUBLE,evb_engine->rc_rank[cplx->id-1],world);
}

/* ---------------------------------------------------------------------- */

void EVB_CEC_V2::decompose_force(double* force)
{
  if(!type) return;
  
  typedef double matrix[3];
  double **f = atom->f;
  
  // test
  // for(int i=0; i<atom->nlocal+atom->nghost; i++) f[i][0]=f[i][1]=f[i][2]=0.0;
  
  // Coordinators
  
  for(int i=0; i<numX; i++)
  {
    int id = XAtom[i];    
    double *ff = f[id];
    
    matrix *dev = dev_xatom[i];
    
    ff[0] += dev[0][0] * force[0];
    ff[0] += dev[1][0] * force[1];
    ff[0] += dev[2][0] * force[2];
    
    ff[1] += dev[0][1] * force[0];
    ff[1] += dev[1][1] * force[1];
    ff[1] += dev[2][1] * force[2];
    
    ff[2] += dev[0][2] * force[0];
    ff[2] += dev[1][2] * force[1];
    ff[2] += dev[2][2] * force[2];
  }
  
  // Hydrogens
  
  for(int i=0; i<numH; i++)
  {    
    int id = HAtom[i];    
    double *ff = f[id];
    
    matrix *dev = dev_hatom[i];
    
    ff[0] += dev[0][0] * force[0];
    ff[0] += dev[1][0] * force[1];
    ff[0] += dev[2][0] * force[2];
    
    ff[1] += dev[0][1] * force[0];
    ff[1] += dev[1][1] * force[1];
    ff[1] += dev[2][1] * force[2];
    
    ff[2] += dev[0][2] * force[0];
    ff[2] += dev[1][2] * force[1];
    ff[2] += dev[2][2] * force[2];
  }
  
  // test
  /*
  double fx=0.0, fy=0.0, fz=0.0;
  for(int i=0; i<atom->nlocal+atom->nghost; i++) 
  {
    fx += atom->f[i][0];
    fy += atom->f[i][1];
    fz += atom->f[i][2];
  }
  printf("FORCE %lf=%lf %lf=%lf %lf=%lf\n", fx, force[0], fy, force[1], fz, force[2]);
  */
}

/* ---------------------------------------------------------------------- */
