/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "lmptype.h"
#include "mpi.h"
#include "string.h"
#include "fix_evb.h"
#include "EVB_write_top.h"
#include "EVB_engine.h"
#include "EVB_type.h"
#include "atom.h"
#include "atom_vec.h"
#include "update.h"
#include "modify.h"
#include "domain.h"
#include "universe.h"
#include "comm.h"
#include "output.h"
#include "thermo.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

enum{IGNORE,WARN,ERROR};                    // same as thermo.cpp

/* ---------------------------------------------------------------------- */

EVBWriteTop::EVBWriteTop(LAMMPS *lmp) : Pointers(lmp)
{
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  fix_evb = NULL;
}

/* ----------------------------------------------------------------------
   called as write_data command in input script
------------------------------------------------------------------------- */

void EVBWriteTop::command(int narg, char **arg)
{
  if (domain->box_exist == 0)
    error->all(FLERR,"Write_top command before simulation box is defined");

  if (narg != 1) error->all(FLERR,"Illegal write_top command");

  // Find evb fix
  int ifix = -1;
  for(ifix=0; ifix<modify->nfix; ifix++) 
    if(strcmp(modify->fix[ifix]->style,"evb") == 0) break;
  if(ifix < 0) error->all(FLERR,"EVB fix must be defined to use write_top command.");
  fix_evb = (FixEVB*) modify->fix[ifix];

  // if filename contains a "*", replace with current timestep

  char *ptr;
  int n = strlen(arg[0]) + 16;
  char *file = new char[n];

  if (ptr = strchr(arg[0],'*')) {
    *ptr = '\0';
    sprintf(file,"%s" BIGINT_FORMAT "%s",arg[0],update->ntimestep,ptr+1);
  } else strcpy(file,arg[0]);

  // init entire system since comm->exchange is done
  // comm::init needs neighbor::init needs pair::init needs kspace::init, etc

  if (comm->me == 0 && screen) fprintf(screen,"System init for write_top ...\n");
  lmp->init();

  // move atoms to new processors before writing file
  // do setup_pre_exchange to force update of per-atom info if needed
  // enforce PBC in case atoms are outside box
  // call borders() to rebuild atom map since exchange() destroys map

  modify->setup_pre_exchange();
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  domain->reset_box();
  comm->setup();
  comm->exchange();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);

  write(file);
  delete [] file;
}

/* ----------------------------------------------------------------------
   called from command()
   later might let it be directly called within run/minimize loop
------------------------------------------------------------------------- */

void EVBWriteTop::write(char *file)
{
  // special case where reneighboring is not done in integrator
  //   on timestep data file is written (due to build_once being set)
  // if box is changing, must be reset, else data file will have
  //   wrong box size and atoms will be lost when data file is read
  // other calls to pbc and domain and comm are not made,
  //   b/c they only make sense if reneighboring is actually performed

  //if (neighbor->build_once) domain->reset_box();

  // natoms = sum of nlocal = value to write into data file
  // if unequal and thermo lostflag is "error", don't write data file

  bigint nblocal = atom->nlocal;
  bigint natoms;
  MPI_Allreduce(&nblocal,&natoms,1,MPI_LMP_BIGINT,MPI_SUM,world);
  // if (natoms != atom->natoms && output->thermo->lostflag == ERROR)
  //   error->all(FLERR,"Atom count is inconsistent, cannot write top file");
  if (natoms != atom->natoms) error->all(FLERR,"Atom count is inconsistent, cannot write top file");

  // open top file

  if (me == 0) {
    fp = fopen(file,"w");
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open top file %s",file);
      error->one(FLERR,str);
    }
  }

  // per atom info

  if (natoms) evb_top();

  // close data file

  if (me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   write out EVB topology to top file
------------------------------------------------------------------------- */

void EVBWriteTop::evb_top()
{
  // communication buffer for all my Atom info
  // maxrow = largest buffer needed by any proc

  int ncol = 3;

  int sendrow = atom->nlocal;
  int maxrow;
  MPI_Allreduce(&sendrow,&maxrow,1,MPI_INT,MPI_MAX,world);

  double *buf;
  if (me == 0) memory->create(buf,maxrow*ncol,"evb_write_top:buf");
  else memory->create(buf,sendrow*ncol,"evb_write_top:buf");

  // pack my atom data into buf

  int n = 0;
  for(int i=0; i<atom->nlocal; i++) {
    buf[n++] = static_cast<double>(atom->tag[i]);
    n+= fix_evb->pack_exchange(i, &buf[n]);
  }

  // write one chunk of atoms per proc to file
  // proc 0 pings each proc, receives its chunk, writes to file
  // all other procs wait for ping, send their chunk to proc 0

  int tmp,recvrow;
  MPI_Status status;
  MPI_Request request;

  if (me == 0) {
    for (int iproc = 0; iproc < nprocs; iproc++) {
      if (iproc) {
        MPI_Irecv(&buf[0],maxrow*ncol,MPI_DOUBLE,iproc,0,world,&request);
        MPI_Send(&tmp,0,MPI_INT,iproc,0,world);
        MPI_Wait(&request,&status);
        MPI_Get_count(&status,MPI_DOUBLE,&recvrow);
        recvrow /= ncol;
      } else recvrow = sendrow;
      
      n = 0;
      for(int i=0; i<recvrow; i++) {
	int tag = static_cast<int>(buf[n++]);
	int type = static_cast<int>(buf[n++]);
	int index = static_cast<int>(buf[n++]);

	if(type) type = fix_evb->Engine->evb_type->id[type-1];

	fprintf(fp,"%d %d %d\n",tag,type,index);
      }
    }
  
  } else {
    MPI_Recv(&tmp,0,MPI_INT,0,0,world,&status);
    MPI_Rsend(&buf[0],sendrow*ncol,MPI_DOUBLE,0,0,world);
  }

  memory->destroy(buf);
}
