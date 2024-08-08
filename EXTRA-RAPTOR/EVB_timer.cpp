/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "string.h"

#include "EVB_timer.h"
#include "comm.h"
#include "universe.h"

#include "mpi.h"
#include <string.h>

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

EVB_Timer::EVB_Timer(LAMMPS *lmp, EVB_Engine *engine) : Pointers(lmp), EVB_Pointers(engine)
{
  memset(stamp,0, sizeof(double)*TIMER_COUNT);
  memset(total,0, sizeof(double)*TIMER_COUNT);
  memset(count,0, sizeof(long)*TIMER_COUNT);
  memset(classname,0,sizeof(char*)*TIMER_COUNT);
  memset(functname,0,sizeof(char*)*TIMER_COUNT);
  
  last_output = 0.0;
  start = MPI_Wtime();
}

void EVB_Timer::_stamp(int handle, const char* cls_name, const char* func_name)
{
  if(classname[handle]==NULL)
  {
    classname[handle] = (char*) cls_name;
    functname[handle] = (char*) func_name;
  }
  
  stamp[handle] = MPI_Wtime(); 
}

void EVB_Timer::_click(int handle)
{
  double t = MPI_Wtime();
  t -= stamp[handle]; 
  total[handle] += t;
  count[handle] ++;
}

void EVB_Timer::output()
{
  double now = MPI_Wtime();
  
  {
    last_output = now;
    
    if(comm->me==0)
    {
      FILE *fp;
      if(universe->nworlds==1) fp = fopen("timing.log","w");
      else
      {
        char fname[100];
	sprintf(fname,"timing.log.%d", universe->iworld);
	fp = fopen(fname,"w");
      }
      
      fprintf(fp, "Total elapsed time is %0.1lf\n\n", now-start);
      fprintf(fp, "%16s %12s   %s\n", "Time", "Count", "Subroutine");
      fprintf(fp, "============================================================\n");
      
      for(int i=0; i<TIMER_COUNT; i++) if(count[i])
        fprintf(fp,"%16.1lf %12ld   %s::%s\n", total[i], count[i], classname[i], functname[i]);
	
      fclose(fp);
    }
  }
}

/* ---------------------------------------------------------------------- */
