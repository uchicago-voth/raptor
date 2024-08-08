/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_OUTPUT_H
#define EVB_OUTPUT_H

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "pointers.h"
#include "EVB_pointers.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Output : protected Pointers, protected EVB_Pointers
{
  public:
    EVB_Output(class LAMMPS *, class EVB_Engine *, char *);
    virtual ~EVB_Output();

    FILE *fp;

    int freq, ifreq;
    int react;
    int bin;
    int forceflush;

    int bCenter;
    int bNState;
    int bStates;
    int bEnergy;
    int bEneDecompose;
	
  public:
    int data_output(char*, int*, int, int);
    void print(const char*);
    void execute();
    void write_txt();
    void write_bin();
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
}

#endif
