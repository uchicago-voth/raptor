/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_TIMER_H
#define EVB_TIMER_H

#include "pointers.h"
#include "EVB_pointers.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

#define USE_EVB_TIMER

#include "EVB_timer_const.h"

class EVB_Timer : protected Pointers, protected EVB_Pointers
{
  public:
  
    EVB_Timer(class LAMMPS *, class EVB_Engine *);
    virtual ~EVB_Timer() {};
    
    int ntimer;
    
    double stamp [TIMER_COUNT];
    double total [TIMER_COUNT];
    long   count [TIMER_COUNT];
    char*  classname [TIMER_COUNT];
    char*  functname [TIMER_COUNT];
    
    void _stamp(int, const char*, const char*);
    void _click(int);
    
    double start;
    double last_output;
    void output();
};

#ifdef USE_EVB_TIMER
  #define TIMER_STAMP(cls, func) { evb_timer->_stamp(TIMER_##cls##_##func, #cls, #func); }
  #define TIMER_CLICK(cls, func) { evb_timer->_click(TIMER_##cls##_##func); }
#else
  #define TIMER_STAMP(cls, func) {}
  #define TIMER_CLICK(cls, func) {}
#endif

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
}

#endif
