/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_POINTERS_H
#define EVB_POINTERS_H

#include "EVB_engine.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Pointers
{
  public:
    EVB_Pointers(EVB_Engine *engine) :
    evb_engine(engine),
    evb_kspace(engine->evb_kspace),
    evb_type(engine->evb_type),
    evb_chain(engine->evb_chain),
    evb_reaction(engine->evb_reaction),
    evb_complex(engine->evb_complex),
    evb_list(engine->evb_list),
    evb_matrix(engine->evb_matrix),
    evb_repulsive(engine->evb_repulsive),
    evb_offdiag(engine->evb_offdiag),
    evb_effpair(engine->evb_effpair), 
    evb_timer(engine->evb_timer)
    { }
    
  virtual ~EVB_Pointers() {}
  
  protected:
    EVB_Engine *evb_engine;
    EVB_KSpace *&evb_kspace;  
        
    EVB_Type *&evb_type;  
    EVB_Chain *&evb_chain;
    EVB_List *&evb_list;
    EVB_Reaction *&evb_reaction;   
    EVB_Matrix *&evb_matrix;
    EVB_Complex *&evb_complex; 
    EVB_Repulsive *&evb_repulsive;
    EVB_OffDiag *&evb_offdiag;  
    EVB_EffPair *&evb_effpair;
    EVB_Timer *&evb_timer;
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
