/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_LIST_H
#define EVB_LIST_H

#include "pointers.h"
#include "EVB_pointers.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

#define SYS_LIST 0
#define ENV_LIST 1
#define EVB_LIST 2
#define CPL_LIST 3

#ifdef DLEVB_MODEL_SUPPORT
  #define EVB_LJ_LIST 4
#endif

class EVB_List : protected Pointers, protected EVB_Pointers
{
  public:
    EVB_List(class LAMMPS *, class EVB_Engine *);
    virtual ~EVB_List();
  
  public:
  
  struct SubPairList
  {
      int inum;
      int *ilist;
      int *numneigh;
      int **firstneigh;
  };
  
  class NeighList *save_pairlist;

  int npair_cpl, max_inum;
  SubPairList sys_pair, env_pair, evb_pair;
  
  int n_sys_bond, n_env_bond, n_evb_bond;
  int **sys_bond, **env_bond, **evb_bond;
  int max_bond;
  
  int n_sys_angle, n_env_angle, n_evb_angle;
  int **sys_angle, **env_angle, **evb_angle;
  int max_angle;
  
  int n_sys_dihedral, n_env_dihedral, n_evb_dihedral;
  int **sys_dihedral, **env_dihedral, **evb_dihedral;
  int max_dihedral;
  
  int n_sys_improper, n_env_improper, n_evb_improper;
  int **sys_improper, **env_improper, **evb_improper;
  int max_improper;

  void setup();
  void single_split();
  void single_split_omp();
  void single_combine();

  void change_list(int);
  int indicator;

  SubPairList cpl_pair;
  void multi_split();
  void multi_split_omp();
  void sci_split_inter();
  void sci_split_inter_omp();
  void sci_split_env();
  void sci_split_env_omp();
  void multi_combine();
  
#ifdef DLEVB_MODEL_SUPPORT
  SubPairList evb_lj_pair;
  void change_pairlist(int);
  void search_14coul();
#endif

};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

}

#endif
