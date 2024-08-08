/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#ifndef EVB_TYPE_H
#define EVB_TYPE_H

#include "pointers.h"
#include "EVB_pointers.h"

namespace LAMMPS_NS {

/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/

class EVB_Type : protected Pointers, protected EVB_Pointers
{
  public:
    EVB_Type(class LAMMPS *, class EVB_Engine *);
    virtual ~EVB_Type();
  
  public:
    int type_count;
    int atom_count;
    int atom_per_molecule;
    int bond_per_atom;
    int angle_per_atom;
    int dihedral_per_atom;
    int improper_per_atom;

    // per type array
    int *type_index, *type_natom;
    int *nbonds, *nangles, *ndihedrals, *nimpropers;
    int *nCOC, **iCOC;
    int *starting_rc;
    int ***bond_map;
    double *qsum, *qsqsum;
    char **name;    
    int *id;

    // per atom array
    int *atom_type;
    double *atom_q;
    int *is_kernel;
    int *num_bond;
    int **bond_type,**bond_atom;
    int *num_angle;
    int **angle_type;
    int **angle_atom1,**angle_atom2,**angle_atom3;
    int *num_dihedral;
    int **dihedral_type;
    int **dihedral_atom1,**dihedral_atom2,**dihedral_atom3,**dihedral_atom4;
    int *num_improper;
    int **improper_type;
    int **improper_atom1,**improper_atom2,**improper_atom3,**improper_atom4;
    
    // for sci-evb effective vdw
    int natp;   // # of atom types used by MS-EVB
    int *atp_list;  // array of types ID, index from 0 to natp-1;
    int *atp_index; // array of types index, from 1 to atom->ntypes, 
                    // atp_index[I], means index of type I in atp_list; 

  public:
    int get_type(char*);
    void alloc_type(int);
    void grow_atom(int);
    int data_type(char*,int*, int,int);
    void init_kspace();
};
  
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
/*------------------------------------------------------------------------*/
}

#endif
