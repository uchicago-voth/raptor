#ifndef EVB_SOURCE_H
#define EVB_SOURCE_H

#define _EVB_COPYRIGHT "VOTH GROUP"
#define _EVB_LINE "-------------------------------------------------------------------\n"

#define _EVB_DEFINE_AVEC_POINTERS \
  int *tag;  \
  int *type; \
  int *molecule; \
  int *mol_type;\
  int *mol_index;\
  double *q;\
  int *num_bond;\
  int *num_angle;\
  int *num_dihedral;\
  int *num_improper;\
  int **bond_type;\
  int **bond_atom;\
  int **angle_type;\
  int **angle_atom1;\
  int **angle_atom2;\
  int **angle_atom3;\
  int **dihedral_type;\
  int **dihedral_atom1;\
  int **dihedral_atom2;\
  int **dihedral_atom3;\
  int **dihedral_atom4;\
  int **improper_type;\
  int **improper_atom1;\
  int **improper_atom2;\
  int **improper_atom3;\
  int **improper_atom4

#define _EVB_REFRESH_AVEC_POINTERS \
  tag = atom->tag; \
  type = atom->type;\
  molecule = atom->molecule;\
  mol_type = evb_engine->mol_type;\
  mol_index = evb_engine->mol_index;\
  q = atom->q;\
  num_bond = atom->num_bond;\
  num_angle = atom->num_angle;\
  num_dihedral = atom->num_dihedral;\
  num_improper = atom->num_improper;\
  bond_type = atom->bond_type;\
  bond_atom = atom->bond_atom;\
  angle_type = atom->angle_type;\
  angle_atom1 = atom->angle_atom1;\
  angle_atom2 = atom->angle_atom2;\
  angle_atom3 = atom->angle_atom3;\
  dihedral_type = atom->dihedral_type;\
  dihedral_atom1 = atom->dihedral_atom1;\
  dihedral_atom2 = atom->dihedral_atom2;\
  dihedral_atom3 = atom->dihedral_atom3;\
  dihedral_atom4 = atom->dihedral_atom4;\
  improper_type = atom->improper_type;\
  improper_atom1 = atom->improper_atom1;\
  improper_atom2 = atom->improper_atom2;\
  improper_atom3 = atom->improper_atom3;\
  improper_atom4 = atom->improper_atom4



#define _EVB_CEC


#define VECTOR_ZERO(a) a[0]=a[1]=a[2]=0.0
#define VECTOR_PBC(a) domain->minimum_image(a[0],a[1],a[2])
#define VECTOR_R2(b,a) b=a[0]*a[0]+a[1]*a[1]+a[2]*a[2]
#define VECTOR_R(b,a) b=sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
#define VECTOR_SELF_ADD(b,a) b[0]+=a[0];b[1]+=a[1];b[2]+=a[2]
#define VECTOR_SELF_SUB(b,a) b[0]-=a[0];b[1]-=a[1];b[2]-=a[2]
#define VECTOR_SCALE(c,b) c[0]*=b;c[1]*=b;c[2]*=b
#define VECTOR_SUB(c,a,b) c[0]=a[0]-b[0];c[1]=a[1]-b[1];c[2]=a[2]-b[2]
#define VECTOR_SCALE_SUB(c,a,b) c[0]-=(a[0]*b);c[1]-=(a[1]*b);c[2]-=(a[2]*b)
#define VECTOR_ADD(c,a,b) c[0]=a[0]+b[0];c[1]=a[1]+b[1];c[2]=a[2]+b[2]
#define VECTOR_SCALE_ADD(c,a,b) c[0]+=(a[0]*b);c[1]+=(a[1]*b);c[2]+=(a[2]*b)
#define VECTOR_COPY(a,b) a[0]=b[0];a[1]=b[1];a[2]=b[2];
#endif
