/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "comm.h"
#include "universe.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "atom.h"

#include "EVB_source.h"
#include "EVB_type.h"

/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

EVB_Type::EVB_Type(LAMMPS *lmp, EVB_Engine *engine) : Pointers(lmp), EVB_Pointers(engine)
{
  type_count = 0;
  atom_count = 0;
  atom_per_molecule = 0;
 
  nCOC = NULL;
  iCOC = NULL;

  nbonds = nangles = ndihedrals = nimpropers = NULL;

  name = NULL;
  type_index = type_natom = NULL;
  atom_type = is_kernel = NULL;
  atom_q = qsum = qsqsum = NULL;
  num_bond = num_angle = num_dihedral = num_improper = starting_rc = id = NULL;
  
  bond_type = bond_atom = angle_type = angle_atom1 = angle_atom2 =
  angle_atom3 = dihedral_type = dihedral_atom1 = dihedral_atom2 =
  dihedral_atom3 = dihedral_atom4 = improper_type = improper_atom1 = 
  improper_atom2 = improper_atom3 = improper_atom4 = NULL;
  
  bond_map = NULL;

  atp_list = atp_index = NULL;
}

/* ---------------------------------------------------------------------- */

void EVB_Type::alloc_type(int n)
{
    for(int i=0; i<type_count; i++)  memory->destroy(bond_map[i]);
    
    type_count = n;
    name = (char**) memory->srealloc(name,sizeof(char*)*n,"EVB_Type:name");
    type_index = (int*) memory->srealloc(type_index,type_count*sizeof(int),"EVB_Type:type_index");
    type_natom = (int*) memory->srealloc(type_natom,type_count*sizeof(int),"EVB_Type:type_index");
    starting_rc = (int*) memory->srealloc(starting_rc,type_count*sizeof(int),"EVB_Type:starting_rc");
    id = (int*) memory->srealloc(id,type_count*sizeof(int),"EVB_Type:id");
    
    nbonds     = (int*) memory->srealloc(nbonds,     type_count*sizeof(int),"EVB_Type:nbonds");
    nangles    = (int*) memory->srealloc(nangles,    type_count*sizeof(int),"EVB_Type:nangles");
    ndihedrals = (int*) memory->srealloc(ndihedrals, type_count*sizeof(int),"EVB_Type:ndihedrals");   
    nimpropers = (int*) memory->srealloc(nimpropers, type_count*sizeof(int),"EVB_Type:nimpropers");

    bond_map  = (int***) memory->srealloc(bond_map,type_count*sizeof(int**),"EVB_Type:bond_map");
    for(int i=0; i<type_count; i++) bond_map[i] = NULL;
    
    nCOC = (int*) memory->srealloc(nCOC,type_count*sizeof(int),"EVB_Type:nCOC");
    iCOC = (int**) memory->srealloc(iCOC,type_count*sizeof(int*),"EVB_Type:iCOC");
    memset(nCOC,0, sizeof(int)*type_count);
    memset(iCOC,0, sizeof(int*)*type_count);
}

/* ---------------------------------------------------------------------- */

void EVB_Type::grow_atom(int n)
{
  atom_count+=n;
  
  int bond_per_atom = 6;
  int angle_per_atom = 12;
  int dihedral_per_atom = 16;
  int improper_per_atom = 3;
  
  atom_type = (int*) memory->srealloc(atom_type,atom_count*sizeof(int),"EVB_Type:atom_type");
  atom_q = (double*) memory->srealloc(atom_q,atom_count*sizeof(double),"EVB_Type:atom_q");
  is_kernel = (int*) memory->srealloc(is_kernel,atom_count*sizeof(int),"EVB_Type:is_kernel");
  num_bond = (int*) memory->srealloc(num_bond,atom_count*sizeof(int),"EVB_Type:num_bond");
  num_angle = (int*) memory->srealloc(num_angle,atom_count*sizeof(int),"EVB_Type:num_angle");
  num_dihedral = (int*) memory->srealloc(num_dihedral,atom_count*sizeof(int),"EVB_Type:num_dihedral");
  num_improper = (int*) memory->srealloc(num_improper,atom_count*sizeof(int),"EVB_Type:num_improper");

  memory->grow(bond_type,atom_count,bond_per_atom,"EVB_Type:bond_type");
  memory->grow(bond_atom,atom_count,bond_per_atom,"EVB_Type:bond_atom");
  
  memory->grow(angle_type,atom_count,angle_per_atom,"EVB_Type:angle_type");
  memory->grow(angle_atom1,atom_count,angle_per_atom,"EVB_Type:angle_atom1");
  memory->grow(angle_atom2,atom_count,angle_per_atom,"EVB_Type:angle_atom2");
  memory->grow(angle_atom3,atom_count,angle_per_atom,"EVB_Type:angle_atom3");
  
  memory->grow(dihedral_type,atom_count,dihedral_per_atom,"EVB_Type:dihedral_type");
  memory->grow(dihedral_atom1,atom_count,dihedral_per_atom,"EVB_Type:dihedral_atom1");
  memory->grow(dihedral_atom2,atom_count,dihedral_per_atom,"EVB_Type:dihedral_atom2");
  memory->grow(dihedral_atom3,atom_count,dihedral_per_atom,"EVB_Type:dihedral_atom3");
  memory->grow(dihedral_atom4,atom_count,dihedral_per_atom,"EVB_Type:dihedral_atom4");
  
  memory->grow(improper_type,atom_count,improper_per_atom,"EVB_Type:improper_type");
  memory->grow(improper_atom1,atom_count,improper_per_atom,"EVB_Type:improper_atom1");
  memory->grow(improper_atom2,atom_count,improper_per_atom,"EVB_Type:improper_atom2");
  memory->grow(improper_atom3,atom_count,improper_per_atom,"EVB_Type:improper_atom3");
  memory->grow(improper_atom4,atom_count,improper_per_atom,"EVB_Type:improper_atom4");
}

/* ---------------------------------------------------------------------- */

EVB_Type::~EVB_Type()
{  
  memory->sfree(type_index);
  memory->sfree(type_natom);
  memory->sfree(starting_rc);
  memory->sfree(id);
  
  memory->sfree(nbonds);
  memory->sfree(nangles);
  memory->sfree(ndihedrals);
  memory->sfree(nimpropers);

  memory->sfree(nCOC);
  for(int i=0;i<type_count; i++)
  {
      memory->sfree(name[i]);
      memory->sfree(iCOC[i]);
  }
  memory->sfree(name);
  memory->sfree(iCOC);
  
  memory->sfree(atom_type);
  memory->sfree(atom_q);
  memory->sfree(qsum);
  memory->sfree(qsqsum);
  memory->sfree(is_kernel);
  memory->sfree(num_bond);
  memory->sfree(num_angle);
  memory->sfree(num_dihedral);
  memory->sfree(num_improper);
  
  memory->destroy(bond_type);
  memory->destroy(bond_atom);
  
  memory->destroy(angle_type);
  memory->destroy(angle_atom1);
  memory->destroy(angle_atom2);
  memory->destroy(angle_atom3);
  
  memory->destroy(dihedral_type);
  memory->destroy(dihedral_atom1);
  memory->destroy(dihedral_atom2);
  memory->destroy(dihedral_atom3);
  memory->destroy(dihedral_atom4);
  
  memory->destroy(improper_type);
  memory->destroy(improper_atom1);
  memory->destroy(improper_atom2);
  memory->destroy(improper_atom3);
  memory->destroy(improper_atom4);
  
  for(int i=0; i<type_count; i++) memory->destroy(bond_map[i]);
  memory->sfree(bond_map);

  memory->sfree(atp_list);
  memory->sfree(atp_index);
}

/* ---------------------------------------------------------------------- */

void EVB_Type::init_kspace()
{
  qsum = (double*)memory->srealloc(qsum,sizeof(double)*type_count,"EVB_Type:qsum");
  qsqsum = (double*)memory->srealloc(qsqsum,sizeof(double)*type_count,"EVB_Type:qsqsum");
  
  for(int i=0; i<type_count; i++)
  {
    qsum[i] = qsqsum[i] = 0.0;
	double *q = atom_q+type_index[i];
	
	for(int j=0; j<type_natom[i]; j++)
	{
	  qsum[i] += q[j];
	  qsqsum[i] += (q[j]*q[j]);
	}
  }
}

/* ---------------------------------------------------------------------- */

int EVB_Type::data_type(char *buf, int *offset, int start, int end)
{
    char name_line[1000];
    char errline[255];

    FILE * fp = evb_engine->fp_cfg_out;
    if(universe->me == 0) fprintf(fp,"\n\nStarting to process molecule templates.\n");

    if(start==end) error->all(FLERR,"[EVB] Unexpected end of file.");
    
    int nset=0;
    int iset=start;   
    while(strcmp(buf+offset[start],"settype")==0) 
    { start+=3; nset++; }
    
    if(strstr(buf+offset[start],"segment.molecule_type")==0)
    {
        sprintf(errline,"[EVB] Cannot find key word [segment.molecule_type] at: \"%s\"",buf+offset[start]);
        error->all(FLERR,errline);
    }
    
    int t,count1=0, count2=0;
    for(t=start+1; t<end; t++)
    {
        if(strstr(buf+offset[t],"molecule_type.start"))
        {
            if(count1>count2) error->all(FLERR,"[EVB] Expecting key word [molecule_type.end].");
            count1++;
        }
        
        else if(strstr(buf+offset[t],"molecule_type.end"))
        {
            count2++;
            if(count2>count1) error->all(FLERR,"[EVB] Unexpected key word [molecule_type.end].");
        }
    }
    if(count1>count2) error->all(FLERR,"[EVB] Expecting key word [molecule_type.end].");

    if(universe->me == 0) fprintf(fp,"Allocating memory for %i kernels.\n",count1);

    alloc_type(count1);
    t = start+1;
    int n=0;
    
    for(int i=0; i<type_count; i++)
    {
        sprintf(errline,"[EVB] Wrong format exists after %s.",buf+offset[t]);
        int _end = t;
        while(strstr(buf+offset[_end],"molecule_type.end")==0) _end++;

        char *pp = strstr(buf+offset[t],"molecule_type.start");
        if(pp==0)
        {
            sprintf(errline,"[EVB] Expecting [molecule_type.start] at: \"%s\"",buf+offset[t]);
            error->all(FLERR,errline);
        }
        pp += 20;
        char *ppp=name_line;
        while(*pp && *pp!=']')
        {
            *ppp=*pp;
            pp++;
            ppp++;
        }
        *ppp = 0;
        name[i] = (char*) memory->smalloc(sizeof(char)*(strlen(name_line)+1),"EVB_Type:name[i]");
        strcpy(name[i],name_line); 
        t++;

	if(universe->me == 0) fprintf(fp,"\n%i: name= %s\n",i,name[i]);

        id[i]=-1;
        for(int j=0; j<nset; j++)
          if(strcmp(buf+offset[iset+3*j+1],name[i])==0)
          { id[i] = atoi(buf+offset[iset+3*j+2]); break; }
          //else fprintf(screen,"%s %s\n", buf+offset[iset+3*j+1],buf+offset[iset+3*j+2]);
        if(id[i]==-1)
        {
          sprintf(errline,"[EVB] ID is not set for type \"%s\".",name[i]);
          error->all(FLERR,errline);
        }
        
        if(t+6>=_end) error->all(FLERR,errline);
        
        type_index[i] = n;
        int count = atoi(buf+offset[t++]);
        type_natom[i] = count;
        if(type_natom[i]>atom_per_molecule) atom_per_molecule=type_natom[i];
        grow_atom(count);
        
        memory->grow(bond_map[i],count+1,count+1,"EVB_Type:bond_map[i]");
        for(int ii=0; ii<=count; ii++)
            for(int jj=0; jj<=count; jj++) bond_map[i][ii][jj]=0;
        
        
        int nbond, nangle, ndihedral, nimproper;
        nbond     = nbonds[i]     = atoi(buf+offset[t++]);
        nangle    = nangles[i]    = atoi(buf+offset[t++]);
        ndihedral = ndihedrals[i] = atoi(buf+offset[t++]);
        nimproper = nimpropers[i] = atoi(buf+offset[t++]);
        starting_rc[i] = atoi(buf+offset[t++]);

	if(universe->me == 0) {
	  fprintf(fp,"   nbond= %i  nangle= %i  ndihedral= %i  nimproper= %i  starting_rc= %i\n",
		  nbond,nangle,ndihedral,nimproper,starting_rc[i]);
	  fprintf(fp,"\n   Atoms in molecule\n");
	}

        if(t+type_natom[i]*3+nbond*3+nangle*4+ndihedral*5+nimproper*5>_end) error->all(FLERR,errline);

        for(int j=n; j<n+type_natom[i]; j++)
        {
            atom_type[j] = atoi(buf+offset[t++]);
            atom_q[j]    = atof(buf+offset[t++]);
            is_kernel[j] = atoi(buf+offset[t++]);
            
            num_bond[j]     = 0;
            num_angle[j]    = 0;
            num_dihedral[j] = 0;
            num_improper[j] = 0;

	    if(universe->me == 0) {
	      fprintf(fp,"   j= %i  atom_type= %i  atom_q= %f  is_kernel= %i\n",
		      j-n,atom_type[j],atom_q[j],is_kernel[j]);
	      if(atom_type[j] == 0) fprintf(fp,"WARNING!!!  atom_type shouldn't be zero.\n");
	    }
        }

	if(nbond > 0 && universe->me == 0) fprintf(fp,"\n   Bonds in molecule\n");

        for(int j=0; j<nbond; j++)
        {
            int a = atoi(buf+offset[t++]);
            int b = atoi(buf+offset[t++]);
            int c = atoi(buf+offset[t++]);
      
            int pos = n+a-1;
      
            bond_atom[pos][num_bond[pos]] = b;
            bond_type[pos][num_bond[pos]] = c;
            num_bond[pos]++;
      
            bond_map[i][a][b] = bond_map[i][b][a] = 1;

	    if(universe->me == 0) {
	      fprintf(fp,"   j= %i  bond_atoms= %i %i  bond_type= %i\n",j,a,b,c);
	      if(c == 0) fprintf(fp,"WARNING!!!  bond_type shouldn't be zero.\n");
	    }	    
        }
        
	if(nangle > 0 && universe->me == 0) fprintf(fp,"\n   Angles in molecule\n");

        for(int j=0; j<nangle; j++)
        {
            int a = atoi(buf+offset[t++]);
            int b = atoi(buf+offset[t++]);
            int c = atoi(buf+offset[t++]);
            int d = atoi(buf+offset[t++]);
            
            int pos = n+b-1;
            
            angle_atom1[pos][num_angle[pos]] = a;
            angle_atom2[pos][num_angle[pos]] = b;
            angle_atom3[pos][num_angle[pos]] = c;
            angle_type[pos][num_angle[pos]]  = d;
            num_angle[pos]++;
            
            bond_map[i][a][c] = bond_map[i][c][a] = 2;

	    if(universe->me == 0) {
	      fprintf(fp,"   j= %i  angle_atoms= %i %i %i  angle_type= %i\n",j,a,b,c,d);
	      if(d == 0) fprintf(fp,"WARNING!!!  angle_type shouldn't be zero.\n");
	    }
        }
    
	if(ndihedral > 0 && universe->me == 0) fprintf(fp,"\n   Dihedrals in molecule\n");

        for(int j=0; j<ndihedral; j++)
        {
            int a = atoi(buf+offset[t++]);
            int b = atoi(buf+offset[t++]);
            int c = atoi(buf+offset[t++]);
            int d = atoi(buf+offset[t++]);
            int e = atoi(buf+offset[t++]);
	  
            int pos = n+b-1;
            
            dihedral_atom1[pos][num_dihedral[pos]] = a;
            dihedral_atom2[pos][num_dihedral[pos]] = b;
            dihedral_atom3[pos][num_dihedral[pos]] = c;
            dihedral_atom4[pos][num_dihedral[pos]] = d;
            dihedral_type[pos][num_dihedral[pos]] =  e;
            num_dihedral[pos]++;
      
            if(bond_map[i][a][d]==0) bond_map[i][a][d] = bond_map[i][d][a] = 3;

	    if(universe->me == 0) {
	      fprintf(fp,"   j= %i  dihedral_atoms= %i %i %i %i  dihedral_type= %i\n",j,a,b,c,d,e);
	      if(e == 0) fprintf(fp,"WARNING!!!  dihedral_type shouldn't be zero.\n");
	    }
        }

	if(nimproper > 0 && universe->me == 0) fprintf(fp,"\n   Impropers in molecule\n");

        for(int j=0; j<nimproper; j++)
        {
            int a = atoi(buf+offset[t++]);
            int b = atoi(buf+offset[t++]);
            int c = atoi(buf+offset[t++]);
            int d = atoi(buf+offset[t++]);
            int e = atoi(buf+offset[t++]);
	  
            int pos = n+b-1;
	  
            improper_atom1[pos][num_improper[pos]] = a;
            improper_atom2[pos][num_improper[pos]] = b;
            improper_atom3[pos][num_improper[pos]] = c;
            improper_atom4[pos][num_improper[pos]] = d;
            improper_type[pos][num_improper[pos]]  = e;
            num_improper[pos]++;

	    if(universe->me == 0) {
	      fprintf(fp,"   j= %i  improper_atoms= %i %i %i %i  improper_type= %i\n",j,a,b,c,d,e);
	      if(e == 0) fprintf(fp,"WARNING!!!  improper_type shouldn't be zero.\n");
	    }
        }
    
	// COC information
	if(starting_rc[i]) {
	  if(universe->me == 0) fprintf(fp,"\n   Molecule has Center of Charge(COC).\n");
	  
	  if(t==_end) error->all(FLERR,errline);
	  nCOC[i] = atoi(buf+offset[t++]);
	  if(t+nCOC[i]>_end) error->all(FLERR,errline);
	  iCOC[i] = (int*) memory->srealloc(iCOC[i],sizeof(int)*nCOC[i],"EVB_Type:iCOC[i]");
	  for(int j=0; j<nCOC[i]; j++) iCOC[i][j] = atoi(buf+offset[t++]);

	  if(universe->me == 0) {
	    fprintf(fp,"   Number of atoms in COC definition: nCOC= %i\n",nCOC[i]);
	    fprintf(fp,"   Atom indicies: iCOC=");
	    for(int j=0; j<nCOC[i]; j++) fprintf(fp," %i",iCOC[i][j]);
	    fprintf(fp,"\n");
	  }
	}
	
        n+=type_natom[i];
        if(t!=_end) error->all(FLERR,errline);
        t++;

	if(universe->me == 0) fprintf(fp,"   Processing molecule template complete\n");
    }
    
    if(universe->me == 0) fprintf(fp,"\nChecking atom information");
   
    /******************************************************************************/
    // Check nbonds nangles ndihedrals nimpropers

    int _narray[4], narray[4];
    memset(_narray, 0, sizeof(int)*4);
    int nlocal = atom->nlocal;

    for(int i=0; i<nlocal; i++)
    {
      _narray[0] += atom->num_bond[i];
      _narray[1] += atom->num_angle[i];
      _narray[2] += atom->num_dihedral[i];
      _narray[3] += atom->num_improper[i];
    }
    MPI_Allreduce(_narray, narray, 4, MPI_INT, MPI_SUM, world);

    if(comm->me==0 && screen)
    {
      fprintf(screen,"[EVB] Check atom information [sum:record] ...\n");
      fprintf(screen,"      ---> bond     %-8d : " BIGINT_FORMAT "\n", narray[0], atom->nbonds);
      fprintf(screen,"      ---> angle    %-8d : " BIGINT_FORMAT "\n", narray[1], atom->nangles);
      fprintf(screen,"      ---> dihedral %-8d : " BIGINT_FORMAT "\n", narray[2], atom->ndihedrals);
      fprintf(screen,"      ---> improper %-8d : " BIGINT_FORMAT "\n", narray[3], atom->nimpropers);
    }

    atom->nbonds     = narray[0];
    atom->nangles    = narray[1];
    atom->ndihedrals = narray[2];
    atom->nimpropers = narray[3];

    /******************************************************************************/
    if(strstr(buf+offset[t],"segment.end")==0) error->all(FLERR,"[EVB] Expecting [segment.end] for [segment.molecule_type]");
    t++;

    /******************************************************************************/
    // for effective vdw
    natp = 0;

    n = atom->ntypes+1;
    atp_index = (int*) memory->srealloc(atp_index,sizeof(int)*n,"EVB_Type:atp_index");
    memset(atp_index,0, sizeof(int)*n);
    
    for(int i=0; i<atom_count; i++)
      if(atp_index[atom_type[i]]==0) atp_index[atom_type[i]]=(++natp);

    atp_list = (int*) memory->srealloc(atp_list,sizeof(int)*natp,"EVB_Type:atp_list");
    for(int i=1; i<n; i++)
      if(atp_index[i]) atp_list[(--atp_index[i])]=i;

    /*******************************************************************************/

    bond_per_atom     = 0;
    angle_per_atom    = 0;
    dihedral_per_atom = 0;
    improper_per_atom = 0;

    for(int i=0; i<atom_count; i++)
    {
      if(num_bond[i]>bond_per_atom) bond_per_atom             = num_bond[i];
      if(num_angle[i]>angle_per_atom) angle_per_atom          = num_angle[i];
      if(num_dihedral[i]>dihedral_per_atom) dihedral_per_atom = num_dihedral[i];
      if(num_improper[i]>improper_per_atom) improper_per_atom = num_improper[i];
    }

    if(comm->me==0 && screen)
    {
      fprintf(screen,"[EVB] max bond per atom: %d/%d\n", bond_per_atom,atom->bond_per_atom);
      fprintf(screen,"[EVB] max angle per atom: %d/%d\n", angle_per_atom,atom->angle_per_atom);
      fprintf(screen,"[EVB] max dihedral per atom: %d/%d\n", dihedral_per_atom,atom->dihedral_per_atom);
      fprintf(screen,"[EVB] max improper per atom: %d/%d\n", improper_per_atom,atom->improper_per_atom);
    }

    /*******************************************************************************/

    if(universe->me == 0) {
      fprintf(fp,"\nProcessing ALL molecule templates complete\n");
      fprintf(fp,"\n==================================\n");
    }

    return t;
}

int EVB_Type::get_type(char* type_name)
{
    int i;
    
    for(i=0; i<type_count; i++)
        if(strcmp(type_name,name[i])==0) break;

    if (i<type_count) return i+1;
    else return -1;
}
