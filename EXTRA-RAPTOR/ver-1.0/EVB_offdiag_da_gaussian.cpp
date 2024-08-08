/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Tianying Yan, Yuxing Peng
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "EVB_complex.h"
#include "EVB_engine.h"
#include "EVB_type.h"
#include "EVB_offdiag_da_gaussian.h"
#include "EVB_kspace.h"
#include "EVB_source.h"
#include "EVB_timer.h"

#include "force.h"
#include "pair.h"
#include "neigh_list.h"
#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "universe.h"
#include "update.h"
#include "mp_verlet.h"

#define DATOM 0
#define AATOM 1

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_OffDiag_DA_Gaussian::EVB_OffDiag_DA_Gaussian(LAMMPS *lmp, EVB_Engine *engine) : EVB_OffDiag(lmp,engine)
{
  etp_A_exch = etp_B_exch = n_A_exch = n_B_exch = 0;
  q_A_exch = q_B_exch = q_A_save = q_B_save = NULL;
  
  size_exch_chg = 0;
  is_exch_chg = exch_list = NULL;
 
#ifdef OUTPUT_3BODY
  if(comm->me==0)
  {
    fp = fopen("3body.dat","w");
    timestep = center = -1;
  }
#endif 
}

/* ---------------------------------------------------------------------- */

EVB_OffDiag_DA_Gaussian::~EVB_OffDiag_DA_Gaussian()
{
  if (q_A_exch) delete [] q_A_exch;
  if (q_B_exch) delete [] q_B_exch;
  if (q_A_save) delete [] q_A_save;
  if (q_B_save) delete [] q_B_save;

  memory->sfree(is_exch_chg);
  memory->sfree(exch_list);

#ifdef OUTPUT_3BODY
  if(comm->me==0)
  {
    fclose(fp);
  }
#endif
}

/* ---------------------------------------------------------------------- */

int EVB_OffDiag_DA_Gaussian::checkout(int* _index)
{
  map = evb_engine->molecule_map;
  natom = atom->nlocal + atom->nghost;

  icomplex = evb_complex->id-1;
  istate = evb_complex->current_status;  
  
  mol_A = evb_complex->molecule_A[istate];
  mol_B = evb_complex->molecule_B[istate];

  int m1, m2, m3;
  if(mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
  if(mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
        
  _index[0] = map[m1][atom_A_Rq[0]];
  _index[1] = map[m2][atom_A_Rq[1]];

  return 2;
}

/* ---------------------------------------------------------------------- */

int EVB_OffDiag_DA_Gaussian::data_offdiag(char *buf, int* offset, int start, int end)
{
    int t = start;

    FILE * fp = evb_engine->fp_cfg_out;
    
    // Input atom index for geometry part
    
    // three-body index  
    mol_A_Rq [DATOM] = atoi(buf+offset[t++]);
    atom_A_Rq[DATOM] = atoi(buf+offset[t++]);
    mol_A_Rq [AATOM] = atoi(buf+offset[t++]);  
    atom_A_Rq[AATOM] = atoi(buf+offset[t++]);
 
    if(universe->me == 0) {
      fprintf(fp,"   Identity of particle 1: molecule= %i  index= %i.\n",mol_A_Rq[DATOM],atom_A_Rq[DATOM]);
      fprintf(fp,"   Identity of particle 2: molecule= %i  index= %i.\n",mol_A_Rq[AATOM],atom_A_Rq[AATOM]);
    }

    _c1  =atof(buf+offset[t++]);
    _c2  =atof(buf+offset[t++]);
    _c3  =atof(buf+offset[t++]);
    
    if(universe->me == 0) {
      fprintf(fp,"\n   Off-diagonal definition: VIJ= (VIJ_CONST + V_EX) * A_GEO.\n");
      fprintf(fp,"\n   Parameters for Gaussian geometric factor: A_GEO(R) = C1 * e^(-C2 * (R - R0)^2).\n");
      fprintf(fp,"   C1= %f\n",_c1);
      fprintf(fp,"   C2= %f\n",_c2);
      fprintf(fp,"   C3= %f\n",_c3);
    }

    // Input Vij information
    Vij_const = atof(buf+offset[t++]);
    is_Vij_ex = atoi(buf+offset[t++]);  
    
    if(universe->me == 0) {
      fprintf(fp,"\n   VIJ_CONST= %f\n",Vij_const);
      fprintf(fp,"\n   Available methods for V_EX: 0= None, 1= K-Space, 2= Debye, 3= Wolf, 4= CGIS, 5= Electrode.\n");
      fprintf(fp,"   Method selected for V_EX: is_Vij_ex= %i (Nonzero values must match across all off-diagonals).\n",is_Vij_ex);
    }

    if(is_Vij_ex)
    {
      if(is_Vij_ex==2) {
	kappa = atof(buf+offset[t++]);
	evb_engine->flag_DIAG_QEFF = 1;
	if(universe->me == 0) fprintf(fp,"   kappa= %f\n",kappa);
      }

      if(is_Vij_ex==3) {
	kappa = atof(buf+offset[t++]);
	if(universe->me == 0) fprintf(fp,"   kappa= %f\n",kappa);
	evb_engine->flag_DIAG_QEFF = 1;
      }

      if(is_Vij_ex==4) evb_engine->flag_DIAG_QEFF = 1;

      if(is_Vij_ex==5) evb_engine->flag_DIAG_QEFF = 1;

        // Input and setup exchange charges
        qsum_exch = qsum_save = qsqsum_exch = qsqsum_save =0.0;
	
        char* type_name;
        char errline[255];
        int type_id;
        
        type_name = buf+offset[t++];
        etp_A_exch = evb_type->get_type(type_name);

	if(universe->me == 0) fprintf(fp,"   First molecule in exchange complex is %s: type= %i.\n",type_name,etp_A_exch);
        
        if(etp_A_exch==-1)
        {
          sprintf(errline,"[EVB] Undefined molecule_type [%s].", type_name);
          error->all(FLERR,errline);
        }

        type_name = buf+offset[t++];
        etp_B_exch = evb_type->get_type(type_name) ;
        
	if(universe->me == 0) fprintf(fp,"   Second molecule in exchange complex is %s: type= %i.\n",type_name,etp_B_exch);

        if(etp_B_exch==-1)
        {
          sprintf(errline,"[EVB] Undefined molecule_type [%s].", type_name);
          error->all(FLERR,errline);
        }
        
        n_A_exch = evb_type->type_natom[etp_A_exch-1];
        n_B_exch = evb_type->type_natom[etp_B_exch-1];
        q_A_exch = new double [n_A_exch];
        q_B_exch = new double [n_B_exch];
        q_A_save = new double [n_A_exch];
        q_B_save = new double [n_B_exch];

        int nexch = n_A_exch + n_B_exch;
        if(nexch>max_nexch) max_nexch = nexch;
        
        double *qA = evb_type->atom_q + evb_type->type_index[etp_A_exch-1] ;
        for(int i=0; i<n_A_exch; i++) {
	  q_A_exch[i] =  atof(buf+offset[t++]);
	  q_A_save[i] = qA[i];
	  qsum_exch += q_A_exch[i]; qsqsum_exch += q_A_exch[i]*q_A_exch[i];
	  qsum_save += q_A_save[i]; qsqsum_save += q_A_save[i]*q_A_save[i];
        }

	if(universe->me == 0) {
	  fprintf(fp,"\n   Number of atoms in first molecule: n_A_exch= %i.\n",n_A_exch);
	  for(int i=0; i<n_A_exch; i++) fprintf(fp,"   i= %i  q_A_exch= %f  q_A_save= %f\n",i,q_A_exch[i],q_A_save[i]);
	}
	        
        double *qB = evb_type->atom_q + evb_type->type_index[etp_B_exch-1] ;
        for(int i=0; i<n_B_exch; i++) {
	  q_B_exch[i] =  atof(buf+offset[t++]);
	  q_B_save[i] = qB[i];
	  qsum_exch += q_B_exch[i]; qsqsum_exch += q_B_exch[i]*q_B_exch[i];
	  qsum_save += q_B_save[i]; qsqsum_save += q_B_save[i]*q_B_save[i];
        }

	if(universe->me == 0) {
	  fprintf(fp,"\n   Number of atoms in second molecule: n_B_exch= %i.\n",n_B_exch);
	  for(int i=0; i<n_B_exch; i++) fprintf(fp,"   i= %i  q_B_exch= %f  q_B_save= %f\n",i,q_B_exch[i],q_B_save[i]);
	}
    }
    
    return t;
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_DA_Gaussian::compute(int vflag)
{
  MP_Verlet *mp_verlet = evb_engine->mp_verlet;
  
  // set up lists pointers and env variables
  
  map = evb_engine->molecule_map;
  natom = atom->nlocal + atom->nghost;
  
  // set up index
  icomplex = evb_complex->id-1;
  istate = evb_complex->current_status;  
  mol_A = evb_complex->molecule_A[istate];
  mol_B = evb_complex->molecule_B[istate];
  
  // init energy, virial
  A_Rq = f_R = g_q = 0.0;
  Vij = Vij_const;
  Vij_ex = Vij_ex_short = Vij_ex_long = 0.0;
  energy = 0.0;  
  if(vflag) {
    memset(virial,0, sizeof(double)*6);
    if(evb_kspace) memset(&(evb_kspace->off_diag_virial[0]), 0.0, sizeof(double)*6);
  }
  
  /**************************************************/
  /****** Geometry Energy ***************************/
  /**************************************************/

  if(!mp_verlet || mp_verlet->is_master) {
  
  if(comm->me == evb_engine->rc_rank[icomplex])
  {
    // Init three-body system
    int m1, m2;
    if(mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
    if(mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
    
    index_A_Rq[DATOM] = map[m1][atom_A_Rq[DATOM]];
    index_A_Rq[AATOM] = map[m2][atom_A_Rq[AATOM]];
    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    
    // Calc dr_DA
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);
    
    // f(R) part
    cal_f_term_sym();
    
    A_Rq = f_R;
    
    index[0] = index_A_Rq[DATOM];
    index[1] = index_A_Rq[AATOM];
  }
  
  MPI_Bcast(&A_Rq,1,MPI_DOUBLE,evb_engine->rc_rank[icomplex],world);
  
  }

  /**************************************************/
  /****** Potential part ****************************/
  /**************************************************/
  
  if(mp_verlet && mp_verlet->is_master==0 && !is_Vij_ex) { energy = 0.0; return; }
  
  if(is_Vij_ex)
  {
    if(mp_verlet && mp_verlet->is_master==0)
      if(is_Vij_ex > 1 || evb_engine->flag_ACC) return;
   
    // init exchanged charge
    
    init_exch_chg(); 

    if(evb_kspace) 
    {
      evb_kspace->off_diag_energy = 0.0;
   
      if(is_Vij_ex==2) Vij_ex_short = exch_chg_debye(vflag);
      else if(is_Vij_ex==3) Vij_ex_short = exch_chg_wolf(vflag);
      else if(is_Vij_ex==4) Vij_ex_short = exch_chg_cgis(vflag);
      else if(is_Vij_ex==5) {
	Vij_ex_short = 0.0;
	evb_kspace->A_Rq = A_Rq;
        evb_kspace->is_exch_chg = is_exch_chg;
        evb_kspace->compute_exch(vflag);
      } else {
TIMER_STAMP(EVB_OffDiag_DA_Gaussian, compute__evb_kspace_compute_exch);

        if(!mp_verlet || mp_verlet->is_master==1) Vij_ex_short = exch_chg_long(vflag);
        
        if(!mp_verlet)
        {
          evb_kspace->A_Rq = A_Rq;
          evb_kspace->is_exch_chg = is_exch_chg;
          evb_kspace->compute_exch(vflag);
        }
        else if(mp_verlet->is_master==0)
        {
          evb_kspace->A_Rq = 1.0 ;
          evb_kspace->is_exch_chg = is_exch_chg;
          evb_kspace->compute_exch(vflag);
        }

TIMER_CLICK(EVB_OffDiag_DA_Gaussian, compute__evb_kspace_compute_exch);
      }
    }
    else Vij_ex_short = exch_chg_cut(vflag);

    if(!mp_verlet || mp_verlet->is_master==1) MPI_Allreduce(&Vij_ex_short,&Vij_ex,1,MPI_DOUBLE,MPI_SUM,world);
    if(!mp_verlet || mp_verlet->is_master==0) if(evb_kspace) Vij_ex += evb_kspace->off_diag_energy;  
    
    Vij += Vij_ex;
    
    // Resume exchanged charges
    resume_chg();
  }

  /**************************************************/
  /**************  Force and Virial *****************/
  /**************************************************/
  
  if(!mp_verlet || mp_verlet->is_master==1) 
  if(comm->me == evb_engine->rc_rank[icomplex])
  {
    cal_force_sym(vflag);
  }
  
  /**************************************************/
  /************** Energy ****************************/
  /**************************************************/
  
  // Hij = Vij * A(R,q)

  energy = Vij * A_Rq;
  
  // local virial + kspace virial devided by total number of cpu's

  if(!mp_verlet || mp_verlet->is_master==0) 
  if (evb_kspace && vflag) 
    for (int i = 0; i < 6; i++)
      virial[i] += (evb_kspace->off_diag_virial[i] / comm->nprocs);
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_DA_Gaussian::cal_f_term_sym()
{
  double r2_da, r_da;
  
  // Cal R_OO
  VECTOR_R2(r2_da,dr_DA);
  r_da = sqrt(r2_da);

  double dr = r_da - _c3;
  
  // f(R)
  f_R = _c1 * exp(-_c2 * dr * dr);
  
  // Derivative of f(R)
  df_O = 2.0 * _c2 * dr * f_R;
  df_R = df_O / r_da;
  df_H = 0.0;
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_DA_Gaussian::cal_force_sym(int vflag)
{
    double **f = atom->f;
    
    double tfa = Vij * df_R;
    
    double dfax, dfay, dfaz;
    
    dfax = tfa * dr_DA[0];
    f[index_A_Rq[DATOM]][0] += dfax;
    f[index_A_Rq[AATOM]][0] -= dfax;
    
    dfay = tfa * dr_DA[1];
    f[index_A_Rq[DATOM]][1] += dfay;
    f[index_A_Rq[AATOM]][1] -= dfay;
    
    dfaz = tfa * dr_DA[2];
    f[index_A_Rq[DATOM]][2] += dfaz;
    f[index_A_Rq[AATOM]][2] -= dfaz;
    
    if(vflag) {
      virial[0] += dfax * dr_DA[0];
      virial[1] += dfay * dr_DA[1];
      virial[2] += dfaz * dr_DA[2];
      virial[3] += dfax * dr_DA[1];
      virial[4] += dfax * dr_DA[2];
      virial[5] += dfay * dr_DA[2];
    }
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_DA_Gaussian::init_exch_chg()
{
  if(natom>size_exch_chg)
  {
    size_exch_chg = natom;
    is_exch_chg = (int *) memory->srealloc(is_exch_chg,sizeof(int)*natom, "EVB_OffDiag_DA_Gaussian:is_exch_chg");
    exch_list = (int *) memory->srealloc(exch_list,sizeof(int)*natom, "EVB_OffDiag_DA_Gaussian:exch_list");
  }

  memset(is_exch_chg,0,sizeof(int)*natom);
  
  n_exch_chg = n_exch_chg_local = 0;
 
  double *q = atom->q;
  int *molecule = atom->molecule;
  int *mol_index = evb_engine->mol_index;

  int nlocal = atom->nlocal;
  for(int i=0; i<natom; i++) 
  {
    int type = 0;
    if(molecule[i]==mol_A) type = 1;
    else if(molecule[i]==mol_B) type = 2;

    if (type)
    {
        is_exch_chg[i]=1;
        exch_list[n_exch_chg] = i;
        iexch[n_exch_chg++] = i;
        
        if(n_exch_chg==max_nexch*27) error->one(FLERR,"[EVB] exch_chg overflow!");
		
        if(type==1) q[i]=q_A_exch[mol_index[i]-1];
        else if(type==2) q[i]=q_B_exch[mol_index[i]-1];

        if(i<nlocal) n_exch_chg_local++;
    }
  }

  // AWGL : helper flag 
  evb_engine->has_exch_chg = (n_exch_chg > 0) ? 1 : 0;

  (*ptr_nexch) = n_exch_chg;
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_DA_Gaussian::resume_chg()
{
  double *q = atom->q;
  int *molecule = atom->molecule;
  int *mol_index = evb_engine->mol_index;
  int nlocal = atom->nlocal;
  
  for(int i=0; i<n_exch_chg; i++)
  {
    int id = exch_list[i];
    qexch[i]=q[id]*A_Rq;
        
    if(molecule[id]==mol_A) q[id] = q_A_save[mol_index[id]-1];
    else if(molecule[id]==mol_B) q[id] = q_B_save[mol_index[id]-1];
  }
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_DA_Gaussian::sci_setup(int vflag)
{
  // set up lists pointers and env variables

  map = evb_engine->molecule_map;
  natom = atom->nlocal + atom->nghost;
  
  // set up index
  icomplex = evb_complex->id-1;
  istate = evb_complex->current_status;  
  mol_A = evb_complex->molecule_A[istate];
  mol_B = evb_complex->molecule_B[istate];
  
  // init energy, virial
  A_Rq = f_R = g_q = 0.0;
  Vij_ex = Vij_ex_short = Vij_ex_long = 0.0;
  energy = 0.0; 
  if (vflag) memset(virial,0, sizeof(double)*6);
  
  /**************************************************/
  /****** Geometry Energy ***************************/
  /**************************************************/
  
  if(comm->me == evb_engine->rc_rank[icomplex])
  {
    // Init three-body system
	
    int m1, m2;
    if(mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
    if(mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
    
    index_A_Rq[DATOM] = map[m1][atom_A_Rq[DATOM]];
    index_A_Rq[AATOM] = map[m2][atom_A_Rq[AATOM]];
    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    
    // Calc dr_DA
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);
    
    // f(R) part
    cal_f_term_sym();

    A_Rq = f_R;
  }
  
  MPI_Bcast(&A_Rq,1,MPI_DOUBLE,evb_engine->rc_rank[icomplex],world);
  
  /**************************************************/
  /****** Potential part ****************************/
  /**************************************************/
  
  Vij = Vij_const;
  
  if(is_Vij_ex)
    {
      init_exch_chg();
      
      if(evb_kspace)
	{
	  if(is_Vij_ex==2) Vij_ex_short = exch_chg_debye(vflag);
	  else if(is_Vij_ex==3) Vij_ex_short = exch_chg_wolf(vflag);
	  else if(is_Vij_ex==4) Vij_ex_short = exch_chg_cgis(vflag);
	  else if(is_Vij_ex==5) {
	    Vij_ex_short = 0.0;
	    evb_kspace->A_Rq = A_Rq;
	    evb_kspace->is_exch_chg = is_exch_chg;
	    evb_kspace->compute_exch(vflag);
	  } else {
	    Vij_ex_short = exch_chg_long(vflag);
	    evb_kspace->A_Rq = A_Rq;
	    evb_kspace->is_exch_chg = is_exch_chg;
	    evb_kspace->compute_exch(vflag);
	  }
	}
      else Vij_ex_short = exch_chg_cut(vflag);
      
      MPI_Allreduce(&Vij_ex_short,&Vij_ex,1,MPI_DOUBLE,MPI_SUM,world);
      if (evb_kspace) Vij_ex += evb_kspace->off_diag_energy;
      
      Vij += Vij_ex;
      
      resume_chg();
    }
  
  // energy
  energy = A_Rq * Vij;  
}

/* ---------------------------------------------------------------------- 
   Same as sci_setup(), but without energy/force calculation.
   ---------------------------------------------------------------------- */

void EVB_OffDiag_DA_Gaussian::sci_setup_mp()
{
  // set up lists pointers and env variables

  map = evb_engine->molecule_map;
  natom = atom->nlocal + atom->nghost;
  
  // set up index
  icomplex = evb_complex->id-1;
  istate = evb_complex->current_status;  
  mol_A = evb_complex->molecule_A[istate];
  mol_B = evb_complex->molecule_B[istate];
  
  // init energy, virial
  A_Rq = f_R = g_q = 0.0;
  
  // Calculate geometric factor
  if(comm->me == evb_engine->rc_rank[icomplex]) {
    int m1, m2;
    if(mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
    if(mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
    
    index_A_Rq[DATOM] = map[m1][atom_A_Rq[DATOM]];
    index_A_Rq[AATOM] = map[m2][atom_A_Rq[AATOM]];

    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    
    // Calc dr_DA
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);
    
    // f(R) part
    cal_f_term_sym();

    A_Rq = f_R;
  }
  
  MPI_Bcast(&A_Rq,1,MPI_DOUBLE,evb_engine->rc_rank[icomplex],world);
  
  // Setup exchange charges
  
  if(is_Vij_ex) {
    init_exch_chg();
    resume_chg();
  }
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_DA_Gaussian::sci_compute(int vflag)
{
  // set up lists pointers and env variables
  map = evb_engine->molecule_map;
  natom = atom->nlocal + atom->nghost;
  
  // set up index
  icomplex = evb_complex->id-1;
  istate = evb_complex->current_status;  
  mol_A = evb_complex->molecule_A[istate];
  mol_B = evb_complex->molecule_B[istate];
  int* parent = evb_complex->parent_id;
  
  Vij *= 2.0 * evb_complex->Cs[istate] * evb_complex->Cs[parent[istate]]; 
  
  // init energy, virial
  if (vflag) memset(virial,0, sizeof(double)*6);
  
  /**************************************************/
  /****** Geometry Force  ***************************/
  /**************************************************/
  
  if(comm->me == evb_engine->rc_rank[icomplex]) {
    // Init three-body system
    
    int m1, m2;
    if(mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
    if(mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
    
    index_A_Rq[DATOM] = map[m1][atom_A_Rq[DATOM]];
    index_A_Rq[AATOM] = map[m2][atom_A_Rq[AATOM]];
    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    
    // Calc dr_DA
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);
    
    // f(R) part
    cal_f_term_sym();
    
    // force
    cal_force_sym(vflag);
  }
}

void EVB_OffDiag_DA_Gaussian::mp_post_compute(int vflag)
{
  // init energy, virial
  if (vflag) memset(virial,0, sizeof(double)*6);
  
  /**************************************************/
  /****** Geometry Force  ***************************/
  /**************************************************/
  
  if(comm->me == evb_engine->rc_rank[icomplex]) {
    index_A_Rq[DATOM] = index[0];
    index_A_Rq[AATOM] = index[1];
    
    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    
    // Calc dr_DA
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);
    
    // f(R) part
    cal_f_term_sym();
    
    // force
    cal_force_sym(vflag);
  }
}
