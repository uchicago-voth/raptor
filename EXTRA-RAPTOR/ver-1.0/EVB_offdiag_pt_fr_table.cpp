/* ----------------------------------------------------------------------
   EVB Package Code
   For Voth Group
   Written by Chris (based on EVB_offdiag_pt and pair_table)
------------------------------------------------------------------------- */

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "math.h"

#include "EVB_complex.h"
#include "EVB_engine.h"
#include "EVB_type.h"
#include "EVB_offdiag_pt_fr_table.h"
#include "EVB_kspace.h"
#include "EVB_source.h"

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
#define HATOM 2

#define LOOKUP 0
#define LINEAR 1  // Currently, the only supported version
#define SPLINE 2
#define BITMAP 3

#define R   1
#define RSQ 2
#define BMP 3

#define MAXLINE 1024


/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_OffDiag_PT_FR_Table::EVB_OffDiag_PT_FR_Table(LAMMPS *lmp, EVB_Engine *engine) : EVB_OffDiag(lmp,engine)
{
  etp_A_exch = etp_B_exch = n_A_exch = n_B_exch = 0;
  q_A_exch = q_B_exch = q_A_save = q_B_save = NULL;
  
  size_exch_chg = 0;
  is_exch_chg = exch_list = NULL;

  ntables = 0;
  tables = NULL;
 
#ifdef OUTPUT_3BODY
  if(comm->me==0)
  {
    fp = fopen("3body.dat","w");
    timestep = center = -1;
  }
#endif 
}

/* ---------------------------------------------------------------------- */

EVB_OffDiag_PT_FR_Table::~EVB_OffDiag_PT_FR_Table()
{
  if (q_A_exch) delete [] q_A_exch;
  if (q_B_exch) delete [] q_B_exch;
  if (q_A_save) delete [] q_A_save;
  if (q_B_save) delete [] q_B_save;

  memory->sfree(is_exch_chg);
  memory->sfree(exch_list);

  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

#ifdef OUTPUT_3BODY
  if(comm->me==0)
  {
    fclose(fp);
  }
#endif
}

/* ---------------------------------------------------------------------- */

int EVB_OffDiag_PT_FR_Table::checkout(int* _index)
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
  if(mol_A_Rq[2]==1) m3 = mol_A; else m3 = mol_B;
        
  _index[0] = map[m1][atom_A_Rq[0]];
  _index[1] = map[m2][atom_A_Rq[1]];
  _index[2] = map[m3][atom_A_Rq[2]];

  return 3;
}

/* ---------------------------------------------------------------------- */

int EVB_OffDiag_PT_FR_Table::data_offdiag(char *buf, int* offset, int start, int end)
{
  int t = start;
  
  FILE * fp = evb_engine->fp_cfg_out;

  // Input atom index for geometry part
  
  // three-body index  
  mol_A_Rq [DATOM] = atoi(buf+offset[t++]);
  atom_A_Rq[DATOM] = atoi(buf+offset[t++]);
  mol_A_Rq [AATOM] = atoi(buf+offset[t++]);  
  atom_A_Rq[AATOM] = atoi(buf+offset[t++]);
  mol_A_Rq [HATOM] = atoi(buf+offset[t++]);
  atom_A_Rq[HATOM] = atoi(buf+offset[t++]);
  
  if(universe->me == 0) {
    fprintf(fp,"   Identity of particle O of hydronium: molecule= %i  index= %i.\n",mol_A_Rq[DATOM],atom_A_Rq[DATOM]);
    fprintf(fp,"   Identity of particle O of water:     molecule= %i  index= %i.\n",mol_A_Rq[AATOM],atom_A_Rq[AATOM]);
    fprintf(fp,"   Identity of particle H+:             molecule= %i  index= %i.\n",mol_A_Rq[HATOM],atom_A_Rq[HATOM]);
  }

  // A_Rq type
  type_A_Rq = atoi(buf+offset[t++]);
  
  if(universe->me == 0) {
    if(type_A_Rq == 1) fprintf(fp,"   Using symmetric coordinate.\n");
    else fprintf(fp,"   Using asymmetric coordinate.\n");
  }

  if(comm->me && screen) fprintf(screen,"[EVB] OffDiag_PT_FR_Table\n");
  ntables = 1;
  
  char *file = buf+offset[t++];
  char *tstyle = buf+offset[t++];
  if(strcmp(tstyle,"LINEAR") == 0) tabstyle = LINEAR;
  else error->all(FLERR,"EVB_OffDiag_PT_FR_Table: Unsupported table style");
  
  if(universe->me == 0) {
    fprintf(fp,"\n   Number of tabulated potentials: ntables= %i.\n",ntables);
    fprintf(fp,"   Name of table potential file: file= %s.\n",file);
    fprintf(fp,"   Interpolation style: tstyle= %s.\n",tstyle);
  }

  tablength = atoi( buf+offset[t++] );
  
  if(universe->me == 0) fprintf(fp,"\n   Number of grid points: tablength= %i.\n\n",tablength);

  char *keyword[1];
  for (int i=0; i<ntables; i++) {
    keyword[i] = buf+offset[t++];
    cutoff[i]  = atof( buf+offset[t++]);
  }
  
  if(universe->me == 0) for(int i=0; i<ntables; i++) fprintf(fp,"   i= %i  keyword= %s  cutoff= %f.\n",i,keyword[i],cutoff[i]);

  // Initialize MS-EVB Tables for atom transfer geometric factor
  MPI_Comm_rank(world,&me);
  
  for (int i=0; i < ntables; i++) {
    tables = (Table *)
      memory -> srealloc(tables, (i+1)*sizeof(Table),"evb_offdiag_pt_fr_table:tables");
    Table *tb = &tables[i];
    null_table(tb);
    if(me==0) read_table(tb,file,keyword[i]);
    bcast_table(tb);
    
    tb->cut = cutoff[i];
    tb->match = 0;
    if (tabstyle == LINEAR && tb->ninput == tablength && 
	tb->rflag == RSQ && tb->rhi == tb->cut) tb->match = 1;
    
    // spline read-in values and compute r,e,f vectors within table
    
    if (tb->match == 0) spline_table(tb);
    compute_table(tb);
  }
  
  if(type_A_Rq == 2) { // Asymmetry type
    _rs  =atof(buf+offset[t++]);
    _l   =atof(buf+offset[t++]);
    _RDA =atof(buf+offset[t++]);

    if(universe->me == 0) fprintf(fp,"\n   Asymmetric coordinate parameters: _rs= %f  _l= %f  _RDA= %f.\n",_rs,_l,_RDA);
  }
  
  // Input Vij information
  Vij_const = atof(buf+offset[t++]);
  is_Vij_ex = atoi(buf+offset[t++]);  
  
  if(universe->me == 0) {
    fprintf(fp,"\n   Off-diagonal definition: VIJ= (VIJ_CONST + V_EX) * A_GEO.\n");
    fprintf(fp,"\n   VIJ_CONST= %f\n",Vij_const);
    fprintf(fp,"\n   Available methods for V_EX: 0= None, 1= K-Space, 2= Debye, 3= Wolf, 4= CGIS, 5= Electrode.\n");
    fprintf(fp,"   Method selected for V_EX: is_Vij_ex= %i (Nonzero values must match across all off-diagonals).\n",is_Vij_ex);
  }

  if(is_Vij_ex) {
    if(is_Vij_ex==2) {
      kappa = atof(buf+offset[t++]);
      evb_engine->flag_DIAG_QEFF = 1;
      if(universe->me == 0) fprintf(fp,"   kappa= %f\n",kappa);
    }

    if(is_Vij_ex==3) {
      kappa = atof(buf+offset[t++]);
      evb_engine->flag_DIAG_QEFF = 1;
      if(universe->me == 0) fprintf(fp,"   kappa= %f\n",kappa);
    }

    if(is_Vij_ex==4) evb_engine->flag_DIAG_QEFF = 1;

    if(is_Vij_ex==5) evb_engine->flag_DIAG_QEFF = 1;
    
    // Input and setup exchange charges
    qsum_exch = qsum_save = qsqsum_exch = qsqsum_save =0.0;
    
    char* type_name;
    char errline[255];
    int type_id;
    
    type_name = buf+offset[t++];
    etp_A_exch = evb_type->get_type(type_name) ;
    
    if(universe->me == 0) fprintf(fp,"   First molecule in exchange complex is %s: type= %i.\n",type_name,etp_A_exch);

    if(etp_A_exch==-1) {
      sprintf(errline,"[EVB] Undefined molecule_type [%s].", type_name);
      error->all(FLERR,errline);
    }
    
    type_name = buf+offset[t++];
    etp_B_exch = evb_type->get_type(type_name) ;
    
    if(universe->me == 0) fprintf(fp,"   Second molecule in exchange complex is %s: type= %i.\n",type_name,etp_B_exch);

    if(etp_B_exch==-1) {
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

void EVB_OffDiag_PT_FR_Table::compute(int vflag)
{
  MP_Verlet *mp_verlet = evb_engine->mp_verlet;

  // set up tables
  Table *tb;
  int tlm1 = tablength - 1;
  int itable;
  double rsq,value,fraction;

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

  if(comm->me == evb_engine->rc_rank[icomplex]) {
    // Init three-body system
    int m1, m2, m3;
    if(mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
    if(mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
    if(mol_A_Rq[2]==1) m3 = mol_A; else m3 = mol_B;
    
    index_A_Rq[DATOM] = map[m1][atom_A_Rq[DATOM]];
    index_A_Rq[AATOM] = map[m2][atom_A_Rq[AATOM]];
    index_A_Rq[HATOM] = map[m3][atom_A_Rq[HATOM]];
    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    x_H = atom->x[index_A_Rq[HATOM]];
    
    // Cal dr_DH, dr_AH, dr_DA
    VECTOR_SUB(dr_DH,x_D,x_H);
    VECTOR_PBC(dr_DH);
    VECTOR_SUB(dr_AH,x_A,x_H);
    VECTOR_PBC(dr_AH);
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);

    // g(q) part
    if(type_A_Rq==1) cal_g_term_sym();
    else if(type_A_Rq==2) cal_g_term_asym();

    // f(R) part
    if(type_A_Rq==1) cal_f_term_sym();
    else if(type_A_Rq==2) cal_f_term_asym();
    
    // A(R,q) = g(q) * f(R)
    A_Rq = g_q * f_R;
    
    //if(type_A_Rq==2) 
    //  fprintf(screen,"%lf %lf %lf\n", g_q, f_R, A_Rq);

    index[0] = index_A_Rq[DATOM];
    index[1] = index_A_Rq[AATOM];
    index[2] = index_A_Rq[HATOM];
  }

  MPI_Bcast(&A_Rq,1,MPI_DOUBLE,evb_engine->rc_rank[icomplex],world);

  }
  
#ifdef OUTPUT_3BODY

  int _A[3], A[3];
  double _B[3], B[2];
  
  _A[0] = _A[1] = _A[2] = 0;
  _B[0] = _B[1] = 0.0;
    
  if(comm->me==0 && update->ntimestep!=timestep) {
    fflush(fp);
    timestep = update->ntimestep;
    fprintf(fp,"TIMESTEP %d\n",timestep);
  }

  if(comm->me == evb_engine->rc_rank[icomplex]) {
    _A[0] = atom->tag[index_A_Rq[DATOM]];
    _A[1] = atom->tag[index_A_Rq[HATOM]];
    _A[2] = atom->tag[index_A_Rq[AATOM]];

    _B[0] = sqrt(dr_DH[0]*dr_DH[0]+dr_DH[1]*dr_DH[1]+dr_DH[2]*dr_DH[2]);
    _B[1] = sqrt(dr_AH[0]*dr_AH[0]+dr_AH[1]*dr_AH[1]+dr_AH[2]*dr_AH[2]);
  }

  MPI_Allreduce(_A,A,3,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(_B,B,2,MPI_DOUBLE,MPI_SUM,world);

  if(comm->me==0) fprintf(fp,"%d %d %d %lf %lf\n",A[0],A[1],A[2],B[0],B[1]);
  
#endif
  
  /**************************************************/
  /****** Potential part ****************************/
  /**************************************************/

  if(mp_verlet && mp_verlet->is_master==0 && !is_Vij_ex) { energy = 0.0; return; }

  if(is_Vij_ex) {
    if(mp_verlet && mp_verlet->is_master==0)
      if(is_Vij_ex > 1 || evb_engine->flag_ACC) return;
    
    // init exchanged charge
    init_exch_chg(); 
    
    if(evb_kspace) {
      if(is_Vij_ex==2) Vij_ex_short = exch_chg_debye(vflag);
      else if(is_Vij_ex==3) Vij_ex_short = exch_chg_wolf(vflag);
      else if(is_Vij_ex==4) Vij_ex_short = exch_chg_cgis(vflag);
      else if(is_Vij_ex==5) {
	Vij_ex_short = 0.0;
	evb_kspace->A_Rq = A_Rq;
        evb_kspace->is_exch_chg = is_exch_chg;
        evb_kspace->compute_exch(vflag);
      } else if(evb_engine->flag_ACC) Vij_ex_short = exch_chg_cut(vflag);
      else {
	if(!mp_verlet || mp_verlet->is_master==1) Vij_ex_short = exch_chg_long(vflag);
	
	if(!mp_verlet) {
	  evb_kspace->A_Rq = A_Rq;
	  evb_kspace->is_exch_chg = is_exch_chg;
	  evb_kspace->compute_exch(vflag);
	} else if(mp_verlet->is_master==0) {
	  evb_kspace->A_Rq = 1.0 ;
	  evb_kspace->is_exch_chg = is_exch_chg;
	  evb_kspace->compute_exch(vflag);
	}
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
    if(comm->me == evb_engine->rc_rank[icomplex]) {
      if (type_A_Rq==1) cal_force_sym(vflag);
      else if (type_A_Rq==2) cal_force_asym(vflag);
    }
  
  /****** force output ******
  FILE *ft = fopen("force","w");
  for(int i=0; i<atom->nlocal; i++)
    fprintf(ft,"%lf %lf %lf\n",atom->f[i][0],atom->f[i][1],atom->f[i][2]); 
  fclose(ft); exit(0);
  /**************************/
  
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

void EVB_OffDiag_PT_FR_Table::cal_g_term_sym()
{
  g_q  = 1.0;
  dg_q = 0.0;
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::cal_f_term_sym()
{
  // set up tables
  Table *tb;
  int tlm1 = tablength - 1;
  int itable;
  double rsq,value,fraction;
  
  // Cal R_OO
  VECTOR_R2(rsq,dr_DA);
  tb = &tables[0];
  if(rsq < tb->innersq) error->one(FLERR,"Geometric Factor distance #1 < table inner cutoff");
  if(tabstyle == LINEAR) {
    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
    if (itable >= tlm1) {
      f_R = 0.0;
      df_R = 0.0;
    } else {
      fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
      f_R  = tb->e[itable] + fraction*tb->de[itable];
      df_R = tb->f[itable] + fraction*tb->df[itable];
    }
  }
  
  df_O = df_R * g_q;
  df_R *= sqrt(rsq);
  df_H = 0.25 * f_R * dg_q;

}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::cal_force_sym(int vflag)
{
  double **f = atom->f;
  
  double tfh = Vij * df_H;
  double tfa = Vij * df_O;
  
  double dfhx, dfax, dfhy, dfay, dfhz, dfaz;
  
  dfhx = tfh * (dr_DH[0]+dr_AH[0]);
  dfax = tfa * (dr_DA[0]);
  f[index_A_Rq[DATOM]][0] += (dfhx + dfax);
  f[index_A_Rq[AATOM]][0] += (dfhx - dfax);
  f[index_A_Rq[HATOM]][0] -= (dfhx + dfhx);
  
  dfhy = tfh * (dr_DH[1]+dr_AH[1]);
  dfay = tfa * (dr_DA[1]);
  f[index_A_Rq[DATOM]][1] += (dfhy + dfay);
  f[index_A_Rq[AATOM]][1] += (dfhy - dfay);
  f[index_A_Rq[HATOM]][1] -= (dfhy + dfhy);
  
  dfhz = tfh * (dr_DH[2]+dr_AH[2]);
  dfaz = tfa * (dr_DA[2]);
  f[index_A_Rq[DATOM]][2] += (dfhz + dfaz);
  f[index_A_Rq[AATOM]][2] += (dfhz - dfaz);
  f[index_A_Rq[HATOM]][2] -= (dfhz + dfhz);
  
  if(vflag) {
    virial[0] += (dfhx * dr_DH[0] + dfhx * dr_AH[0] + dfax * dr_DA[0]);
    virial[1] += (dfhy * dr_DH[1] + dfhy * dr_AH[1] + dfay * dr_DA[1]);
    virial[2] += (dfhz * dr_DH[2] + dfhz * dr_AH[2] + dfaz * dr_DA[2]);
    virial[3] += (dfhx * dr_DH[1] + dfhx * dr_AH[1] + dfax * dr_DA[1]);
    virial[4] += (dfhx * dr_DH[2] + dfhx * dr_AH[2] + dfax * dr_DA[2]);
    virial[5] += (dfhy * dr_DH[2] + dfhy * dr_AH[2] + dfay * dr_DA[2]);
  }
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::cal_g_term_asym()
{
  error->one(FLERR,"EVB_OffDiag_PT_FR_Table::cal_g_term_asym() not yet coded.");
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::cal_f_term_asym()
{
  error->one(FLERR,"EVB_OffDiag_PT_FR_Table::cal_f_term_asym() not yet coded.");
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::cal_force_asym(int vflag)
{
  error->one(FLERR,"EVB_OffDiag_PT_FR_Table::cal_force_asym() not yet coded.");
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::init_exch_chg()
{
  if(natom>size_exch_chg) {
    size_exch_chg = natom;
    is_exch_chg = (int *) memory->srealloc(is_exch_chg,sizeof(int)*natom, "EVB_OffDiag_PT_FR_Table:is_exch_chg");
    exch_list = (int *) memory->srealloc(exch_list,sizeof(int)*natom, "EVB_OffDiag_PT_FR_Table:exch_list");
  }

  memset(is_exch_chg,0,sizeof(int)*natom);
  
  n_exch_chg = n_exch_chg_local = 0;
 
  double *q = atom->q;
  int *molecule = atom->molecule;
  int *mol_index = evb_engine->mol_index;

  int nlocal = atom->nlocal;
  for(int i=0; i<natom; i++) {
    int type = 0;
    if(molecule[i]==mol_A) type = 1;
    else if(molecule[i]==mol_B) type = 2;
    
    if (type) {      
      if(n_exch_chg==max_nexch*27) error->one(FLERR,"[EVB] exch_chg overflow!");

      is_exch_chg[i]=1;
      exch_list[n_exch_chg] = i;
      iexch[n_exch_chg++] = i;
      
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

void EVB_OffDiag_PT_FR_Table::resume_chg()
{
  double *q = atom->q;
  int *molecule = atom->molecule;
  int *mol_index = evb_engine->mol_index;
  int nlocal = atom->nlocal;
  
  for(int i=0; i<n_exch_chg; i++) {
    int id = exch_list[i];
    qexch[i]=q[id]*A_Rq;
        
    if(molecule[id]==mol_A) q[id] = q_A_save[mol_index[id]-1];
    else if(molecule[id]==mol_B) q[id] = q_B_save[mol_index[id]-1];
  }
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::sci_setup(int vflag)
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
  
  if(comm->me == evb_engine->rc_rank[icomplex]) {
    // Init three-body system

    int m1, m2, m3;
    if(mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
    if(mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
    if(mol_A_Rq[2]==1) m3 = mol_A; else m3 = mol_B;
    
    index_A_Rq[DATOM] = map[m1][atom_A_Rq[DATOM]];
    index_A_Rq[AATOM] = map[m2][atom_A_Rq[AATOM]];
    index_A_Rq[HATOM] = map[m3][atom_A_Rq[HATOM]];
    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    x_H = atom->x[index_A_Rq[HATOM]];
    
    // Cal dr_DH, dr_AH, dr_DA
    VECTOR_SUB(dr_DH,x_D,x_H);
    VECTOR_PBC(dr_DH);
    VECTOR_SUB(dr_AH,x_A,x_H);
    VECTOR_PBC(dr_AH);
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);
    
    // g(q) part
    if(type_A_Rq==1) cal_g_term_sym();
    else if(type_A_Rq==2) cal_g_term_asym();
    
    // f(R) part
    if(type_A_Rq==1) cal_f_term_sym();
    else if(type_A_Rq==2) cal_f_term_asym();
    
    // A(R,q) = g(q) * f(R)
    A_Rq = g_q * f_R;
  }
  
  MPI_Bcast(&A_Rq,1,MPI_DOUBLE,evb_engine->rc_rank[icomplex],world);

  /**************************************************/
  /****** Potential part ****************************/
  /**************************************************/

  Vij = Vij_const;

  if(is_Vij_ex) {
    init_exch_chg(); 
 
    if(evb_kspace) {
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

void EVB_OffDiag_PT_FR_Table::sci_setup_mp()
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
    int m1, m2, m3;
    if(mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
    if(mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
    if(mol_A_Rq[2]==1) m3 = mol_A; else m3 = mol_B;
    
    index_A_Rq[DATOM] = map[m1][atom_A_Rq[DATOM]];
    index_A_Rq[AATOM] = map[m2][atom_A_Rq[AATOM]];
    index_A_Rq[HATOM] = map[m3][atom_A_Rq[HATOM]];

    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    x_H = atom->x[index_A_Rq[HATOM]];
    
    // Cal dr_DH, dr_AH, dr_DA
    VECTOR_SUB(dr_DH,x_D,x_H);
    VECTOR_PBC(dr_DH);
    VECTOR_SUB(dr_AH,x_A,x_H);
    VECTOR_PBC(dr_AH);
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);
    
    // g(q) part
    if(type_A_Rq==1) cal_g_term_sym();
    else if(type_A_Rq==2) cal_g_term_asym();
    
    // f(R) part
    if(type_A_Rq==1) cal_f_term_sym();
    else if(type_A_Rq==2) cal_f_term_asym();
    
    // A(R,q) = g(q) * f(R)
    A_Rq = g_q * f_R;
  }
  
  MPI_Bcast(&A_Rq,1,MPI_DOUBLE,evb_engine->rc_rank[icomplex],world);

  // Setup exchange charges
  
  if(is_Vij_ex) {
    init_exch_chg();
    resume_chg();
  }
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::sci_compute(int vflag)
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
    
    int m1, m2, m3;
    if(mol_A_Rq[0]==1) m1 = mol_A; else m1 = mol_B;
    if(mol_A_Rq[1]==1) m2 = mol_A; else m2 = mol_B;
    if(mol_A_Rq[2]==1) m3 = mol_A; else m3 = mol_B;
    
    index_A_Rq[DATOM] = map[m1][atom_A_Rq[DATOM]];
    index_A_Rq[AATOM] = map[m2][atom_A_Rq[AATOM]];
    index_A_Rq[HATOM] = map[m3][atom_A_Rq[HATOM]];
    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    x_H = atom->x[index_A_Rq[HATOM]];
    
    // Cal dr_DH, dr_AH, dr_DA
    VECTOR_SUB(dr_DH,x_D,x_H);
    VECTOR_PBC(dr_DH);
    VECTOR_SUB(dr_AH,x_A,x_H);
    VECTOR_PBC(dr_AH);
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);
    
    // g(q) part
    if(type_A_Rq==1) cal_g_term_sym();
    else if(type_A_Rq==2) cal_g_term_asym();
    
    // f(R) part
    if(type_A_Rq==1) cal_f_term_sym();
    else if(type_A_Rq==2) cal_f_term_asym();
    
    // force
    if (type_A_Rq==1) cal_force_sym(vflag);
    else if (type_A_Rq==2) cal_force_asym(vflag);
  }
}


void EVB_OffDiag_PT_FR_Table::mp_post_compute(int vflag)
{
  // init energy, virial
  if (vflag) memset(virial,0, sizeof(double)*6);
  
  /**************************************************/
  /****** Geometry Force  ***************************/
  /**************************************************/
  
  if(comm->me == evb_engine->rc_rank[icomplex]) {
    index_A_Rq[DATOM] = index[0];
    index_A_Rq[AATOM] = index[1];
    index_A_Rq[HATOM] = index[2];
    
    x_D = atom->x[index_A_Rq[DATOM]];
    x_A = atom->x[index_A_Rq[AATOM]];
    x_H = atom->x[index_A_Rq[HATOM]];
    
    // Cal dr_DH, dr_AH, dr_DA
    VECTOR_SUB(dr_DH,x_D,x_H);
    VECTOR_PBC(dr_DH);
    VECTOR_SUB(dr_AH,x_A,x_H);
    VECTOR_PBC(dr_AH);
    VECTOR_SUB(dr_DA,x_D,x_A);
    VECTOR_PBC(dr_DA);
    
    // g(q) part
    if(type_A_Rq==1) cal_g_term_sym();
    else if(type_A_Rq==2) cal_g_term_asym();
    
    // f(R) part
    if(type_A_Rq==1) cal_f_term_sym();
    else if(type_A_Rq==2) cal_f_term_asym();
    
    // force
    if (type_A_Rq==1) cal_force_sym(vflag);
    else if (type_A_Rq==2) cal_force_asym(vflag);
  }
}






/* ----------------------------------------------------------------------
   set all ptrs in a table to NULL, so can be freed safely
------------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::null_table(Table *tb)
{
  tb->rfile = tb->efile = tb->ffile = NULL;
  tb->e2file = tb->f2file = NULL;
  tb->rsq = tb->drsq = tb->e = tb->de = NULL;
  tb->f = tb->df = tb->e2 = tb->f2 = NULL;
}
/* ----------------------------------------------------------------------*/

void EVB_OffDiag_PT_FR_Table::free_table(Table *tb)
{
  memory->destroy(tb->rfile);
  memory->destroy(tb->efile);
  memory->destroy(tb->ffile);
  memory->destroy(tb->e2file);
  memory->destroy(tb->f2file);

  memory->destroy(tb->rsq);
  memory->destroy(tb->drsq);
  memory->destroy(tb->e);
  memory->destroy(tb->de);
  memory->destroy(tb->f);
  memory->destroy(tb->df);
  memory->destroy(tb->e2);
  memory->destroy(tb->f2);
}

/* ----------------------------------------------------------------------*/

void EVB_OffDiag_PT_FR_Table::read_table(Table *tb, char *file, char *keyword)
{
  char line[MAXLINE];

  // open file

  FILE *fp = fopen(file,"r");
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open file %s",file);
    error->one(FLERR,str);
  }
  
  if(comm->me==0 && screen) fprintf(screen,"[EVB] Looking for keyword: %s",keyword);

  while (1) {
    if (fgets(line,MAXLINE,fp) == NULL)
      error->one(FLERR,"Did not find keyword in table file");
    if (strspn(line," \t\n") == strlen(line)) continue;    // blank line
    if (line[0] == '#') continue;                          // comment
    if (strstr(line,keyword) == line) break;               // matching keyword
    fgets(line,MAXLINE,fp);                         // no match, skip section
    param_extract(tb,line);
    fgets(line,MAXLINE,fp);
    for (int i = 0; i < tb->ninput; i++) fgets(line,MAXLINE,fp);
  }

  // read args on 2nd line of section
  // allocate table arrays for file values

  fgets(line,MAXLINE,fp);
  param_extract(tb,line);
  memory->create(tb->rfile,tb->ninput,"evb_offdiag_pt_table:rfile");
  memory->create(tb->efile,tb->ninput,"evb_offdiag_pt_table:efile");
  memory->create(tb->ffile,tb->ninput,"evb_offdiag_pt_table:ffile");

  // read r,e,f table values from file
  // if rflag set, compute r
  // if rflag not set, use r from file

  int itmp;
  double rtmp;

  fgets(line,MAXLINE,fp);
  for (int i = 0; i < tb->ninput; i++) {
    fgets(line,MAXLINE,fp);
    sscanf(line,"%d %lg %lg %lg",&itmp,&rtmp,&tb->efile[i],&tb->ffile[i]);

    if (tb->rflag == R)
      rtmp = tb->rlo + (tb->rhi - tb->rlo)*i/(tb->ninput-1);
    else if (tb->rflag == RSQ) {
      rtmp = tb->rlo*tb->rlo + 
	(tb->rhi*tb->rhi - tb->rlo*tb->rlo)*i/(tb->ninput-1);
      rtmp = sqrt(rtmp);
    }

    tb->rfile[i] = rtmp;
  }

  // close file

  fclose(fp);
}

/* ----------------------------------------------------------------------
   broadcast read-in table info from proc 0 to other procs
   this function communicates these values in Table:
     ninput,rfile,efile,ffile,rflag,rlo,rhi,fpflag,fplo,fphi
------------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::bcast_table(Table *tb)
{
  MPI_Bcast(&tb->ninput,1,MPI_INT,0,world);

  int me;
  MPI_Comm_rank(world,&me);
  if (me > 0) {
    memory->create(tb->rfile,tb->ninput,"evb_offdiag_pt_table:rfile");
    memory->create(tb->efile,tb->ninput,"evb_offdiag_pt_table:efile");
    memory->create(tb->ffile,tb->ninput,"evb_offdiag_pt_table:ffile");
  }

  MPI_Bcast(tb->rfile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->efile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->ffile,tb->ninput,MPI_DOUBLE,0,world);

  MPI_Bcast(&tb->rflag,1,MPI_INT,0,world);
  if (tb->rflag) {
    MPI_Bcast(&tb->rlo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->rhi,1,MPI_DOUBLE,0,world);
  }
  MPI_Bcast(&tb->fpflag,1,MPI_INT,0,world);
  if (tb->fpflag) {
    MPI_Bcast(&tb->fplo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->fphi,1,MPI_DOUBLE,0,world);
  }
}

/* ----------------------------------------------------------------------
   extract attributes from parameter line in table section
   format of line: N value R/RSQ/BITMAP lo hi FP fplo fphi
   N is required, other params are optional
------------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::param_extract(Table *tb, char *line)
{
  tb->ninput = 0;
  tb->rflag = 0;
  tb->fpflag = 0;
  
  char *word = strtok(line," \t\n\r\f");
  while (word) {
    if (strcmp(word,"N") == 0) {
      word = strtok(NULL," \t\n\r\f");
      tb->ninput = atoi(word);
    } else if (strcmp(word,"R") == 0 || strcmp(word,"RSQ") == 0 ||
	       strcmp(word,"BITMAP") == 0) {
      if (strcmp(word,"R") == 0) tb->rflag = R;
      else if (strcmp(word,"RSQ") == 0) tb->rflag = RSQ;
      else if (strcmp(word,"BITMAP") == 0) tb->rflag = BMP;
      word = strtok(NULL," \t\n\r\f");
      tb->rlo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->rhi = atof(word);
    } else if (strcmp(word,"FP") == 0) {
      tb->fpflag = 1;
      word = strtok(NULL," \t\n\r\f");
      tb->fplo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->fphi = atof(word);
    } else {
      printf("WORD: %s\n",word);
      error->one(FLERR,"Invalid keyword in pair table parameters");
    }
    word = strtok(NULL," \t\n\r\f");
  }

  if (tb->ninput == 0) error->one(FLERR,"Pair table parameters did not set N");
}

/* ----------------------------------------------------------------------
   compute r,e,f vectors from splined values
------------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::compute_table(Table *tb)
{
  if(comm->me==0 && screen) fprintf(screen,"  Computing Table\n");
  double tol_zero = 0.0000001; // For when the grid point 0.0 is included in table.
  bool test_zero = false;
  int tlm1 = tablength-1;

  // inner = inner table bound
  // cut = outer table bound
  // delta = table spacing in rsq for N-1 bins

  double inner;
  if (tb->rflag) inner = tb->rlo;
  else inner = tb->rfile[0];
  tb->innersq = inner*inner;
  tb->delta = (tb->cut*tb->cut - tb->innersq) / tlm1;
  tb->invdelta = 1.0/tb->delta;

  // linear tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // de,df values = delta from lower edge to upper edge of bin
  // rsq,e,f are N in length so de,df arrays can compute difference
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == LINEAR) {
    memory->create(tb->rsq,tablength,"evb_offdiag_pt_table:rsq");
    memory->create(tb->e,  tablength,"evb_offdiag_pt_table:e");
    memory->create(tb->f,  tablength,"evb_offdiag_pt_table:f");
    memory->create(tb->de, tlm1,"evb_offdiag_pt_table:de");
    memory->create(tb->df, tlm1,"evb_offdiag_pt_table:df");

    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
	tb->e[i] = tb->efile[i];
	if(r < tol_zero) test_zero = true;
	else tb->f[i] = tb->ffile[i]/r;
      } else {
	tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
	if(r < tol_zero) test_zero = true;
	else tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
    }
    
    for (int i = 0; i < tlm1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
    }
  }

} 

/* ----------------------------------------------------------------------
   spline and splint routines modified from Numerical Recipes
------------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::spline(double *x, double *y, int n,
		       double yp1, double ypn, double *y2)
{
  int i,k;
  double p,qn,sig,un;
  double *u = new double[n];

  if (yp1 > 0.99e30) y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0]) / (x[1]-x[0]) - yp1);
  }
  for (i = 1; i < n-1; i++) {
    sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
    p = sig*y2[i-1] + 2.0;
    y2[i] = (sig-1.0) / p;
    u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1]);
    u[i] = (6.0*u[i] / (x[i+1]-x[i-1]) - sig*u[i-1]) / p;
  }
  if (ypn > 0.99e30) qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0/(x[n-1]-x[n-2])) * (ypn - (y[n-1]-y[n-2]) / (x[n-1]-x[n-2]));
  }
  y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2] + 1.0);
  for (k = n-2; k >= 0; k--) y2[k] = y2[k]*y2[k+1] + u[k];

  delete [] u;
}

/* ---------------------------------------------------------------------- */

double EVB_OffDiag_PT_FR_Table::splint(double *xa, double *ya, double *y2a, int n, double x)
{
  int klo,khi,k;
  double h,b,a,y;

  klo = 0;
  khi = n-1;
  while (khi-klo > 1) {
    k = (khi+klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  h = xa[khi]-xa[klo];
  a = (xa[khi]-x) / h;
  b = (x-xa[klo]) / h;
  y = a*ya[klo] + b*ya[khi] + 
    ((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi]) * (h*h)/6.0;
  return y;
}

/* ----------------------------------------------------------------------
   build spline representation of e,f over entire range of read-in table
   this function sets these values in Table: e2file,f2file
------------------------------------------------------------------------- */

void EVB_OffDiag_PT_FR_Table::spline_table(Table *tb)
{
  memory->create(tb->e2file,tb->ninput,"evb_offdiag_pt_table:e2file");
  memory->create(tb->f2file,tb->ninput,"evb_offdiag_pt_table:f2file");

  double ep0 = - tb->ffile[0];
  double epn = - tb->ffile[tb->ninput-1];
  spline(tb->rfile,tb->efile,tb->ninput,ep0,epn,tb->e2file);

  if (tb->fpflag == 0) {
    tb->fplo = (tb->ffile[1] - tb->ffile[0]) / (tb->rfile[1] - tb->rfile[0]);
    tb->fphi = (tb->ffile[tb->ninput-1] - tb->ffile[tb->ninput-2]) / 
      (tb->rfile[tb->ninput-1] - tb->rfile[tb->ninput-2]);
  }

  double fp0 = tb->fplo;
  double fpn = tb->fphi;
  spline(tb->rfile,tb->ffile,tb->ninput,fp0,fpn,tb->f2file);
}
