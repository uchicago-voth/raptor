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
#include "EVB_offdiag_pt.h"
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
#define HATOM 2

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

EVB_OffDiag_PT::EVB_OffDiag_PT(LAMMPS *lmp, EVB_Engine *engine) : EVB_OffDiag(lmp,engine)
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

EVB_OffDiag_PT::~EVB_OffDiag_PT()
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

int EVB_OffDiag_PT::checkout(int* _index)
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

int EVB_OffDiag_PT::data_offdiag(char *buf, int* offset, int start, int end)
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

  if(type_A_Rq == 1) { // Symmetry type
    _g =atof(buf+offset[t++]);
    _P0=atof(buf+offset[t++]);
    _k =atof(buf+offset[t++]);
    _D =atof(buf+offset[t++]);
    _b =atof(buf+offset[t++]);
    _R =atof(buf+offset[t++]);
    _P1=atof(buf+offset[t++]);
    _a =atof(buf+offset[t++]);
    _r =atof(buf+offset[t++]);

    if(universe->me == 0) {
      fprintf(fp,"\n   A_GEO(R,q) = f(R) * g(q)\n              = f_G(R) * [f_T(R) + f_E(R)] * g(q)\n");
      fprintf(fp,"\n   q = 0.5 * (RO1 + RO2) - R_H+\n");
      fprintf(fp,"\n   g(q) = e^-(g * q^2)\n");
      fprintf(fp,"      g= %f\n",_g);
      fprintf(fp,"\n   f_G(R) = 1 + P0 * e^-(k * (R - D))\n");
      fprintf(fp,"      P0= %f\n      k= %f\n      D= %f\n",_P0,_k,_D);
      fprintf(fp,"\n   f_T(R) = 0.5 * (1 - tanh[B * (R - R0)])\n");
      fprintf(fp,"      B= %f\n      R0= %f\n",_b,_R);
      fprintf(fp,"\n   f_E(R) = P1 * e^(-a * (R - r))\n");
      fprintf(fp,"      P1= %f\n      a= %f\n      r= %f\n",_P1,_a,_r);
    }
  } else if(type_A_Rq == 2) { // Asymmetry type
    _rs  =atof(buf+offset[t++]);
    _l   =atof(buf+offset[t++]);
    _RDA =atof(buf+offset[t++]);
    _C   =atof(buf+offset[t++]);
    _a   =atof(buf+offset[t++]);
    _aDA =atof(buf+offset[t++]);
    _b   =atof(buf+offset[t++]);
    _bDA =atof(buf+offset[t++]);
    _eps =atof(buf+offset[t++]);
    _cDA =atof(buf+offset[t++]);
    _ga  =atof(buf+offset[t++]);

    if(universe->me == 0) {
      fprintf(fp,"\n   A_GEO(R,q) = f(R) * g(q)\n              = [f_G1(R) + f_G2(R)] * f_T(R) * g(q)\n");
      fprintf(fp,"\n   q = R_DH+ - 0.5 * rsc * R_DA\n");
      fprintf(fp,"   rsc = rsc_0 - l * (R_DA - R_DA0)");
      fprintf(fp,"\n   g(q) = e^-(g * q^2)\n");
      fprintf(fp,"      g= %f\n      rsc_0= %f\n      l= %f      R_DA0= %f\n",_ga,_rs,_l,_RDA);
      fprintf(fp,"\n   f_G1(R) = C * e^-(a * (R - aDA)^2)\n");
      fprintf(fp,"      C= %f\n      a= %f\n      aDA= %f\n",_C,_a,_aDA);
      fprintf(fp,"\n   f_G2(R) = (1-C) * e^(-b * (R - bDA)^2)\n");
      fprintf(fp,"      b= %f\n      bDA= %f\n",_b,_bDA);
      fprintf(fp,"\n   f_T(R) = 1 + tanh[eps * (R - cDA)])\n");
      fprintf(fp,"      eps= %f\n      cDA= %f\n",_eps,_cDA);
    }
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

void EVB_OffDiag_PT::compute(int vflag)
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
    
  if(comm->me==0 && update->ntimestep!=timestep)
  {
    fflush(fp);
    timestep = update->ntimestep;
    fprintf(fp,"TIMESTEP %d\n",timestep);
  }

  if(comm->me == evb_engine->rc_rank[icomplex])
  {
    _A[0] = atom->tag[index_A_Rq[DATOM]];
    _A[1] = atom->tag[index_A_Rq[HATOM]];
    _A[2] = atom->tag[index_A_Rq[AATOM]];

    _B[0] = sqrt(dr_DH[0]*dr_DH[0]+dr_DH[1]*dr_DH[1]+dr_DH[2]*dr_DH[2]);
    _B[1] = sqrt(dr_AH[0]*dr_AH[0]+dr_AH[1]*dr_AH[1]+dr_AH[2]*dr_AH[2]);
  }

  MPI_Allreduce(_A,A,3,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(_B,B,2,MPI_DOUBLE,MPI_SUM,world);

  if(comm->me==0)
      fprintf(fp,"%d %d %d %lf %lf\n",A[0],A[1],A[2],B[0],B[1]);
  
  #endif

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
      else if(is_Vij_ex==5) // User defined exchange-charge interaction
      {
        /* Why is this part here? 
      
        In the normal calculation of off-diagonal, the computing task is done in this EVB_Offdiag module,
	as we do the normal pair-wise model. However when dealing with the Electrode model I found that
	sometimes the off-diagonal exchange-charge part has the cumstomed part that can not be directly
	recognized by the Raptor.
	
	For Raptor, the EVB_Kspace module is defined as porting to MSEVB, however, no EVB_Pair exists. One
	mechanism is using EVB_Kspace as the bridge to linking the user-defined exchange-part in pair_style
	with the Offdiagonal-exchage-charge part.
	
	Note: EVB_OffDiag::compute() ---> EVB_Kspace::compute_exchange() ---> Pair::compute_exchange()
	The energy will be stored in EVB_Kspace::off_diag_energy.
	
	*/
	
	Vij_ex_short = 0.0;
	evb_kspace->A_Rq = A_Rq;
        evb_kspace->is_exch_chg = is_exch_chg;
        evb_kspace->compute_exch(vflag);
      }
      else 
      {
TIMER_STAMP(EVB_OffDiag_PT, compute__evb_kspace_compute_exch);

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

TIMER_CLICK(EVB_OffDiag_PT, compute__evb_kspace_compute_exch);
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
  //fprintf(screen,"OFF-DIAG: %lf %lf %lf %lf\n",evb_kspace->off_diag_energy,Vij,A_Rq);
  
  // local virial + kspace virial devided by total number of cpu's

  if(!mp_verlet || mp_verlet->is_master==0) 
  if (evb_kspace && vflag) 
    for (int i = 0; i < 6; i++)
      virial[i] += (evb_kspace->off_diag_virial[i] / comm->nprocs);

  //for(int i=0; i<atom->nlocal; i++) fprintf(screen,"%lf %lf %lf\n",atom->f[i][0],atom->f[i][1],atom->f[i][2]);
  //exit(0);
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT::cal_g_term_sym()
{
  double q[3];
  double r_q, r2_q;
  
  // Cal q
  VECTOR_ADD(q,dr_DH,dr_AH);
  q[0]/=2.0; q[1]/=2.0; q[2]/=2.0;
  
  // PBC modification of q
  VECTOR_PBC(q);
  
  // Cal |q| and |q|^2
  VECTOR_R2(r2_q,q);
  r_q  = sqrt(r2_q);
  
  // Cal g(q) = exp(-g*q^2)
  g_q = exp(-_g*r2_q); 
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT::cal_f_term_sym()
{
  double r2_da, r_da;
  
  // Cal R_OO
  VECTOR_R2(r2_da,dr_DA);
  r_da = sqrt(r2_da);
  
  // First part: { 1+P*exp[-k(R_OO-D_OO)^2] }
  double dr1 = r_da-_D;
  double part12 = _P0 * exp(-_k*dr1*dr1);
  double part1 = 1.0+ part12;
  
  // Second part: { 0.5*[1-tanh(b(R_OO-R0_OO))] + P'*exp[-a(R_OO-r0_OO)] }
  double dr2 = r_da-_R;
  double b_dr2 = _b*dr2;
  double dr3 = r_da-_r;
  double part22 = _P1*exp(-_a*dr3);
  double part2 = 0.5 *(1-tanh(b_dr2)) + part22;
  
  // f(R)
  f_R = part1 * part2;
  
  // Derivative of f(R)
  df_R = part2 * 2.0 * _k * dr1 * part12 +
         part1 * (0.5 * _b / cosh(b_dr2) / cosh(b_dr2) + _a * part22);
	
  df_O = df_R * g_q / r_da;
  df_H = 0.5 * f_R * _g * g_q;
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT::cal_force_sym(int vflag)
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
    
    if(vflag)
    {
        virial[0] += (dfhx * dr_DH[0] + dfhx * dr_AH[0] + dfax * dr_DA[0]);
        virial[1] += (dfhy * dr_DH[1] + dfhy * dr_AH[1] + dfay * dr_DA[1]);
        virial[2] += (dfhz * dr_DH[2] + dfhz * dr_AH[2] + dfaz * dr_DA[2]);
        virial[3] += (dfhx * dr_DH[1] + dfhx * dr_AH[1] + dfax * dr_DA[1]);
        virial[4] += (dfhx * dr_DH[2] + dfhx * dr_AH[2] + dfax * dr_DA[2]);
        virial[5] += (dfhy * dr_DH[2] + dfhy * dr_AH[2] + dfay * dr_DA[2]);
    }
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT::cal_g_term_asym()
{
    double _t;
    double r2_da;
    double q2;

    for(int i=0; i<3; i++)
    {
        dr_DH[i] = -dr_DH[i];
        dr_DA[i] = -dr_DA[i];
    }
    
    // r_sc
    VECTOR_R2(r2_da,dr_DA);
    r_da = sqrt(r2_da);
    r_sc = _rs - _l * ( r_da - _RDA );
    
    // q
    _t = r_sc*0.5;
    VECTOR_COPY(q,dr_DH);
    VECTOR_SCALE_SUB(q,dr_DA,_t);
    VECTOR_R2(q2,q);
    sumq = _l*(q[0]*dr_DA[0]+q[1]*dr_DA[1]+q[2]*dr_DA[2])/r_da;

    // g(q)
    g_q = exp(-_ga*q2);

    //fprintf(screen,"q2 %lf\n", q2);
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT::cal_f_term_asym()
{
    // part A
    fa = _C * exp ( -_a*(r_da-_aDA)*(r_da-_aDA) );
    
    // part B
    fb = (1.0-_C) * exp ( -_b*(r_da-_bDA)*(r_da-_bDA) );

    // part C
    ftanh = 1.0 + tanh( _eps * (r_da-_cDA) );
    
    // f(R)
    f_R = (fa+fb) * ftanh;
    
    //fprintf(screen,"r_DA %lf\n", r_da);
}

/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT::cal_force_asym(int vflag)
{
    double **f = atom->f;
    double fH, fA, fD;
    
    // Derivative of g(q)

    double _t0 = Vij * f_R * g_q * _ga;
    double _t1 = _t0 * 2.0;
    double _t2 = _t0 * (-r_sc);
    double _t3 = _t0 * sumq;
    
    // x
    fH = _t1*q[0];
    fA = _t2*q[0]+_t3*dr_DA[0];
    f[index_A_Rq[DATOM]][0] -= (fH + fA);
    f[index_A_Rq[AATOM]][0] += fA;
    f[index_A_Rq[HATOM]][0] += fH;

    // y
    fH = _t1*q[1];
    fA = _t2*q[1]+_t3*dr_DA[1];
    f[index_A_Rq[DATOM]][1] -= (fH + fA);
    f[index_A_Rq[AATOM]][1] += fA;
    f[index_A_Rq[HATOM]][1] += fH;

    // z
    fH = _t1*q[2];
    fA = _t2*q[2]+_t3*dr_DA[2];
    f[index_A_Rq[DATOM]][2] -= (fH + fA);
    f[index_A_Rq[AATOM]][2] += fA;
    f[index_A_Rq[HATOM]][2] += fH;

    // Devrivative of f(R)

    _t0 =  -Vij * g_q;
    _t1 = _t0 * ftanh * (fa * (-2.0*_a) * (r_da-_aDA) + fb * (-2.0*_b) * (r_da-_bDA));
    _t2 = cosh(_eps * (r_da-_cDA));
    _t3 = (_t0 * (fa+fb) / _t2 / _t2 * _eps + _t1) / r_da;

    // x
    fA = _t3 * dr_DA[0];
    f[index_A_Rq[DATOM]][0] -= fA;
    f[index_A_Rq[AATOM]][0] += fA;

    // y
    fA = _t3 * dr_DA[1];
    f[index_A_Rq[DATOM]][1] -= fA;
    f[index_A_Rq[AATOM]][1] += fA;

    // z
    fA = _t3 * dr_DA[2];
    f[index_A_Rq[DATOM]][2] -= fA;
    f[index_A_Rq[AATOM]][2] += fA;
}

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

void EVB_OffDiag_PT::init_exch_chg()
{
  if(natom>size_exch_chg)
  {
    size_exch_chg = natom;
    is_exch_chg = (int *) memory->srealloc(is_exch_chg,sizeof(int)*natom, "EVB_OffDiag_PT:is_exch_chg");
    exch_list = (int *) memory->srealloc(exch_list,sizeof(int)*natom, "EVB_OffDiag_PT:exch_list");
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

void EVB_OffDiag_PT::resume_chg()
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

void EVB_OffDiag_PT::sci_setup(int vflag)
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

void EVB_OffDiag_PT::sci_setup_mp()
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

void EVB_OffDiag_PT::sci_compute(int vflag)
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
  
  if(comm->me == evb_engine->rc_rank[icomplex])
    {
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

void EVB_OffDiag_PT::mp_post_compute(int vflag)
{
  // init energy, virial
  if (vflag) memset(virial,0, sizeof(double)*6);
  
  /**************************************************/
  /****** Geometry Force  ***************************/
  /**************************************************/
  
  if(comm->me == evb_engine->rc_rank[icomplex])
    {
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
