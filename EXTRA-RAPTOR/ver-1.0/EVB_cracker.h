#define LAMMPS_VERSION "29 Sep 2021"

#ifdef _CRACKER_KSPACE
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_KSPACE_H
#define LMP_KSPACE_H

#include "pointers.h"    // IWYU pragma: export

#ifdef FFT_SINGLE
typedef float FFT_SCALAR;
#define MPI_FFT_SCALAR MPI_FLOAT
#else
typedef double FFT_SCALAR;
#define MPI_FFT_SCALAR MPI_DOUBLE
#endif

namespace LAMMPS_NS {

class KSpace : protected Pointers {
  friend class ThrOMP;
  friend class FixOMP;

 public:
  double energy;    // accumulated energies
  double energy_1, energy_6;
  double virial[6];          // accumulated virial: xx,yy,zz,xy,xz,yz
  double *eatom, **vatom;    // accumulated per-atom energy/virial
  double e2group;            // accumulated group-group energy
  double f2group[3];         // accumulated group-group force
  int triclinic_support;     // 1 if supports triclinic geometries

  int ewaldflag;         // 1 if a Ewald solver
  int pppmflag;          // 1 if a PPPM solver
  int msmflag;           // 1 if a MSM solver
  int dispersionflag;    // 1 if a LJ/dispersion solver
  int tip4pflag;         // 1 if a TIP4P solver
  int dipoleflag;        // 1 if a dipole solver
  int spinflag;          // 1 if a spin solver
  int differentiation_flag;
  int neighrequest_flag;    // used to avoid obsolete construction
                            // of neighbor lists
  int mixflag;              // 1 if geometric mixing rules are enforced
                            // for LJ coefficients
  int slabflag;
  int scalar_pressure_flag;    // 1 if using MSM fast scalar pressure
  double slab_volfactor;

  int warn_nonneutral;    // 0 = error if non-neutral system
                          // 1 = warn once if non-neutral system
                          // 2 = warn, but already warned
  int warn_nocharge;      // 0 = already warned
                          // 1 = warn if zero charge

  int order, order_6, order_allocated;
  double accuracy;             // accuracy of KSpace solver (force units)
  double accuracy_absolute;    // user-specified accuracy in force units
  double accuracy_relative;    // user-specified dimensionless accuracy
                               // accuracy = acc_rel * two_charge_force
  double accuracy_real_6;      // real space accuracy for
                               // dispersion solver (force units)
  double accuracy_kspace_6;    // reciprocal space accuracy for
                               // dispersion solver (force units)
  int auto_disp_flag;          // use automatic parameter generation for pppm/disp
  double two_charge_force;     // force in user units of two point
                               // charges separated by 1 Angstrom

  double g_ewald, g_ewald_6;
  int nx_pppm, ny_pppm, nz_pppm;          // global FFT grid for Coulombics
  int nx_pppm_6, ny_pppm_6, nz_pppm_6;    // global FFT grid for dispersion
  int nx_msm_max, ny_msm_max, nz_msm_max;

  int group_group_enable;    // 1 if style supports group/group calculation

  int centroidstressflag;    // centroid stress compared to two-body stress
                             // CENTROID_SAME = same as two-body stress
                             // CENTROID_AVAIL = different and implemented
                             // CENTROID_NOTAVAIL = different, not yet implemented

  // KOKKOS host/device flag and data masks

  ExecutionSpace execution_space;
  unsigned int datamask_read, datamask_modify;
  int copymode;

  int compute_flag;       // 0 if skip compute()
  int fftbench;           // 0 if skip FFT timing
  int collective_flag;    // 1 if use MPI collectives for FFT/remap
  int stagger_flag;       // 1 if using staggered PPPM grids

  double splittol;    // tolerance for when to truncate splitting

  KSpace(class LAMMPS *);
  virtual ~KSpace();
  void two_charge();
  void triclinic_check();
  void modify_params(int, char **);
  void *extract(const char *);
  void compute_dummy(int, int);

  // triclinic

  void x2lamdaT(double *, double *);
  void lamda2xT(double *, double *);
  void lamda2xvector(double *, double *);
  void kspacebbox(double, double *);

  // public so can be called by commands that change charge

  void qsum_qsq(int warning_flag = 1);

  // general child-class methods

  virtual void settings(int, char **){};
  virtual void init() = 0;
  virtual void setup() = 0;
  virtual void setup_grid(){};
  virtual void compute(int, int) = 0;
  virtual void compute_group_group(int, int, int){};

  virtual void pack_forward_grid(int, void *, int, int *){};
  virtual void unpack_forward_grid(int, void *, int, int *){};
  virtual void pack_reverse_grid(int, void *, int, int *){};
  virtual void unpack_reverse_grid(int, void *, int, int *){};

  virtual int timing(int, double &, double &) { return 0; }
  virtual int timing_1d(int, double &) { return 0; }
  virtual int timing_3d(int, double &) { return 0; }

  virtual int modify_param(int, char **) { return 0; }
  virtual double memory_usage() { return 0.0; }

  /* ----------------------------------------------------------------------
   compute gamma for MSM and pair styles
   see Eq 4 from Parallel Computing 35 (2009) 164-177
------------------------------------------------------------------------- */

  double gamma(const double &rho) const
  {
    if (rho <= 1.0) {
      const int split_order = order / 2;
      const double rho2 = rho * rho;
      double g = gcons[split_order][0];
      double rho_n = rho2;
      for (int n = 1; n <= split_order; n++) {
        g += gcons[split_order][n] * rho_n;
        rho_n *= rho2;
      }
      return g;
    } else
      return (1.0 / rho);
  }

  /* ----------------------------------------------------------------------
   compute the derivative of gamma for MSM and pair styles
   see Eq 4 from Parallel Computing 35 (2009) 164-177
------------------------------------------------------------------------- */

  double dgamma(const double &rho) const
  {
    if (rho <= 1.0) {
      const int split_order = order / 2;
      const double rho2 = rho * rho;
      double dg = dgcons[split_order][0] * rho;
      double rho_n = rho * rho2;
      for (int n = 1; n < split_order; n++) {
        dg += dgcons[split_order][n] * rho_n;
        rho_n *= rho2;
      }
      return dg;
    } else
      return (-1.0 / rho / rho);
  }

  double **get_gcons() { return gcons; }
  double **get_dgcons() { return dgcons; }

 public:
  int gridflag, gridflag_6;
  int gewaldflag, gewaldflag_6;
  int minorder, overlap_allowed;
  int adjust_cutoff_flag;
  int suffix_flag;    // suffix compatibility flag
  bigint natoms_original;
  double scale, qqrd2e;
  double qsum, qsqsum, q2;
  double **gcons, **dgcons;    // accumulated per-atom energy/virial

  int evflag, evflag_atom;
  int eflag_either, eflag_global, eflag_atom;
  int vflag_either, vflag_global, vflag_atom;
  int maxeatom, maxvatom;

  int kewaldflag;                      // 1 if kspace range set for Ewald sum
  int kx_ewald, ky_ewald, kz_ewald;    // kspace settings for Ewald sum

  void pair_check();
  void ev_init(int eflag, int vflag, int alloc = 1)
  {
    if (eflag || vflag)
      ev_setup(eflag, vflag, alloc);
    else
      evflag = evflag_atom = eflag_either = eflag_global = eflag_atom = vflag_either =
          vflag_global = vflag_atom = 0;
  }
  void ev_setup(int, int, int alloc = 1);
  double estimate_table_accuracy(double, double);
};

}    // namespace LAMMPS_NS

#endif

/* ERROR/WARNING messages:

E: KSpace style does not yet support triclinic geometries

The specified kspace style does not allow for non-orthogonal
simulation boxes.

E: KSpace solver requires a pair style

No pair style is defined.

E: KSpace style is incompatible with Pair style

Setting a kspace style requires that a pair style with matching
long-range Coulombic or dispersion components be used.

W: Using kspace solver on system with no charge

Self-explanatory.

E: System is not charge neutral, net charge = %g

The total charge on all atoms on the system is not 0.0.
For some KSpace solvers this is an error.

W: System is not charge neutral, net charge = %g

The total charge on all atoms on the system is not 0.0.
For some KSpace solvers this is only a warning.

W: For better accuracy use 'pair_modify table 0'

The user-specified force accuracy cannot be achieved unless the table
feature is disabled by using 'pair_modify table 0'.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Bad kspace_modify slab parameter

Kspace_modify value for the slab/volume keyword must be >= 2.0.

E: Kspace_modify mesh parameter must be all zero or all positive

Valid kspace mesh parameters are >0. The code will try to auto-detect
suitable values when all three mesh sizes are set to zero (the default).

E: Kspace_modify mesh/disp parameter must be all zero or all positive

Valid kspace mesh/disp parameters are >0. The code will try to auto-detect
suitable values when all three mesh sizes are set to zero [and]
the required accuracy via {force/disp/real} as well as
{force/disp/kspace} is set.

W: Kspace_modify slab param < 2.0 may cause unphysical behavior

The kspace_modify slab parameter should be larger to insure periodic
grids padded with empty space do not overlap.

E: Bad kspace_modify kmax/ewald parameter

Kspace_modify values for the kmax/ewald keyword must be integers > 0

E: Kspace_modify eigtol must be smaller than one

Self-explanatory.

*/
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_GRIDCOMM
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_GRIDCOMM_H
#define LMP_GRIDCOMM_H

#include "pointers.h"

namespace LAMMPS_NS {

class GridComm : protected Pointers {
 public:
  enum { KSPACE = 0, FIX = 1 };    // calling classes

  GridComm(class LAMMPS *, MPI_Comm, int, int, int, int, int, int, int, int, int, int, int, int,
           int, int, int);
  GridComm(class LAMMPS *, MPI_Comm, int, int, int, int, int, int, int, int, int, int, int, int,
           int, int, int, int, int, int, int, int, int, int);
  virtual ~GridComm();
  void setup(int &, int &);
  int ghost_adjacent();
  void forward_comm(int, void *, int, int, int, void *, void *, MPI_Datatype);
  void reverse_comm(int, void *, int, int, int, void *, void *, MPI_Datatype);
  void gather(int, void *, int, int, int, void *, MPI_Datatype);

 public:
  int me, nprocs;
  int layout;           // REGULAR or TILED
  MPI_Comm gridcomm;    // communicator for this class
                        // usually world, but MSM calls with subset

  // inputs from caller via constructor

  int nx, ny, nz;      // size of global grid in all 3 dims
  int inxlo, inxhi;    // inclusive extent of my grid chunk
  int inylo, inyhi;    //   0 <= in <= N-1
  int inzlo, inzhi;
  int outxlo, outxhi;      // inclusive extent of my grid chunk plus
  int outylo, outyhi;      //   ghost cells in all 6 directions
  int outzlo, outzhi;      //   lo indices can be < 0, hi indices can be >= N
  int fullxlo, fullxhi;    // extent of grid chunk that caller stores
  int fullylo, fullyhi;    //   can be same as out indices or larger
  int fullzlo, fullzhi;

  // -------------------------------------------
  // internal variables for REGULAR layout
  // -------------------------------------------

  int procxlo, procxhi;    // 6 neighbor procs that adjoin me
  int procylo, procyhi;    // not used for comm_style = tiled
  int proczlo, proczhi;

  int ghostxlo, ghostxhi;    // # of my owned grid planes needed
  int ghostylo, ghostyhi;    // by neighobr procs in each dir as their ghost planes
  int ghostzlo, ghostzhi;

  // swap = exchange of owned and ghost grid cells between 2 procs, including self

  struct Swap {
    int sendproc;       // proc to send to for forward comm
    int recvproc;       // proc to recv from for forward comm
    int npack;          // # of datums to pack
    int nunpack;        // # of datums to unpack
    int *packlist;      // 3d array offsets to pack
    int *unpacklist;    // 3d array offsets to unpack
  };

  int nswap, maxswap;
  Swap *swap;

  // -------------------------------------------
  // internal variables for TILED layout
  // -------------------------------------------

  int *overlap_procs;       // length of Nprocs in communicator
  MPI_Request *requests;    // length of max messages this proc receives

  // RCB tree of cut info
  // each proc contributes one value, except proc 0

  struct RCBinfo {
    int dim;    // 0,1,2 = which dim the cut is in
    int cut;    // grid index of lowest cell in upper half of cut
  };

  RCBinfo *rcbinfo;

  // overlap = a proc whose owned cells overlap with my extended ghost box
  // includes overlaps across periodic boundaries, can also be self

  struct Overlap {
    int proc;      // proc whose owned cells overlap my ghost cells
    int box[6];    // box that overlaps otherproc's owned cells
                   // this box is wholly contained within global grid
    int pbc[3];    // PBC offsets to convert box to a portion of my ghost box
                   // my ghost box may extend beyond global grid
  };

  int noverlap, maxoverlap;
  Overlap *overlap;

  // request = sent to each proc whose owned cells overlap my ghost cells

  struct Request {
    int sender;    // sending proc
    int index;     // index of overlap on sender
    int box[6];    // box that overlaps receiver's owned cells
                   // wholly contained within global grid
  };

  Request *srequest, *rrequest;

  // response = reply from each proc whose owned cells overlap my ghost cells

  struct Response {
    int index;     // index of my overlap for the initial request
    int box[6];    // box that overlaps responder's owned cells
                   // wholly contained within global grid
                   // has to unwrapped by PBC to map to my ghost cells
  };

  Response *sresponse, *rresponse;

  // send = proc to send a subset of my owned cells to, for forward comm
  // for reverse comm, proc I receive ghost overlaps with my owned cells from
  // offset used in reverse comm to recv a message in middle of a large buffer

  struct Send {
    int proc;
    int npack;
    int *packlist;
    int offset;
  };

  // recv = proc to recv a subset of my ghost cells from, for forward comm
  // for reverse comm, proc I send a subset of my ghost cells to
  // offset used in forward comm to recv a message in middle of a large buffer

  struct Recv {
    int proc;
    int nunpack;
    int *unpacklist;
    int offset;
  };

  int adjacent;    // 0 on a proc who receives ghosts from a non-neighbor proc

  // copy = subset of my owned cells to copy into subset of my ghost cells
  // that describes forward comm, for reverse comm it is the opposite

  struct Copy {
    int npack;
    int nunpack;
    int *packlist;
    int *unpacklist;
  };

  int nsend, nrecv, ncopy;
  Send *send;
  Recv *recv;
  Copy *copy;

  // -------------------------------------------
  // internal methods
  // -------------------------------------------

  void initialize(MPI_Comm, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
                  int, int, int, int, int, int, int, int, int, int, int, int, int);
  virtual void setup_regular(int &, int &);
  virtual void setup_tiled(int &, int &);
  void ghost_box_drop(int *, int *);
  void box_drop_grid(int *, int, int, int &, int *);

  int ghost_adjacent_regular();
  int ghost_adjacent_tiled();

  template <class T> void forward_comm_regular(T *, int, int, int, void *, void *, MPI_Datatype);
  template <class T> void forward_comm_tiled(T *, int, int, int, void *, void *, MPI_Datatype);
  template <class T> void reverse_comm_regular(T *, int, int, int, void *, void *, MPI_Datatype);
  template <class T> void reverse_comm_tiled(T *, int, int, int, void *, void *, MPI_Datatype);

  virtual void grow_swap();
  void grow_overlap();

  int indices(int *&, int, int, int, int, int, int);
};

}    // namespace LAMMPS_NS

#endif
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_NEIGHBOR
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_NEIGHBOR_H
#define LMP_NEIGHBOR_H

#include "pointers.h"

namespace LAMMPS_NS {

class Neighbor : protected Pointers {
 public:
  enum { NSQ, BIN, MULTI_OLD, MULTI };
  int style;           // 0,1,2,3 = nsq, bin, multi/old, multi
  int every;           // build every this many steps
  int delay;           // delay build for this many steps
  int dist_check;      // 0 = always build, 1 = only if 1/2 dist
  int ago;             // how many steps ago neighboring occurred
  int pgsize;          // size of neighbor page
  int oneatom;         // max # of neighbors for one atom
  int includegroup;    // only build pairwise lists for this group
  int build_once;      // 1 if only build lists once per run

  double skin;                    // skin distance
  double cutneighmin;             // min neighbor cutoff for all type pairs
  double cutneighmax;             // max neighbor cutoff for all type pairs
  double cutneighmaxsq;           // cutneighmax squared
  double **cutneighsq;            // neighbor cutneigh sq for each type pair
  double **cutneighghostsq;       // cutneigh sq for each ghost type pair
  double *cuttype;                // for each type, max neigh cut w/ others
  double *cuttypesq;              // cuttype squared
  double cut_inner_sq;            // outer cutoff for inner neighbor list
  double cut_middle_sq;           // outer cutoff for middle neighbor list
  double cut_middle_inside_sq;    // inner cutoff for middle neighbor list

  int binsizeflag;        // user-chosen bin size
  double binsize_user;    // set externally by some accelerator pkgs

  bigint ncalls;      // # of times build has been called
  bigint ndanger;     // # of dangerous builds
  bigint lastcall;    // timestep of last neighbor::build() call

  // geometry and static info, used by other Neigh classes

  double *bboxlo, *bboxhi;    // ptrs to full domain bounding box
                              // different for orthog vs triclinic

  // exclusion info, used by NeighPair

  int exclude;    // 0 if no type/group exclusions, 1 if yes

  int nex_type;                // # of entries in type exclusion list
  int *ex1_type, *ex2_type;    // pairs of types to exclude
  int **ex_type;               // 2d array of excluded type pairs

  int nex_group;                 // # of entries in group exclusion list
  int *ex1_group, *ex2_group;    // pairs of group #'s to exclude
  int *ex1_bit, *ex2_bit;        // pairs of group bits to exclude

  int nex_mol;          // # of entries in molecule exclusion list
  int *ex_mol_group;    // molecule group #'s to exclude
  int *ex_mol_bit;      // molecule group bits to exclude
  int *ex_mol_intra;    // 0 = exclude if in 2 molecules (inter)
                        // 1 = exclude if in same molecule (intra)

  // special info, used by NeighPair

  int special_flag[4];    // flags for 1-2, 1-3, 1-4 neighbors

  // cluster setting, used by NeighTopo

  int cluster_check;    // 1 if check bond/angle/etc satisfies minimg

  // pairwise neighbor lists and corresponding requests

  int nlist;           // # of pairwise neighbor lists
  int nrequest;        // # of requests, same as nlist
  int old_nrequest;    // # of requests for previous run

  class NeighList **lists;
  class NeighRequest **requests;        // from Pair,Fix,Compute,Command classes
  class NeighRequest **old_requests;    // copy of requests to compare to

  // data from topology neighbor lists

  int nbondlist;    // list of bonds to compute
  int **bondlist;
  int nanglelist;    // list of angles to compute
  int **anglelist;
  int ndihedrallist;    // list of dihedrals to compute
  int **dihedrallist;
  int nimproperlist;    // list of impropers to compute
  int **improperlist;

  // optional type grouping for multi

  int custom_collection_flag;      // 1 if custom collections are defined for multi
  int interval_collection_flag;    // 1 if custom collections use intervals
  int finite_cut_flag;             // 1 if multi considers finite atom size
  int ncollections;                // # of custom collections
  int nmax_collection;             // maximum atoms stored in collection array
  int *type2collection;            // ntype array mapping types to custom collections
  double *collection2cut;          // ncollection array with upper bounds on cutoff intervals
  double **cutcollectionsq;        // cutoffs for each combination of collections
  int *collection;                 // local per-atom array to store collection id

  // public methods

  Neighbor(class LAMMPS *);
  virtual ~Neighbor();
  virtual void init();
  int request(void *, int instance = 0);
  int decide();                     // decide whether to build or not
  virtual int check_distance();     // check max distance moved since last build
  void setup_bins();                // setup bins based on box and cutoff
  virtual void build(int);          // build all perpetual neighbor lists
  virtual void build_topology();    // pairwise topology neighbor lists
  void build_one(class NeighList *list, int preflag = 0);
  // create a one-time pairwise neigh list
  void set(int, char **);                     // set neighbor style and skin distance
  void reset_timestep(bigint);                // reset of timestep counter
  void modify_params(int, char **);           // modify params that control builds
  void modify_params(const std::string &);    // convenience overload

  void exclusion_group_group_delete(int, int);    // rm a group-group exclusion
  int exclude_setting();                          // return exclude value to accelerator pkg
  class NeighRequest *find_request(void *);       // find a neighbor request
  int any_full();                // Check if any old requests had full neighbor lists
  void build_collection(int);    // build peratom collection array starting at the given index

  double memory_usage();

  bigint last_setup_bins;    // step of last neighbor::setup_bins() call

 public:
  int me, nprocs;
  int firsttime;    // flag for calling init_styles() only once

  int dimension;      // 2/3 for 2d/3d
  int triclinic;      // 0 if domain is orthog, 1 if triclinic
  int newton_pair;    // 0 if newton off for pairwise, 1 if on

  int must_check;       // 1 if must check other classes to reneigh
  int restart_check;    // 1 if restart enabled, 0 if no
  int fix_check;        // # of fixes that induce reneigh
  int *fixchecklist;    // which fixes to check

  double triggersq;    // trigger = build when atom moves this dist

  double **xhold;    // atom coords at last neighbor build
  int maxhold;       // size of xhold array

  int boxcheck;                           // 1 if need to store box size
  double boxlo_hold[3], boxhi_hold[3];    // box size at last neighbor build
  double corners_hold[8][3];              // box corners at last neighbor build
  double (*corners)[3];                   // ptr to 8 corners of triclinic box

  double inner[2], middle[2];    // rRESPA cutoffs for extra lists

  int old_style, old_triclinic;    // previous run info
  int old_pgsize, old_oneatom;     // used to avoid re-creating neigh lists

  int nstencil_perpetual;    // # of perpetual NeighStencil classes
  int npair_perpetual;       // #x of perpetual NeighPair classes
  int *slist;                // indices of them in neigh_stencil
  int *plist;                // indices of them in neigh_pair

  int maxex_type;     // max # in exclusion type list
  int maxex_group;    // max # in exclusion group list
  int maxex_mol;      // max # in exclusion molecule list

  int maxatom;       // max size of atom-based NeighList arrays
  int maxrequest;    // max size of NeighRequest list

  // info for other Neigh classes: NBin,NStencil,NPair,NTopo

  int nbin, nstencil;
  int nbclass, nsclass, npclass;
  int bondwhich, anglewhich, dihedralwhich, improperwhich;

  typedef class NBin *(*BinCreator)(class LAMMPS *);
  BinCreator *binclass;
  char **binnames;
  int *binmasks;
  class NBin **neigh_bin;

  typedef class NStencil *(*StencilCreator)(class LAMMPS *);
  StencilCreator *stencilclass;
  char **stencilnames;
  int *stencilmasks;
  class NStencil **neigh_stencil;

  typedef class NPair *(*PairCreator)(class LAMMPS *);
  PairCreator *pairclass;
  char **pairnames;
  int *pairmasks;
  class NPair **neigh_pair;

  class NTopo *neigh_bond;
  class NTopo *neigh_angle;
  class NTopo *neigh_dihedral;
  class NTopo *neigh_improper;

  // internal methods
  // including creator methods for Nbin,Nstencil,Npair instances

  void init_styles();
  int init_pair();
  virtual void init_topology();

  void morph_unique();
  void morph_skip();
  void morph_granular();
  void morph_halffull();
  void morph_copy();

  void print_pairwise_info();
  void requests_new2old();

  int choose_bin(class NeighRequest *);
  int choose_stencil(class NeighRequest *);
  int choose_pair(class NeighRequest *);

  template <typename T> static NBin *bin_creator(class LAMMPS *);
  template <typename T> static NStencil *stencil_creator(class LAMMPS *);
  template <typename T> static NPair *pair_creator(class LAMMPS *);

  // dummy functions provided by NeighborKokkos, called in init()
  // otherwise NeighborKokkos would have to overwrite init()

  int copymode;

  virtual void init_cutneighsq_kokkos(int) {}
  virtual void create_kokkos_list(int) {}
  virtual void init_ex_type_kokkos(int) {}
  virtual void init_ex_bit_kokkos() {}
  virtual void init_ex_mol_bit_kokkos() {}
  virtual void grow_ex_mol_intra_kokkos() {}
  virtual void set_binsize_kokkos() {}
};

namespace NeighConst {
  enum {
    NB_INTEL = 1 << 0,
    NB_KOKKOS_DEVICE = 1 << 1,
    NB_KOKKOS_HOST = 1 << 2,
    NB_SSA = 1 << 3,
    NB_STANDARD = 1 << 4,
    NB_MULTI = 1 << 5
  };

  enum {
    NS_BIN = 1 << 0,
    NS_MULTI = 1 << 1,
    NS_HALF = 1 << 2,
    NS_FULL = 1 << 3,
    NS_2D = 1 << 4,
    NS_3D = 1 << 5,
    NS_ORTHO = 1 << 6,
    NS_TRI = 1 << 7,
    NS_GHOST = 1 << 8,
    NS_SSA = 1 << 9,
    NS_MULTI_OLD = 1 << 10
  };

  enum {
    NP_NSQ = 1 << 0,
    NP_BIN = 1 << 1,
    NP_MULTI = 1 << 2,
    NP_HALF = 1 << 3,
    NP_FULL = 1 << 4,
    NP_ORTHO = 1 << 5,
    NP_TRI = 1 << 6,
    NP_ATOMONLY = 1 << 7,
    NP_MOLONLY = 1 << 8,
    NP_NEWTON = 1 << 9,
    NP_NEWTOFF = 1 << 10,
    NP_GHOST = 1 << 11,
    NP_SIZE = 1 << 12,
    NP_ONESIDE = 1 << 13,
    NP_RESPA = 1 << 14,
    NP_BOND = 1 << 15,
    NP_OMP = 1 << 16,
    NP_INTEL = 1 << 17,
    NP_KOKKOS_DEVICE = 1 << 18,
    NP_KOKKOS_HOST = 1 << 19,
    NP_SSA = 1 << 20,
    NP_COPY = 1 << 21,
    NP_SKIP = 1 << 22,
    NP_HALF_FULL = 1 << 23,
    NP_OFF2ON = 1 << 24,
    NP_MULTI_OLD = 1 << 25
  };
}    // namespace NeighConst

}    // namespace LAMMPS_NS

#endif

/* ERROR/WARNING messages:

E: Neighbor delay must be 0 or multiple of every setting

The delay and every parameters set via the neigh_modify command are
inconsistent.  If the delay setting is non-zero, then it must be a
multiple of the every setting.

E: Neighbor page size must be >= 10x the one atom setting

This is required to prevent wasting too much memory.

E: Invalid atom type in neighbor exclusion list

Atom types must range from 1 to Ntypes inclusive.

W: Neighbor exclusions used with KSpace solver may give inconsistent Coulombic energies

This is because excluding specific pair interactions also excludes
them from long-range interactions which may not be the desired effect.
The special_bonds command handles this consistently by insuring
excluded (or weighted) 1-2, 1-3, 1-4 interactions are treated
consistently by both the short-range pair style and the long-range
solver.  This is not done for exclusions of charged atom pairs via the
neigh_modify exclude command.

E: Cannot request an occasional binned neighbor list with ghost info

UNDOCUMENTED

E: Requested neighbor bin option does not exist

UNDOCUMENTED

E: Requested neighbor stencil method does not exist

UNDOCUMENTED

E: Requested neighbor pair method does not exist

UNDOCUMENTED

E: Could not assign bin method to neighbor stencil

UNDOCUMENTED

E: Could not assign bin method to neighbor pair

UNDOCUMENTED

E: Could not assign stencil method to neighbor pair

UNDOCUMENTED

E: Neighbor include group not allowed with ghost neighbors

This is a current restriction within LAMMPS.

E: Too many local+ghost atoms for neighbor list

The number of nlocal + nghost atoms on a processor
is limited by the size of a 32-bit integer with 2 bits
removed for masking 1-2, 1-3, 1-4 neighbors.

E: Trying to build an occasional neighbor list before initialization completed

This is not allowed.  Source code caller needs to be modified.

E: Neighbor build one invoked on perpetual list

UNDOCUMENTED

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid group ID in neigh_modify command

A group ID used in the neigh_modify command does not exist.

E: Neigh_modify include group != atom_modify first group

Self-explanatory.

E: Neigh_modify exclude molecule requires atom attribute molecule

Self-explanatory.

E: Unable to find group-group exclusion

UNDOCUMENTED

U: Neighbor multi not yet enabled for ghost neighbors

This is a current restriction within LAMMPS.

U: Neighbor multi not yet enabled for granular

Self-explanatory.

U: Neighbor multi not yet enabled for rRESPA

Self-explanatory.

U: Domain too large for neighbor bins

The domain has become extremely large so that neighbor bins cannot be
used.  Most likely, one or more atoms have been blown out of the
simulation box to a great distance.

U: Cannot use neighbor bins - box size << cutoff

Too many neighbor bins will be created.  This typically happens when
the simulation box is very small in some dimension, compared to the
neighbor cutoff.  Use the "nsq" style instead of "bin" style.

U: Too many neighbor bins

This is likely due to an immense simulation box that has blown up
to a large size.

*/
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_INTEGRATE
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_INTEGRATE_H
#define LMP_INTEGRATE_H

#include "pointers.h"

namespace LAMMPS_NS {

class Integrate : protected Pointers {
 public:
  Integrate(class LAMMPS *, int, char **);
  virtual ~Integrate();
  virtual void init();
  virtual void setup(int flag) = 0;
  virtual void setup_minimal(int) = 0;
  virtual void run(int) = 0;
  virtual void cleanup() {}
  virtual void reset_dt() {}
  virtual double memory_usage() { return 0; }

 public:
  int eflag, vflag;            // flags for energy/virial computation
  int virial_style;            // compute virial explicitly or implicitly
  int external_force_clear;    // clear forces locally or externally

  int nelist_global, nelist_atom;    // # of PE,virial computes to check
  int nvlist_global, nvlist_atom, ncvlist_atom;
  class Compute **elist_global;    // lists of PE,virial Computes
  class Compute **elist_atom;
  class Compute **vlist_global;
  class Compute **vlist_atom;
  class Compute **cvlist_atom;

  int pair_compute_flag;      // 0 if pair->compute is skipped
  int kspace_compute_flag;    // 0 if kspace->compute is skipped

  void ev_setup();
  void ev_set(bigint);
};

}    // namespace LAMMPS_NS

#endif

/* ERROR/WARNING messages:

*/
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_PAIR_H
#define LMP_PAIR_H

#include "pointers.h"    // IWYU pragma: export

namespace LAMMPS_NS {

class Pair : protected Pointers {
  friend class AngleSDK;
  friend class AngleSDKOMP;
  friend class BondQuartic;
  friend class BondQuarticOMP;
  friend class DihedralCharmm;
  friend class DihedralCharmmOMP;
  friend class FixGPU;
  friend class FixIntel;
  friend class FixOMP;
  friend class FixQEq;
  friend class PairHybrid;
  friend class PairHybridScaled;
  friend class ThrOMP;
  friend class Info;

 public:
  static int instance_total;    // # of Pair classes ever instantiated

  double eng_vdwl, eng_coul;    // accumulated energies
  double virial[6];             // accumulated virial: xx,yy,zz,xy,xz,yz
  double *eatom, **vatom;       // accumulated per-atom energy/virial
  double **cvatom;              // accumulated per-atom centroid virial

  double cutforce;    // max cutoff for all atom pairs
  double **cutsq;     // cutoff sq for each atom pair
  int **setflag;      // 0/1 = whether each i,j has been set

  int comm_forward;        // size of forward communication (0 if none)
  int comm_reverse;        // size of reverse communication (0 if none)
  int comm_reverse_off;    // size of reverse comm even if newton off

  int single_enable;              // 1 if single() routine exists
  int single_hessian_enable;      // 1 if single_hessian() routine exists
  int restartinfo;                // 1 if pair style writes restart info
  int respa_enable;               // 1 if inner/middle/outer rRESPA routines
  int one_coeff;                  // 1 if allows only one coeff * * call
  int manybody_flag;              // 1 if a manybody potential
  int unit_convert_flag;          // value != 0 indicates support for unit conversion.
  int no_virial_fdotr_compute;    // 1 if does not invoke virial_fdotr_compute()
  int writedata;                  // 1 if writes coeffs to data file
  int finitecutflag;              // 1 if cut depends on finite atom size
  int ghostneigh;                 // 1 if pair style needs neighbors of ghosts
  double **cutghost;              // cutoff for each ghost pair

  int ewaldflag;         // 1 if compatible with Ewald solver
  int pppmflag;          // 1 if compatible with PPPM solver
  int msmflag;           // 1 if compatible with MSM solver
  int dispersionflag;    // 1 if compatible with LJ/dispersion solver
  int tip4pflag;         // 1 if compatible with TIP4P solver
  int dipoleflag;        // 1 if compatible with dipole solver
  int spinflag;          // 1 if compatible with spin solver
  int reinitflag;        // 1 if compatible with fix adapt and alike

  int centroidstressflag;    // centroid stress compared to two-body stress
                             // CENTROID_SAME = same as two-body stress
                             // CENTROID_AVAIL = different and implemented
                             // CENTROID_NOTAVAIL = different, not yet implemented

  int tail_flag;          // pair_modify flag for LJ tail correction
  double etail, ptail;    // energy/pressure tail corrections
  double etail_ij, ptail_ij;

  int evflag;    // energy,virial settings
  int eflag_either, eflag_global, eflag_atom;
  int vflag_either, vflag_global, vflag_atom, cvflag_atom;

  int ncoultablebits;    // size of Coulomb table, accessed by KSpace
  int ndisptablebits;    // size of dispersion table
  double tabinnersq;
  double tabinnerdispsq;
  double *rtable, *drtable, *ftable, *dftable, *ctable, *dctable;
  double *etable, *detable, *ptable, *dptable, *vtable, *dvtable;
  double *rdisptable, *drdisptable, *fdisptable, *dfdisptable;
  double *edisptable, *dedisptable;
  int ncoulshiftbits, ncoulmask;
  int ndispshiftbits, ndispmask;

  int nextra;         // # of extra quantities pair style calculates
  double *pvector;    // vector of extra pair quantities

  int single_extra;    // number of extra single values calculated
  double *svector;     // vector of extra single quantities

  class NeighList *list;        // standard neighbor list used by most pairs
  class NeighList *listhalf;    // half list used by some pairs
  class NeighList *listfull;    // full list used by some pairs

  int allocated;       // 0/1 = whether arrays are allocated
                       //       public so external driver can check
  int compute_flag;    // 0 if skip compute()
  int mixed_flag;      // 1 if all itype != jtype coeffs are from mixing

  enum { GEOMETRIC, ARITHMETIC, SIXTHPOWER };    // mixing options

  int beyond_contact, nondefault_history_transfer;    // for granular styles

  // KOKKOS host/device flag and data masks

  ExecutionSpace execution_space;
  unsigned int datamask_read, datamask_modify;
  int kokkosable;    // 1 if Kokkos pair

  Pair(class LAMMPS *);
  virtual ~Pair();

  // top-level Pair methods

  void init();
  virtual void reinit();
  virtual void setup() {}
  double mix_energy(double, double, double, double);
  double mix_distance(double, double);
  void write_file(int, char **);
  void init_bitmap(double, double, int, int &, int &, int &, int &);
  virtual void modify_params(int, char **);
  void compute_dummy(int, int);

  // need to be public, so can be called by pair_style reaxc

  void ev_tally(int, int, int, int, double, double, double, double, double, double);
  void ev_tally3(int, int, int, double, double, double *, double *, double *, double *);
  void v_tally2_newton(int, double *, double *);
  void v_tally3(int, int, int, double *, double *, double *, double *);
  void v_tally4(int, int, int, int, double *, double *, double *, double *, double *, double *);

  // general child-class methods

  virtual void compute(int, int) = 0;
  virtual void compute_inner() {}
  virtual void compute_middle() {}
  virtual void compute_outer(int, int) {}

  virtual double single(int, int, int, int, double, double, double, double &fforce)
  {
    fforce = 0.0;
    return 0.0;
  }

  void hessian_twobody(double fforce, double dfac, double delr[3], double phiTensor[6]);

  virtual double single_hessian(int, int, int, int, double, double[3], double, double,
                                double &fforce, double d2u[6])
  {
    fforce = 0.0;
    for (int i = 0; i < 6; i++) d2u[i] = 0;
    return 0.0;
  }

  virtual void settings(int, char **) = 0;
  virtual void coeff(int, char **) = 0;

  virtual void init_style();
  virtual void init_list(int, class NeighList *);
  virtual double init_one(int, int) { return 0.0; }

  virtual void init_tables(double, double *);
  virtual void init_tables_disp(double);
  virtual void free_tables();
  virtual void free_disp_tables();

  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *) {}
  virtual void read_restart_settings(FILE *) {}
  virtual void write_data(FILE *) {}
  virtual void write_data_all(FILE *) {}

  virtual int pack_forward_comm(int, int *, double *, int, int *) { return 0; }
  virtual void unpack_forward_comm(int, int, double *) {}
  virtual int pack_reverse_comm(int, int, double *) { return 0; }
  virtual void unpack_reverse_comm(int, int *, double *) {}
  virtual double memory_usage();

  void set_copymode(int value) { copymode = value; }

  // specific child-class methods for certain Pair styles

  virtual void *extract(const char *, int &) { return nullptr; }
  virtual void swap_eam(double *, double **) {}
  virtual void reset_dt() {}
  virtual void min_xf_pointers(int, double **, double **) {}
  virtual void min_xf_get(int) {}
  virtual void min_x_set(int) {}
  virtual void transfer_history(double *, double *) {}
  virtual double atom2cut(int) { return 0.0; }
  virtual double radii2cut(double, double) { return 0.0; }

  // management of callbacks to be run from ev_tally()

 public:
  int num_tally_compute;
  class Compute **list_tally_compute;

 public:
  virtual void add_tally_callback(class Compute *);
  virtual void del_tally_callback(class Compute *);

 public:
  int instance_me;      // which Pair class instantiation I am
  int special_lj[4];    // copied from force->special_lj for Kokkos
  int suffix_flag;      // suffix compatibility flag

  // pair_modify settings
  int offset_flag, mix_flag;    // flags for offset and mixing
  double tabinner;              // inner cutoff for Coulomb table
  double tabinner_disp;         // inner cutoff for dispersion table

 public:
  // for mapping of elements to atom types and parameters
  // mostly used for manybody potentials
  int nelements;        // # of unique elements
  char **elements;      // names of unique elements
  int *elem1param;      // mapping from elements to parameters
  int **elem2param;     // mapping from element pairs to parameters
  int ***elem3param;    // mapping from element triplets to parameters
  int *map;             // mapping from atom types to elements
  int nparams;          // # of stored parameter sets
  int maxparam;         // max # of parameter sets
  void map_element2type(int, char **, bool update_setflag = true);

 public:
  // custom data type for accessing Coulomb tables

  typedef union {
    int i;
    float f;
  } union_int_float_t;

  // Accessor for the INTEL package to determine virial calc for hybrid

  inline int fdotr_is_set() const { return vflag_fdotr; }

 public:
  int vflag_fdotr;
  int maxeatom, maxvatom, maxcvatom;

  int copymode;    // if set, do not deallocate during destruction
                   // required when classes are used as functors by Kokkos

  void ev_init(int eflag, int vflag, int alloc = 1)
  {
    if (eflag || vflag)
      ev_setup(eflag, vflag, alloc);
    else
      ev_unset();
  }
  virtual void ev_setup(int, int, int alloc = 1);
  void ev_unset();
  void ev_tally_full(int, double, double, double, double, double, double);
  void ev_tally_xyz_full(int, double, double, double, double, double, double, double, double);
  void ev_tally4(int, int, int, int, double, double *, double *, double *, double *, double *,
                 double *);
  void ev_tally_tip4p(int, int *, double *, double, double);
  void ev_tally_xyz(int, int, int, int, double, double, double, double, double, double, double,
                    double);
  void v_tally2(int, int, double, double *);
  void v_tally_tensor(int, int, int, int, double, double, double, double, double, double);
  void virial_fdotr_compute();

  inline int sbmask(int j) const { return j >> SBBITS & 3; }
};

}    // namespace LAMMPS_NS

#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Too many total bits for bitmapped lookup table

Table size specified via pair_modify command is too large.  Note that
a value of N generates a 2^N size table.

E: Cannot have both pair_modify shift and tail set to yes

These 2 options are contradictory.

E: Cannot use pair tail corrections with 2d simulations

The correction factors are only currently defined for 3d systems.

W: Using pair tail corrections with non-periodic system

This is probably a bogus thing to do, since tail corrections are
computed by integrating the density of a periodic system out to
infinity.

W: Using pair tail corrections with pair_modify compute no

The tail corrections will thus not be computed.

W: Using pair potential shift with pair_modify compute no

The shift effects will thus not be computed.

W: Using a manybody potential with bonds/angles/dihedrals and special_bond exclusions

This is likely not what you want to do.  The exclusion settings will
eliminate neighbors in the neighbor list, which the manybody potential
needs to calculated its terms correctly.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Fix adapt interface to this pair style not supported

New coding for the pair style would need to be done.

E: Pair style requires a KSpace style

No kspace style is defined.

E: BUG: restartinfo=1 but no restart support in pair style

The pair style has a bug, where it does not support reading
and writing information to a restart file, but does not set
the member variable restartinfo to 0 as required in that case.

E: Cannot yet use compute tally with Kokkos

This feature is not yet supported.

E: Pair style does not support pair_write

The pair style does not have a single() function, so it can
not be invoked by pair write.

E: Invalid atom types in pair_write command

Atom types must range from 1 to Ntypes inclusive.

E: Invalid style in pair_write command

Self-explanatory.  Check the input script.

E: Invalid cutoffs in pair_write command

Inner cutoff must be larger than 0.0 and less than outer cutoff.

E: Cannot open pair_write file

The specified output file for pair energies and forces cannot be
opened.  Check that the path and name are correct.

E: Bitmapped lookup tables require int/float be same size

Cannot use pair tables on this machine, because of word sizes.  Use
the pair_modify command with table 0 instead.

W: Table inner cutoff >= outer cutoff

You specified an inner cutoff for a Coulombic table that is longer
than the global cutoff.  Probably not what you wanted.

E: Too many exponent bits for lookup table

Table size specified via pair_modify command does not work with your
machine's floating point representation.

E: Too many mantissa bits for lookup table

Table size specified via pair_modify command does not work with your
machine's floating point representation.

E: Too few bits for lookup table

Table size specified via pair_modify command does not work with your
machine's floating point representation.

*/
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_HYBRID
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(hybrid,PairHybrid);
// clang-format on
#else

#ifndef LMP_PAIR_HYBRID_H
#define LMP_PAIR_HYBRID_H

#include "pair.h"

namespace LAMMPS_NS {

class PairHybrid : public Pair {
  friend class ComputeSpin;
  friend class FixGPU;
  friend class FixIntel;
  friend class FixNVESpin;
  friend class FixOMP;
  friend class Force;
  friend class Info;
  friend class Neighbor;
  friend class PairDeprecated;
  friend class Respa;
  friend class Scafacos;

 public:
  PairHybrid(class LAMMPS *);
  virtual ~PairHybrid();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void setup();
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);
  void modify_params(int narg, char **arg);
  double memory_usage();

  void compute_inner();
  void compute_middle();
  void compute_outer(int, int);
  void *extract(const char *, int &);
  void reset_dt();

  int check_ijtype(int, int, char *);

  virtual void add_tally_callback(class Compute *);
  virtual void del_tally_callback(class Compute *);
  double atom2cut(int);
  double radii2cut(double, double);

 public:
  int nstyles;        // # of sub-styles
  Pair **styles;      // list of Pair style classes
  char **keywords;    // style name of each Pair style
  int *multiple;      // 0 if style used once, else Mth instance

  int outerflag;    // toggle compute() when invoked by outer()
  int respaflag;    // 1 if different substyles are assigned to
                    // different r-RESPA levels

  int **nmap;               // # of sub-styles itype,jtype points to
  int ***map;               // list of sub-styles itype,jtype points to
  double **special_lj;      // list of per style LJ exclusion factors
  double **special_coul;    // list of per style Coulomb exclusion factors
  int *compute_tally;       // list of on/off flags for tally computes

  void allocate();
  void flags();

  virtual void init_svector();
  virtual void copy_svector(int, int);

  void modify_special(int, int, char **);
  double *save_special();
  void set_special(int);
  void restore_special(double *);
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Pair style hybrid cannot have hybrid as an argument

Self-explanatory.

E: Pair style hybrid cannot have none as an argument

Self-explanatory.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Cannot yet use pair hybrid with Kokkos

This feature is not yet supported.

E: Pair coeff for hybrid has invalid style

Style in pair coeff must have been listed in pair_style command.

E: Pair hybrid sub-style is not used

No pair_coeff command used a sub-style specified in the pair_style
command.

E: Pair_modify special setting for pair hybrid incompatible with global special_bonds setting

Cannot override a setting of 0.0 or 1.0 or change a setting between
0.0 and 1.0.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Invoked pair single on pair style none

A command (e.g. a dump) attempted to invoke the single() function on a
pair style none, which is illegal.  You are probably attempting to
compute per-atom quantities with an undefined pair style.

E: Pair hybrid sub-style does not support single call

You are attempting to invoke a single() call on a pair style
that doesn't support it.

E: Pair hybrid single calls do not support per sub-style special bond values

Self-explanatory.

E: Unknown pair_modify hybrid sub-style

The choice of sub-style is unknown.

E: Coulomb cutoffs of pair hybrid sub-styles do not match

If using a Kspace solver, all Coulomb cutoffs of long pair styles must
be the same.

*/
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CUT_COUL_LONG
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/coul/long,PairLJCutCoulLong);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_COUL_LONG_H
#define LMP_PAIR_LJ_CUT_COUL_LONG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutCoulLong : public Pair {

 public:
  PairLJCutCoulLong(class LAMMPS *);
  virtual ~PairLJCutCoulLong();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);

  void compute_inner();
  void compute_middle();
  virtual void compute_outer(int, int);
  virtual void *extract(const char *, int &);

 public:
  double cut_lj_global;
  double **cut_lj, **cut_ljsq;
  double cut_coul, cut_coulsq;
  double **epsilon, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;
  double *cut_respa;
  double qdist;    // TIP4P distance from O site to negative charge
  double g_ewald;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style lj/cut/coul/long requires atom attribute q

The atom style defined does not have this attribute.

E: Pair style requires a KSpace style

No kspace style is defined.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CUT_COUL_CUT
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/coul/cut,PairLJCutCoulCut);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_COUL_CUT_H
#define LMP_PAIR_LJ_CUT_COUL_CUT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutCoulCut : public Pair {
 public:
  PairLJCutCoulCut(class LAMMPS *);
  virtual ~PairLJCutCoulCut();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

 public:
  double cut_lj_global, cut_coul_global;
  double **cut_lj, **cut_ljsq;
  double **cut_coul, **cut_coulsq;
  double **epsilon, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style lj/cut/coul/cut requires atom attribute q

The atom style defined does not have this attribute.

*/
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CHARMM_COUL_LONG
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/charmm/coul/long,PairLJCharmmCoulLong);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CHARMM_COUL_LONG_H
#define LMP_PAIR_LJ_CHARMM_COUL_LONG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCharmmCoulLong : public Pair {
 public:
  PairLJCharmmCoulLong(class LAMMPS *);
  virtual ~PairLJCharmmCoulLong();

  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);

  void compute_inner();
  void compute_middle();
  virtual void compute_outer(int, int);
  virtual void *extract(const char *, int &);

 public:
  int implicit;
  double cut_lj_inner, cut_lj;
  double cut_lj_innersq, cut_ljsq;
  double cut_coul, cut_coulsq;
  double cut_bothsq;
  double cut_in_off, cut_in_on, cut_out_off, cut_out_on;
  double cut_in_diff, cut_out_diff;
  double cut_in_diff_inv, cut_out_diff_inv;
  double cut_in_off_sq, cut_in_on_sq, cut_out_off_sq, cut_out_on_sq;
  double denom_lj, denom_lj_inv;
  double **epsilon, **sigma, **eps14, **sigma14;
  double **lj1, **lj2, **lj3, **lj4, **offset;
  double **lj14_1, **lj14_2, **lj14_3, **lj14_4;
  double *cut_respa;
  double g_ewald;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style lj/charmm/coul/long requires atom attribute q

The atom style defined does not have these attributes.

E: Pair inner cutoff >= Pair outer cutoff

The specified cutoffs for the pair style are inconsistent.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

E: Pair inner cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

E: Pair style requires a KSpace style

No kspace style is defined.

*/
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CHARMM_COUL_CHARMM
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/charmm/coul/charmm,PairLJCharmmCoulCharmm);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CHARMM_COUL_CHARMM_H
#define LMP_PAIR_LJ_CHARMM_COUL_CHARMM_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCharmmCoulCharmm : public Pair {
 public:
  PairLJCharmmCoulCharmm(class LAMMPS *);
  virtual ~PairLJCharmmCoulCharmm();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  virtual double single(int, int, int, int, double, double, double, double &);
  virtual void *extract(const char *, int &);

 public:
  int implicit;
  double cut_lj_inner, cut_lj, cut_coul_inner, cut_coul;
  double cut_lj_innersq, cut_ljsq, cut_coul_innersq, cut_coulsq, cut_bothsq;
  double denom_lj, denom_coul;
  double **epsilon, **sigma, **eps14, **sigma14;
  double **lj1, **lj2, **lj3, **lj4;
  double **lj14_1, **lj14_2, **lj14_3, **lj14_4;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style lj/charmm/coul/charmm requires atom attribute q

The atom style defined does not have these attributes.

E: Pair inner cutoff >= Pair outer cutoff

The specified cutoffs for the pair style are inconsistent.

*/
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CUT_COUL_LONG_OMP
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CUT_COUL_LONG_GPU
#endif

/*----------------------------------------------------------*/
