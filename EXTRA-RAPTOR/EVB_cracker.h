#define LAMMPS_VERSION "3 Nov 2022"

#ifdef _CRACKER_KSPACE
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

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
  enum {
    REVERSE_RHO,
    REVERSE_RHO_GEOM,
    REVERSE_RHO_ARITH,
    REVERSE_RHO_NONE,
    REVERSE_RHO_GPU,
    REVERSE_RHO_G,
    REVERSE_RHO_A,
    REVERSE_AD,
    REVERSE_AD_PERATOM,
    REVERSE_MU,
    REVERSE_MU_PERATOM
  };
  enum {
    FORWARD_RHO,
    FORWARD_IK,
    FORWARD_AD,
    FORWARD_MU,
    FORWARD_IK_PERATOM,
    FORWARD_AD_PERATOM,
    FORWARD_MU_PERATOM,
    FORWARD_IK_GEOM,
    FORWARD_AD_GEOM,
    FORWARD_IK_PERATOM_GEOM,
    FORWARD_AD_PERATOM_GEOM,
    FORWARD_IK_ARITH,
    FORWARD_AD_ARITH,
    FORWARD_IK_PERATOM_ARITH,
    FORWARD_AD_PERATOM_ARITH,
    FORWARD_IK_NONE,
    FORWARD_AD_NONE,
    FORWARD_IK_PERATOM_NONE,
    FORWARD_AD_PERATOM_NONE,
    FORWARD_IK_G,
    FORWARD_AD_G,
    FORWARD_IK_PERATOM_G,
    FORWARD_AD_PERATOM_G,
    FORWARD_IK_A,
    FORWARD_AD_A,
    FORWARD_IK_PERATOM_A,
    FORWARD_AD_PERATOM_A
  };

  double energy;    // accumulated energies
  double energy_1, energy_6;
  double virial[6];          // accumulated virial: xx,yy,zz,xy,xz,yz
  double *eatom, **vatom;    // accumulated per-atom energy/virial
  double e2group;            // accumulated group-group energy
  double f2group[3];         // accumulated group-group force
  int triclinic_support;     // 1 if supports triclinic geometries

  int ewaldflag;         // 1 if a Ewald solver
  int pppmflag;          // 1 if a PPPM solver
  int espflag;           // 1 if a ESP solver
  int msmflag;           // 1 if a MSM solver
  int dispersionflag;    // 1 if a LJ/dispersion solver
  int tip4pflag;         // 1 if a TIP4P solver
  int dipoleflag;        // 1 if a dipole solver
  int spinflag;          // 1 if a spin solver
  int rk_flag;           /* 1 if a solver uses two distinct communicator worlds for
                            r-space and k-space computations*/
  int differentiation_flag;
  int neighrequest_flag;    // used to avoid obsolete construction
                            // of neighbor lists
  int mixflag;              // 1 if geometric mixing rules are enforced
                            // for LJ coefficients
  bool conp_one_step;       // calculate A matrix in one step with pppm
  int slabflag, wireflag;
  int slab_auto;
  int scalar_pressure_flag;    // 1 if using MSM fast scalar pressure
  double slab_volfactor, wire_volfactor;

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

  // Parameters required for ESP methods (automatically determined based on the user-specified force accuracy)

  double *force_poly_coeff, *energy_poly_coeff,
      *fourier_split_poly_coeff;    // polynomial coefficients
  int num_of_force_poly, num_of_energy_poly,
      num_of_Fourier_poly;    // order of polynomial approximations
  double select_c;            // the c value for ESP
  double Lambda_0;            // the Lambda_0 value for ESP

  double spreading_accuracy;    // using the pswf as the spreading function
  double spreading_select_c;    // select the c value for the spreading step
  double spreading_Lambda_0;    // select the Lambda_0 value for the spreading step
  int poly_order, fourier_spreading_order;
  double *fourier_spread_poly_coeff;

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
  uint64_t datamask_read, datamask_modify;
  int copymode;

  int compute_flag;        // 0 if skip compute()
  int fftbench;            // 0 if skip FFT timing
  int collective_flag;     // 1 if use MPI collectives for FFT/remap
  int nonblocking_flag;    // 1 if use MPI_Isend for FFT/remap
  int selfcopy_flag;       // collective self-copy mode
                           // 0 if no self-copy
                           // 1 if always self-copy
                           // 2 if only use self-copy for one rank runs
  int stagger_flag;        // 1 if using staggered PPPM grids

  double splittol;    // tolerance for when to truncate splitting

  KSpace(class LAMMPS *);
  ~KSpace() override;
  void two_charge();
  void triclinic_check();
  void modify_params(int, char **);
  void *extract(const char *);
  void compute_dummy(int eflag, int vflag, int alloc = 1);

  // triclinic

  void x2lamdaT(double *, double *);
  void lamda2xT(double *, double *);
  void lamda2xvector(double *, double *);

  // public so can be called by commands that change charge

  virtual void qsum_qsq(int warning_flag = 1);

  // general child-class methods

  virtual void settings(int, char **) {};
  virtual void init() = 0;
  virtual void setup() = 0;
  virtual void reset_grid() {};
  virtual void compute(int, int) = 0;
  virtual void compute_group_group(int, int, int) {};

  virtual void pack_forward_grid(int, void *, int, int *) {};
  virtual void unpack_forward_grid(int, void *, int, int *) {};
  virtual void pack_reverse_grid(int, void *, int, int *) {};
  virtual void unpack_reverse_grid(int, void *, int, int *) {};

  virtual int timing(int, double &, double &) { return 0; }
  virtual int timing_1d(int, double &) { return 0; }
  virtual int timing_3d(int, double &) { return 0; }

  virtual int modify_param(int, char **) { return 0; }
  virtual double memory_usage()
  {
    double bytes = (double) maxeatom * sizeof(double);
    bytes += (double) maxvatom * 6 * sizeof(double);
    return bytes;
  }

  // to be overridden in the *RK subclasses for which rk_flag == 1
  virtual void r2k_comm(int &, int &) {};               //rk_flag == 1
  virtual void k2r_comm(int, int) {};                   //rk_flag == 1
  virtual void compute_grid_potentials(int, int) {};    //rk_flag == 1

  /* ----------------------------------------------------------------------
   compute gamma for MSM and pair styles
   see Eq 4 from Parallel Computing 35 (2009) 164-177
------------------------------------------------------------------------- */

  [[nodiscard]] double gamma(const double &rho) const
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

  [[nodiscard]] double dgamma(const double &rho) const
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
  int eflag_either, eflag_global, eflag_atom, eflag_only;
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
      evflag = evflag_atom = eflag_either = eflag_global = eflag_atom = eflag_only = vflag_either =
          vflag_global = vflag_atom = 0;
  }
  void ev_setup(int, int, int alloc = 1);
  double estimate_table_accuracy(double, double);
};

}    // namespace LAMMPS_NS

#endif
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_NEIGHBOR
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

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

// forward declarations
class NeighRequest;
class NeighList;

class Neighbor : protected Pointers {
 public:
  enum { NSQ, BIN, MULTI };
  int style;           // 0,1,2 = nsq, bin, multi
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

  NeighList **lists;
  NeighRequest **requests;        // from Pair,Fix,Compute,Command classes
  NeighRequest **old_requests;    // copy of requests to compare to
  int *j_sorted;                  // index of requests sorted by cutoff distance

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

  int bin_hash;                    // 1 if using hash tables to store atoms in a bin
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
  ~Neighbor() override;
  virtual void init();

  // old API for creating neighbor list requests

  int request(void *, int instance = 0);

  // new API for creating neighbor list requests

  NeighRequest *add_request(class Pair *, int flags = 0);
  NeighRequest *add_request(class Fix *, int flags = 0);
  NeighRequest *add_request(class Compute *, int flags = 0);
  NeighRequest *add_request(class Command *, const char *, int flags = 0);

  // set neighbor list request OpenMP flag

  void set_omp_neighbor(int);

  // report if we have INTEL package neighbor lists

  [[nodiscard]] bool has_intel_request() const;

  int decide();                     // decide whether to build or not
  virtual int check_distance();     // check max distance moved since last build
  void setup_bins();                // setup bins based on box and cutoff
  virtual void build(int);          // build all perpetual neighbor lists
  virtual void build_topology();    // pairwise topology neighbor lists

  void build_one(class NeighList *list);      // create an occasional pairwise neigh list
  void set(int, char **);                     // set neighbor style and skin distance
  void reset_timestep(bigint);                // reset of timestep counter
  void modify_params(int, char **);           // modify params that control builds
  void modify_params(const std::string &);    // convenience overload

  void exclusion_group_group_delete(int, int);    // rm a group-group exclusion
  int exclude_setting();                          // return exclude value to accelerator pkg

  // option to call build_topology (e.g. from gpu styles instead for overlapped computation)

  int overlap_topo;    // 0 for default/old non-overlap mode
  void set_overlap_topo(int);

  // find a neighbor list or request based on requestor

  NeighList *find_list(void *, const int id = 0) const;
  NeighRequest *find_request(void *, const int id = 0) const;

  [[nodiscard]] const std::vector<NeighRequest *> get_pair_requests() const;
  int any_full();                // check if any old requests had full neighbor lists
  void build_collection(int);    // build peratom collection array starting at the given index

  bigint get_nneigh_full();    // return number of neighbors in a regular full neighbor list
  bigint get_nneigh_half();    // return number of neighbors in a regular half neighbor list

  // return "best" non-skip pair neighbor list (used by dump image)
  NeighList *get_best_pair_list();

  void add_temporary_bond(int, int, int);    // add temporary bond to bondlist array
  double memory_usage();

  bigint last_setup_bins;    // step of last neighbor::setup_bins() call
  double **get_xhold();      // access the latest-computed neighbor list positions

 public:
  int me, nprocs;
  int firsttime;    // flag for calling init_styles() only once

  int dimension;      // 2/3 for 2d/3d
  int triclinic;      // 0 if domain is orthog, 1 if triclinic
  int newton_pair;    // 0 if newton off for pairwise, 1 if on

  int must_check;                     // 1 if must check other classes to reneigh
  int restart_check;                  // 1 if restart enabled, 0 if no
  std::vector<Fix *> fixchecklist;    // which fixes to check

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
  int npair_perpetual;       // # of perpetual NeighPair classes
  int *slist;                // indices of them in neigh_stencil
  int *plist;                // indices of them in neigh_pair
  int npair_occasional;      // # of occasional NeighPair classes
  int *olist;                // indices of them in neigh_pair

  int maxex_type;     // max # in exclusion type list
  int maxex_group;    // max # in exclusion group list
  int maxex_mol;      // max # in exclusion molecule list

  int maxatom;       // max size of atom-based NeighList arrays
  int maxrequest;    // max size of NeighRequest list

  // info for other Neigh classes: NBin,NStencil,NPair,NTopo

  int nbin, nstencil;
  int nbclass, nsclass, npclass;
  int bondwhich, anglewhich, dihedralwhich, improperwhich;

  using BinCreator = class NBin *(*) (class LAMMPS *);
  BinCreator *binclass;
  char **binnames;
  int *binmasks;
  class NBin **neigh_bin;

  using StencilCreator = class NStencil *(*) (class LAMMPS *);
  StencilCreator *stencilclass;
  char **stencilnames;
  int *stencilmasks;
  class NStencil **neigh_stencil;

  using PairCreator = class NPair *(*) (class LAMMPS *);
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

  void sort_requests();

  void morph_unique();
  void morph_skip();
  void morph_granular();
  void morph_halffull();
  void morph_copy_trim();

  void print_pairwise_info();
  void requests_new2old();

  int choose_bin(class NeighRequest *);
  int choose_stencil(class NeighRequest *);
  int choose_pair(class NeighRequest *);

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
    NS_INTEL = 1 << 9,
    NS_SSA = 1 << 10
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
    NP_TRIM = 1 << 25
  };

  enum {
    REQ_DEFAULT = 0,
    REQ_FULL = 1 << 0,
    REQ_GHOST = 1 << 1,
    REQ_SIZE = 1 << 2,
    REQ_HISTORY = 1 << 3,
    REQ_OCCASIONAL = 1 << 4,
    REQ_RESPA_INOUT = 1 << 5,
    REQ_RESPA_ALL = 1 << 6,
    REQ_NEWTON_ON = 1 << 8,
    REQ_NEWTON_OFF = 1 << 9,
    REQ_SSA = 1 << 10,
    REQ_ONESIDED = 1 << 11
  };
}    // namespace NeighConst

}    // namespace LAMMPS_NS

#endif
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_INTEGRATE
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

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
class Compute;

class Integrate : protected Pointers {
 public:
  Integrate(class LAMMPS *, int, char **);
  virtual void init();
  virtual void setup(int flag) = 0;
  virtual void setup_minimal(int) = 0;
  virtual void run(int) = 0;
  virtual void force_clear() = 0;
  virtual void cleanup() {}
  virtual void reset_dt() {}
  virtual double memory_usage() { return 0; }

 public:
  int eflag, vflag;            // flags for energy/virial computation
  int virial_style;            // compute virial explicitly or implicitly
  int external_force_clear;    // clear forces locally or externally

  // lists of PE,virial Computes
  std::vector<Compute *> elist_global, elist_atom, vlist_global, vlist_atom, cvlist_atom;

  int pair_compute_flag;      // 0 if pair->compute is skipped
  int kspace_compute_flag;    // 0 if kspace->compute is skipped
  int rk_flag;                // 1 if an rk decomposition is supported

  void ev_setup();
  void ev_set(bigint);
};

}    // namespace LAMMPS_NS

#endif
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

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
  friend class AngleSPICA;
  friend class AngleSPICAOMP;
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
  friend class Neighbor;

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

  int single_enable;            // 1 if single() routine exists
  int born_matrix_enable;       // 1 if born_matrix() routine exists
  int single_hessian_enable;    // 1 if single_hessian() routine exists
  int atomic_energy_enable;     // 1 if compute_atomic_energy() routine exists

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
  int suffix_flag;                // suffix compatibility flag

  int ewaldflag;         // 1 if compatible with Ewald solver
  int pppmflag;          // 1 if compatible with PPPM solver
  int espflag;           // 1 if compatible with PS solver
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
  int trim_flag;    // pair_modify flag for trimming neigh list

  int evflag;    // energy,virial settings
  int eflag_either, eflag_global, eflag_atom, eflag_only;
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
  bool did_mix;        // set to true by mix_energy() to indicate that mixing was performed

  enum { GEOMETRIC, ARITHMETIC, SIXTHPOWER };    // mixing options

  int beyond_contact, nondefault_history_transfer;    // for granular styles

  // KOKKOS flags and variables

  ExecutionSpace execution_space;
  uint64_t datamask_read, datamask_modify;
  int kokkosable;               // 1 if Kokkos pair
  int reverse_comm_device;      // 1 if reverse comm on Device
  int fuse_force_clear_flag;    // 1 if can fuse force clear with force compute

  Pair(class LAMMPS *);
  ~Pair() override;

  // top-level Pair methods

  void init();
  virtual void reinit();
  virtual void setup() {}
  double mix_energy(double, double, double, double);
  double mix_distance(double, double);
  void write_file(int, char **);
  void init_bitmap(double, double, int, int &, int &, int &, int &);
  virtual void modify_params(int, char **);
  void compute_dummy(int eflag, int vflag, int alloc = 1);

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
  virtual double compute_atomic_energy(int, NeighList *) { return 0.0; }

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

  virtual void born_matrix(int /*i*/, int /*j*/, int /*itype*/, int /*jtype*/, double /*rsq*/,
                           double /*factor_coul*/, double /*factor_lj*/, double &du, double &du2)
  {
    du = du2 = 0.0;
  }

  virtual void finish() {}
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

  virtual void reset_grid() {}

  virtual void pack_forward_grid(int, void *, int, int *) {}
  virtual void unpack_forward_grid(int, void *, int, int *) {}
  virtual void pack_reverse_grid(int, void *, int, int *) {}
  virtual void unpack_reverse_grid(int, void *, int, int *) {}

  virtual double memory_usage();

  void set_copymode(int value) { copymode = value; }

  // specific child-class methods for certain Pair styles

  virtual void *extract(const char *, int &) { return nullptr; }
  virtual void *extract_peratom(const char *, int &) { return nullptr; }
  virtual void swap_eam(double *, double **) {}
  virtual void reset_dt() {}
  virtual void min_xf_pointers(int, double **, double **) {}
  virtual void min_xf_get(int) {}
  virtual void min_x_set(int) {}
  virtual void transfer_history(double *, double *, int, int) {}

  // management of callbacks to be run from ev_tally()

 public:
  int num_tally_compute, did_tally_flag;
  class Compute **list_tally_compute;

 public:
  virtual void add_tally_callback(class Compute *);
  virtual void del_tally_callback(class Compute *);
  [[nodiscard]] bool did_tally_callback() const { return did_tally_flag != 0; }

 public:
  int instance_me;      // which Pair class instantiation I am
  int special_lj[4];    // copied from force->special_lj for Kokkos

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
  // NOLINTBEGIN
  typedef union {
    int i;
    float f;
  } union_int_float_t;
  // NOLINTEND

  // Accessor for the INTEL package to determine virial calc for hybrid

  [[nodiscard]] int fdotr_is_set() const { return vflag_fdotr; }

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

  [[nodiscard]] int sbmask(int j) const { return j >> SBBITS & 3; }
};

}    // namespace LAMMPS_NS

#endif
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_HYBRID
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

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
friend class AtomVecDielectric;
  friend class ComputeSpin;
  friend class FixAtomSwap;
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
  ~PairHybrid() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void setup() override;
  void finish() override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;
  void born_matrix(int, int, int, int, double, double, double, double &, double &) override;

  void modify_params(int narg, char **arg) override;
  double memory_usage() override;

  void compute_inner() override;
  void compute_middle() override;
  void compute_outer(int, int) override;
  void *extract(const char *, int &) override;
  void reset_dt() override;

  int check_ijtype(int, int, char *);

  void add_tally_callback(class Compute *) override;
  void del_tally_callback(class Compute *) override;

 public:
  int nstyles;             // # of sub-styles
  Pair **styles;           // list of Pair style classes
  double *cutmax_style;    // max cutoff for each style
  char **keywords;         // style name of each Pair style
  int *multiple;           // 0 if style used once, else Mth instance

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
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CUT_COUL_LONG
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

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
  ~PairLJCutCoulLong() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;

  void compute_inner() override;
  void compute_middle() override;
  void compute_outer(int, int) override;
  void *extract(const char *, int &) override;

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
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CUT_COUL_CUT
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

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
  ~PairLJCutCoulCut() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;
  void *extract(const char *, int &) override;

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
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CHARMM_COUL_LONG
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

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
  ~PairLJCharmmCoulLong() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;

  void compute_inner() override;
  void compute_middle() override;
  void compute_outer(int, int) override;
  void *extract(const char *, int &) override;

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
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CHARMM_COUL_CHARMM
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

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
  ~PairLJCharmmCoulCharmm() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;
  void *extract(const char *, int &) override;

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
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CHARMMFSW_COUL_LONG
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/charmmfsw/coul/long,PairLJCharmmfswCoulLong);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CHARMMFSW_COUL_LONG_H
#define LMP_PAIR_LJ_CHARMMFSW_COUL_LONG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCharmmfswCoulLong : public Pair {
 public:
  PairLJCharmmfswCoulLong(class LAMMPS *);
  ~PairLJCharmmfswCoulLong() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;

  void compute_inner() override;
  void compute_middle() override;
  void compute_outer(int, int) override;
  void *extract(const char *, int &) override;

 public:
  int implicit;
  int dihedflag;

  double cut_lj_inner, cut_lj, cut_ljinv, cut_lj_innerinv;
  double cut_lj_innersq, cut_ljsq;
  double cut_lj3inv, cut_lj_inner3inv, cut_lj3, cut_lj_inner3;
  double cut_lj6inv, cut_lj_inner6inv, cut_lj6, cut_lj_inner6;
  double cut_coul, cut_coulsq;
  double cut_bothsq;
  double denom_lj, denom_lj12, denom_lj6;
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
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CUT_COUL_LONG_OMP
#endif

/*----------------------------------------------------------*/

#ifdef _CRACKER_PAIR_LJ_CUT_COUL_LONG_GPU
#endif

/*----------------------------------------------------------*/
