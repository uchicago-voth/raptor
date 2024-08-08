/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
   
   splitted for MS-EVB by Yuxing Peng
   
------------------------------------------------------------------------- */

#ifdef KSPACE_CLASS

KSpaceStyle(evb_pppm,EVB_PPPM)

#else

#ifndef EVB_PPPM_H
#define EVB_PPPM_H

#include <mpi.h>
#ifdef FFT_SINGLE
typedef float FFT_SCALAR;
#define  MPI_FFT_SCALAR MPI_FLOAT
#else
typedef double FFT_SCALAR;
#define  MPI_FFT_SCALAR MPI_DOUBLE
#endif

#include "EVB_kspace.h"

namespace LAMMPS_NS {

class EVB_PPPM : public EVB_KSpace {
 public:
 
  /*********************************************************/
  /*********************************************************/
  /*********************************************************/
  int natm_env;
  
  int nx, ny, nz, mx, my, mz, nlocal;
  FFT_SCALAR dx, dy, dz, x0, y0, z0;
  FFT_SCALAR ekx, eky, ekz;
  
  double *q;
  double **x, **f;
  
  double dipole_env; // z-component of dipole of environment for slab correction
  double dipole_r2_env;

  double **part2grid_dr;
  FFT_SCALAR ***env_density_brick;
  
  void clear_density();
  void load_env_density();
  void poisson_energy(int);
  void reduce_ev(int,bool);

  void evb_setup();
  // ** AWGL : made these guys virtual for pppm/omp k-space style ** //
  virtual void compute_env(int);
  virtual void compute_env_density(int);
  virtual void compute_cplx(int);
  virtual void compute_cplx_eff(int); // Compute forces on cplx atoms in effective field of ENV
  virtual void compute_exch(int);
  virtual void compute_eff(int);
  virtual void map2density_one(int);
  virtual void map2density_one(int, int);
  virtual void map2density_one_subtract(int);
  virtual void field2force_one_ik(int, bool);
  virtual void field2force_one_ad(int, bool);
  LAMMPS * lmp_pointer;

  void sci_setup_iteration();
  void sci_setup_init() {};
  void sci_compute_env(int);
  void sci_compute_cplx(int);
  void sci_compute_exch(int);
  void sci_compute_eff(int);
  void sci_compute_eff_cplx(int);
  void sci_compute_eff_mp(int);
  void sci_compute_eff_cplx_mp(int);
  void poisson_mp(int, int, int, int);
  void poisson_ik_mp(int, int, int);
  
  /*********************************************************/
  /*********************************************************/
  /*********************************************************/
 
  EVB_PPPM(class LAMMPS*);
  virtual void settings(int, char **);
  virtual ~EVB_PPPM();
  virtual void init();
  virtual void setup();
  void setup_grid();
  virtual void compute(int, int);
  virtual int timing_1d(int, double &);
  virtual int timing_3d(int, double &);
  virtual double memory_usage();

  // GridComm
  FFT_SCALAR * cg_buf1;
  FFT_SCALAR * cg_buf2;
  int ngc_buf1, ngc_buf2, npergrid;

 protected:
  int me,nprocs;
  int nfactors;
  int *factors;
  double qsum,qsqsum,q2;
  double qqrd2e;
  double cutoff;
  double volume;
  double delxinv,delyinv,delzinv,delvolinv;
  double shift,shiftone;

  int nxlo_in,nylo_in,nzlo_in,nxhi_in,nyhi_in,nzhi_in;
  int nxlo_out,nylo_out,nzlo_out,nxhi_out,nyhi_out,nzhi_out;
  int nxlo_ghost,nxhi_ghost,nylo_ghost,nyhi_ghost,nzlo_ghost,nzhi_ghost;
  int nxlo_fft,nylo_fft,nzlo_fft,nxhi_fft,nyhi_fft,nzhi_fft;
  int nlower,nupper;
  int ngrid,nfft,nbuf,nfft_both;

  FFT_SCALAR ***density_brick;
  FFT_SCALAR ***vdx_brick,***vdy_brick,***vdz_brick;
  FFT_SCALAR ***u_brick;

  // ** AWGL : Additional stuff for OpenMP PPPM added ** //
  FFT_SCALAR ***v0_brick,***v1_brick,***v2_brick;
  FFT_SCALAR ***v3_brick,***v4_brick,***v5_brick;

  double *greensfn;
  double **vg;
  double *fkx,*fky,*fkz;
  FFT_SCALAR *density_fft;
  FFT_SCALAR *work1,*work2;
  FFT_SCALAR *buf1,*buf2; // DELETE ME

  double *gf_b;
  FFT_SCALAR **rho1d,**rho_coeff,**drho1d,**drho_coeff;
  double *sf_precoeff1, *sf_precoeff2, *sf_precoeff3;
  double *sf_precoeff4, *sf_precoeff5, *sf_precoeff6;
  double sf_coeff[6];          // coefficients for calculating ad self-forces
  double **acons;

  class FFT3d *fft1,*fft2;
  class Remap *remap;
  class GridComm *cg;

  int **part2grid;             // storage for particle -> grid mapping
  int nmax;

  int triclinic;               // domain settings, orthog or triclinic
  void setup_triclinic();
  void compute_gf_ik_triclinic();
  void poisson_ik_triclinic();
  
  double *boxlo;
                               // TIP4P settings
  int typeH,typeO;             // atom types of TIP4P water H and O atoms
  double qdist;                // distance from O site to negative charge
  double alpha;                // geometric factor

  int cr_N;                      // CR-HYDROXIDE settings
  double cr_height, cr_diameter;
  int typeB;                     // bond type
  double cr_qsqsum;              // qsqsum correction for charged-ring

  void set_grid_global();
  void set_grid_local();
  void adjust_gewald();
  double newton_raphson_f();
  double derivf();
  double final_accuracy();

  void field2force_one(int, bool) {}; // temporary
  void fillbrick() {}; // temporary

  virtual void allocate();
  virtual void deallocate();
  int factorable(int);
  double compute_df_kspace();
  double estimate_ik_error(double, double, bigint);
  double compute_qopt();
  void compute_gf_denom();
  virtual void compute_gf_ik();
  virtual void compute_gf_ad();
  void compute_sf_precoeff();

  virtual void particle_map();
  virtual void make_rho();
  virtual void brick2fft();

  virtual void poisson(int, int);
  virtual void poisson_ik(int);
  virtual void poisson_ad(int);

  virtual void fieldforce();
  virtual void fieldforce_ik();
  virtual void fieldforce_ad();

  // Used to calculate forces on only ENV atoms when using HF PPPM forces
  virtual void fieldforce_env();
  virtual void fieldforce_env_ik();
  virtual void fieldforce_env_ad();

  void procs2grid2d(int,int,int,int *, int*);
  void compute_rho1d(const FFT_SCALAR &, const FFT_SCALAR &,
		     const FFT_SCALAR &);
  void compute_drho1d(const FFT_SCALAR &, const FFT_SCALAR &,
		     const FFT_SCALAR &);
  void compute_rho_coeff();

  // Slab-correction
  void slabcorr_cplx();
  void slabcorr_exch();
  void slabcorr_sci_cplx();
  void slabcorr_sci_eff();

  // Eliminate redundant FFTs each MD step
  int sci_first_iteration_test;           // Whether first iteration computed by partition or not.
  int * do_sci_compute_cplx_other;        // Whether to compute self-energy of other complexes in sci_compute_cplx
  double * energy_sci_compute_cplx_other; // self-energy of other complexes in sci_compute_cplx

  int * do_sci_compute_cplx_self;        // Whether to compute self-energy in sci_compute_cplx
  double * energy_sci_compute_cplx_self; // self-energy in sci_compute_cplx

  // GridComm
  
  virtual void pack_forward_grid(int, void *, int, int *);
  virtual void unpack_forward_grid(int, void *, int, int *);
  virtual void pack_reverse_grid(int, void *, int, int *);
  virtual void unpack_reverse_grid(int, void *, int, int *);

/* ----------------------------------------------------------------------
   denominator for Hockney-Eastwood Green's function
     of x,y,z = sin(kx*deltax/2), etc

            inf                 n-1
   S(n,k) = Sum  W(k+pi*j)**2 = Sum b(l)*(z*z)**l
           j=-inf               l=0

          = -(z*z)**n /(2n-1)! * (d/dx)**(2n-1) cot(x)  at z = sin(x)
   gf_b = denominator expansion coeffs
------------------------------------------------------------------------- */

  inline double gf_denom(const double &x, const double &y,
                         const double &z) const {
    double sx,sy,sz;
    sz = sy = sx = 0.0;
    for (int l = order-1; l >= 0; l--) {
      sx = gf_b[l] + sx*x;
      sy = gf_b[l] + sy*y;
      sz = gf_b[l] + sz*z;
    }
    double s = sx*sy*sz;
    return s*s;
  };
};

}

#endif
#endif
