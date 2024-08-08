/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(table/lj/cut/coul/long,PairTableLJCutCoulLong)

#else

#ifndef LMP_PAIR_TABLE_LJ_CUT_COUL_LONG_H
#define LMP_PAIR_TABLE_LJ_CUT_COUL_LONG_H

#include "pair.h"

namespace LAMMPS_NS {

  class PairTableLJCutCoulLong : public Pair {
  public:
    PairTableLJCutCoulLong(class LAMMPS *); 
    virtual ~PairTableLJCutCoulLong();
    virtual void compute(int, int);
    virtual void settings(int, char **);
    void coeff(int, char **);
    virtual void init_style();
    double init_one(int, int);
    void write_restart(FILE *);
    void read_restart(FILE *);
    virtual void write_restart_settings(FILE *);
    virtual void read_restart_settings(FILE *);
    virtual double single(int, int, int, int, double, double, double, double &);
    
    void *extract(const char *, int &);
    
  public:
    double cut_lj_global;
    double **cut_lj,**cut_ljsq;
    double cut_coul,cut_coulsq;
    double **epsilon,**sigma;
    double **lj1,**lj2,**lj3,**lj4,**offset;
    double *cut_respa;
    double qdist;
    double g_ewald;
    
    double tabinnersq;
    double *rtable,*drtable,*ftable,*dftable,*ctable,*dctable;
    double *etable,*detable,*ptable,*dptable,*vtable,*dvtable;
    int ncoulshiftbits,ncoulmask;
    
    void allocate();
    void init_tables();
    void free_tables();
    
    int tabstyle,tablength;
    enum {LOOKUP=0, LINEAR=1, SPLINE=2, BITMAP=3};
    struct Table {
      int ninput,rflag,fpflag,match,ntablebits;
      int nshiftbits,nmask;
      double rlo,rhi,fplo,fphi,cut;
      double *rfile,*efile,*ffile;
      double *e2file,*f2file;
      double innersq,delta,invdelta,deltasq6;
      double *rsq,*drsq,*e,*de,*f,*df,*e2,*f2;
    };
    int ntables;
    Table *tables;
    
    int **tabindex;
    int **tabflag;
    
    void read_table(Table *, char *, char *);
    void param_extract(Table *, char *);
    void bcast_table(Table *);
    void spline_table(Table *);
    void compute_table(Table *);
    void null_table(Table *);
    void free_table(Table *);
    void spline(double *, double *, int, double, double, double *);
    double splint(double *, double *, double *, int, double);

    // SCI-MS-EVB functions
    virtual double single_ener_noljcoul(int, int, int, int, double, double);
    virtual double single_fpair_noljcoul(int, int, int, int, double, double);
  };
  
}

#endif
#endif

/* ERROR/WARNING messages:

E: Pair distance < table inner cutoff

Two atoms are closer together than the pairwise table allows.

E: Pair distance > table outer cutoff

Two atoms are further apart than the pairwise table allows.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Unknown table style in pair_style command

Style of table is invalid for use with pair_style table command.

E: Illegal number of pair table entries

There must be at least 2 table entries.

E: Invalid pair table length

Length of read-in pair table is invalid

E: Invalid pair table cutoff

Cutoffs in pair_coeff command are not valid with read-in pair table.

E: Bitmapped table in file does not match requested table

Setting for bitmapped table in pair_coeff command must match table
in file exactly.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Cannot open file %s

The specified file cannot be opened.  Check that the path and name are
correct.

E: Did not find keyword in table file

Keyword used in pair_coeff command was not found in table file.

E: Bitmapped table is incorrect length in table file

Number of table entries is not a correct power of 2.

E: Invalid keyword in pair table parameters

Keyword used in list of table parameters is not recognized.

E: Pair table parameters did not set N

List of pair table parameters must include N setting.

E: Pair table cutoffs must all be equal to use with KSpace

When using pair style table with a long-range KSpace solver, the
cutoffs for all atom type pairs must all be the same, since the
long-range solver starts at that cutoff.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style lj/cut/coul/long requires atom attribute q

The atom style defined does not have this attribute.

E: Pair style is incompatible with KSpace style

If a pair style with a long-range Coulombic component is selected,
then a kspace style must also be used.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
