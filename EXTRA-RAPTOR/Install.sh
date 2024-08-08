#!/bin/bash

# Find the correct path of lammps/src

DES=".."

PKG_DIR=`pwd`
PKG_NAME=`basename $PKG_DIR`

if [ "$LAMMPS_SRC" != "" ]
then
  DES=$LAMMPS_SRC
fi

echo PWD=`pwd`
DES=`cd $DES; pwd`
echo DES=$DES

if [ ! -f $DES/lammps.h ]
then
  echo "ERROR: CAN NOT FIND THE LAMMPS SOURCE DIRECTORY."
  exit 
fi
  
# Some subroutines

update_file()
{
  des=$1
  file=$2
  filename=`basename $file`
  
  if [ ! -f $des/$filename ]; then cp -p $file $des; action='new-file'
  else 
    if [ $file -nt $des/$filename ]; then cp -p $file $des; action='update'
    else action="no-action"; fi
  fi

  if [ "$action" != "no-action" ]; then printf "  SYNC FILE %-32s [%-10s]\n" $file $action; fi
}

force_update_file_if_present()
{
  des=$1
  file=$2
  filename=`basename $file`
  
  if ! diff $filename $des/$filename >/dev/null ; then
    cp -p $file $des; action='forced-update'
  fi

  if [ "$action" != "no-action" ]; then printf "  SYNC FILE %-32s [%-10s]\n" $file $action; fi
}

check_module () 
{
  list=`grep -l $1 ./$2*.h`
  
  if (test -e EVB_module_$3.tmp) then
    rm -f EVB_module_$3.tmp
  fi
  
  for file in $list; do
    qfile="\"$file\""
    echo "#include $qfile" >> EVB_module_$3.tmp
  done
  
  if (test ! -e EVB_module_$3.tmp) then
    rm -f EVB_module_$3.h
    touch EVB_module_$3.h
  elif (test ! -e EVB_module_$3.h) then
    mv EVB_module_$3.tmp EVB_module_$3.h
  elif (test "`diff --brief EVB_module_$3.h EVB_module_$3.tmp`" != "") then
    mv EVB_module_$3.tmp EVB_module_$3.h
  else
    rm -f EVB_module_$3.tmp
  fi
}

crack_class ()
{
  src=$1

  echo "" >> EVB_cracker.h
  echo "#ifdef _CRACKER_$2" >> EVB_cracker.h
  if [ -f $src/$3 ]; then awk '{ if($1=="private:" || $1=="protected:") { print " public:" } else {print $0} }' $src/$3 >> EVB_cracker.h; fi
  echo "#endif" >> EVB_cracker.h
  echo "" >> EVB_cracker.h
  echo "/*----------------------------------------------------------*/" >> EVB_cracker.h
}

# Install ($1==1) ****************************************************

if (test $1 = 1) then

  # ****************************************************

  echo "[$PKG_NAME] Update timer constants if needed ..."

  cat *.cpp | grep TIMER_STAMP | awk 'BEGIN { FS="[(,),,,]"} { print $2,$3}' | sort | uniq | awk '{ printf "#define TIMER_%s_%s %d\n", $1, $2, NR-1} END {print "#define TIMER_COUNT",NR}' > tmp.dat
  DFF=`diff EVB_timer_const.h tmp.dat`

  if [ "${DFF}" = "" ]; then /bin/rm tmp.dat; else mv tmp.dat EVB_timer_const.h; fi
  
  # ****************************************************

  echo "[$PKG_NAME] Update EVB_cracker.h if needed ..."; 
  
  VER_LAMMPS=`head -n 1 $DES/version.h`
  if [ -f EVB_cracker.h ]; then VER_CURRENT=`head -n 1 EVB_cracker.h`
  else VER_CURRENT=NULL
  fi
  
  if [ "$VER_LAMMPS" != "$VER_CURRENT" ]
  then
    echo "Updating crackers ..."
    echo $VER_LAMMPS > EVB_cracker.h
    
    crack_class  $DES  KSPACE                     kspace.h
    crack_class  $DES  GRIDCOMM                   gridcomm.h
    crack_class  $DES  NEIGHBOR                   neighbor.h
    crack_class  $DES  INTEGRATE                  integrate.h
    crack_class  $DES  PAIR                       pair.h
    crack_class  $DES  PAIR_HYBRID                pair_hybrid.h
    crack_class  $DES  PAIR_LJ_CUT_COUL_LONG      pair_lj_cut_coul_long.h
    crack_class  $DES  PAIR_LJ_CUT_COUL_CUT       pair_lj_cut_coul_cut.h
    crack_class  $DES  PAIR_LJ_CHARMM_COUL_LONG   pair_lj_charmm_coul_long.h
    crack_class  $DES  PAIR_LJ_CHARMM_COUL_CHARMM pair_lj_charmm_coul_charmm.h
    crack_class  $DES  PAIR_LJ_CHARMMFSW_COUL_LONG pair_lj_charmmfsw_coul_long.h
    crack_class  $DES  PAIR_LJ_CUT_COUL_LONG_OMP  pair_lj_cut_coul_long_omp.h
    crack_class  $DES  PAIR_LJ_CUT_COUL_LONG_GPU  pair_lj_cut_coul_long_gpu.h
  fi
  
  # ****************************************************

  echo "[$PKG_NAME] Update module files if needed ..."; 
  
  check_module EVB_MODULE_OFFDIAG     EVB_offdiag_      offdiag      EVB_engine
  check_module EVB_MODULE_REP         EVB_rep_          rep          EVB_engine
  
  # ****************************************************

  echo "[$PKG_NAME] Install SRC files into $DES ..."; 
  
  FILE_LIST=`ls *.h *.cpp`
  for FILE in $FILE_LIST; do update_file $DES $FILE; done

  # Update RAPTOR-enabled COLVARS fix

  if [ -f $DES/fix_colvars.h ]; then
    cd colvars
    FILE_LIST=`ls *.h *.cpp`
    for FILE in $FILE_LIST; do force_update_file_if_present ../$DES $FILE; done
    cd ../
  fi

# Uninstall ($1==0) ****************************************************

 
elif (test $1 = 0) then

  echo "---> Uninstall SRC files in package $PKG_NAME ...";
  
  FILE_LIST=`ls *.h *.cpp`
  
  for FILE in $FILE_LIST; 
  do 
    rm -f $DES/$FILE
    
    dep=${FILE/.cpp/.d}
    obj=${FILE/.cpp/.o}

    if [ "$dep" != "$FILE" ]
    then
      rm -f $DES/Obj_*/$dep
      rm -f $DES/Obj_*/$obj
    fi
    
  done

  if [ ! -e $DES/multipro.cpp ]
  then
    rm $DES/mp_verlet.h
    rm $DES/Obj_*/mp_verlet.h
  else
    cp ../USER-MULTIPRO/depend/fix_evb.h $DES
    cp ../USER-MULTIPRO/depend/EVB_engine.h $DES
  fi
fi
