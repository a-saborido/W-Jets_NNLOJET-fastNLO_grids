#!/bin/bash
#

echo "This is: $0 $@, $1 $2 $3"
echo ""
env
echo ""

JOB_NAME=$1
ARRAY_ID=$2  # SLURM_ARRAY_TASK_ID

#SEED=$(printf "%03d%03d%02d" $SLURM_ARRAY_TASK_ID $SLURM_NODEID $proc )


NN_ARRAY=$(printf "%03d" $ARRAY_ID )
NN_PROC=$(printf "%02d"  $SLURM_PROCID )

SEED=$NN_ARRAY$NN_PROC

THISDIR=$PWD


WORKDIR=/tmp/$USER/$JOB_NAME/$NN_ARRAY/$NN_PROC
mkdir -p $WORKDIR

RUNCARD=WpJ.ATLAS8TeV.RRa.run
cp $THISDIR/*.wrm       $WORKDIR/.
cp $THISDIR/*.str $WORKDIR/.
cp $THISDIR/*00E-07*     $WORKDIR/.
cp $THISDIR/$RUNCARD    $WORKDIR/.

#_______EDIT_THE_PATH_TO_LCG_setup_FILE__________
source /lcgenv-x86_64-centos7-gcc12-opt-LCG_102b.sh

echo "Running NNLOJET..."
cd $WORKDIR

#____________EDIT_THE_PATH_TO_NNLOJET_____________
/NNLOJET_rev7902/driver/NNLOJET -run $RUNCARD -iseed $SEED




echo "... NNLOJET done."
mkdir -p $THISDIR/$JOB_NAME/$NN_ARRAY
cp -r $WORKDIR $THISDIR/$JOB_NAME/$NN_ARRAY/.
rm -rf $WORKDIR
