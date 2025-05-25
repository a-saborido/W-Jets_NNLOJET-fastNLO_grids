#!/bin/bash   
THISDIR=$PWD
#_____CHANGE_THE_PATH_TO_LCG_SETUP_SCRIPT__________________
source /lcgenv-x86_64-centos7-gcc12-opt-LCG_102b.sh

#COMBINE DAT FILES
#___________CHANGE_THE_PATH_TO_nnlojet-combine.py___________________
/nnlojet-combine.py -j 10 -C combine.ini  

#COMBINE GRID FILES
rm -rf combine_grid
mkdir combine_grid
cd combine_grid


#EDIT *APPLfast.txt FILES SO THAT *.dat NAME FORMAT IS CHANGED TO *.tab.gz FORMAT
# Define the observables list
observables=("ptj1" "ptw" "abs_yj1" "ht_full")

# Loop over each observable
for obs in "${observables[@]}"; do
    # Copy the weight files
    cp ../combined/Final/NNLO.${obs}_var.weights.txt .

    # Update the suffix and extension for each observable
    sed -i s/dat/tab.gz/g NNLO.${obs}_var.weights.txt
    sed -i s/.${obs}_var/.${obs}/g NNLO.${obs}_var.weights.txt

    # Run fnlo-tk-merge2 for each observable
    nohup nice fnlo-tk-merge2 -w NNLOJET NNLO.${obs}_var.weights.txt -f -o LO.${obs}.tab.gz ../../LO/*/*/*/W*J.*${obs}*.s*.tab.gz &> log.LO.${obs}_tab.txt &
    nohup nice fnlo-tk-merge2 -w NNLOJET NNLO.${obs}_var.weights.txt -f -o R.${obs}.tab.gz ../../R/*/*/*/W*J.*${obs}*.s*.tab.gz &> log.R.${obs}_tab.txt &
    nohup nice fnlo-tk-merge2 -w NNLOJET NNLO.${obs}_var.weights.txt -f -o V.${obs}.tab.gz ../../V/*/*/*/W*J.*${obs}*.s*.tab.gz &> log.V.${obs}_tab.txt &
    nohup nice fnlo-tk-merge2 -w NNLOJET NNLO.${obs}_var.weights.txt -f -o RV.${obs}.tab.gz ../../RV/*/*/*/W*J.*${obs}*.s*.tab.gz &> log.RV.${obs}_tab.txt &
    nohup nice fnlo-tk-merge2 -w NNLOJET NNLO.${obs}_var.weights.txt -f -o VV.${obs}.tab.gz ../../VV/*/*/*/W*J.*${obs}*.s*.tab.gz &> log.VV.${obs}_tab.txt &
    nohup nice fnlo-tk-merge2 -w NNLOJET NNLO.${obs}_var.weights.txt -f -o RRa.${obs}.tab.gz ../../RRa/*/*/*/W*J.*${obs}*.s*.tab.gz &> log.RRa.${obs}_tab.txt &
    nohup nice fnlo-tk-merge2 -w NNLOJET NNLO.${obs}_var.weights.txt -f -o RRb.${obs}.tab.gz ../../RRb/*/*/*/W*J.*${obs}*.s*.tab.gz &> log.RRb.${obs}_tab.txt &
done
