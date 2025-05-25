#_____CHANGE_THE_PATH_TO_LCG_SETUP_SCRIPT__________________
source /lcgenv-x86_64-centos7-gcc12-opt-LCG_102b.sh

cd combine_grid

# Define the observables list                                                                                                                                                                                     
observables=("ptj1" "ptw" "abs_yj1" "ht_full")

# Loop over each observable to merge the relevant files                                                                                                                                                           
for obs in "${observables[@]}"; do
    fnlo-tk-merge2 -add LO.${obs}.tab.gz V.${obs}.tab.gz R.${obs}.tab.gz NLO.${obs}.tab.gz
    fnlo-tk-merge2 -add LO.${obs}.tab.gz V.${obs}.tab.gz R.${obs}.tab.gz VV.${obs}.tab.gz RV.${obs}.tab.gz RRa.${obs}.tab.gz RRb.${obs}.tab.gz NNLO.${obs}.tab.gz
done

cp ../read_tab_gz.sh .
./read_tab_gz.sh
