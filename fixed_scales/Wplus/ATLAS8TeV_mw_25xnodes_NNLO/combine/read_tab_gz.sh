#!/bin/bash
THISDIR=$PWD

for file in $THISDIR/*.tab.gz; do
    out=$file.txt.out
    touch $out
    fnlo-tk-cppread $file NNPDF31_nnlo_as_0118 7 LHAPDF no kScale1 > $out                                                                    
    echo "finished processing $out"
done
echo "done"
