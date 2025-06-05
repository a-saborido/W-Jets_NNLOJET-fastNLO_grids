source ~/xfitter/tools/setup.sh
nohup xfitter &
xfitter-draw output_HERA+DY+absyj1 --bands --root --q2all (or '--q2 10' for example to plot only Q2=10)

------> Root files are saved in output/plots.root

rootls output/plots.root:Graphs

The script plot_pdf.C can be used for pdf plotting like 

root -l -b -q 'plot_pdf.C("uv",1.9,"output_HERA+DY/plots.root","uv","output_HERA+DY+absyj1/plots.root")'


Where the first argument "uv" refers to the pdf to plot, the econd argument (1.9) refers to the scale. The first output argument is the reference pdf. The other pdf is divided by the central value of the reference and displayed in a ratio plot.
