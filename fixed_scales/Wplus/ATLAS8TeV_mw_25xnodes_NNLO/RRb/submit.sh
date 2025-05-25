#!/bin/bash
#

#SBATCH --job-name=RRbjob_highstat
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1

#SBATCH --cpus-per-task=1
#SBATCH --array=1-1500
#SBATCH --mem-per-cpu=2000
#SBATCH --output=log/log.main.%A.%a.%n.%x.%j.o.txt
#SBATCH --error=log/log.main.%A.%a.%n.%x.%j.e.txt

# Output environment information for debugging
env

echo "Now running: /ATLAS8TeV_mw_25xnodes_NNLO/RRb/job_node.sh"
./job_node.sh $SLURM_JOB_NAME $SLURM_ARRAY_TASK_ID  $SLURM_LOCLAID 
