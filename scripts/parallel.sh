#!/bin/bash
# To run this script, do 
# qsub -t 1-30 BASiCS_parallel.sh
#
# IDS is the list of ids to run the R script on
#
#$ -N nutpie_parallel
#$ -j y
#$ -S /bin/bash
#$ -cwd
#$ -pe sharedmem 10
#$ -l h_vmem=30G
#$ -l h_rt=24:00:00
#$ -o /exports/igmm/eddie/tchandra-lab/EJYang/ephemeral
#$ -e /exports/igmm/eddie/tchandra-lab/EJYang/ephemeral

unset MODULEPATH
. /etc/profile.d/modules.sh

module load anaconda/5.3.1
source activate NSEA_env

cd /exports/igmm/eddie/tchandra-lab/EJYang/methylclock/ProbAge/scripts
mylist=()
mylist+=("site_position")
# Use a loop to add numbers to the array
for ((i=0; i<=3000; i+=100)); do
  mylist+=("$i")
done

num1=${mylist[$SGE_TASK_ID]}
num2=${mylist[$SGE_TASK_ID+1]}

#data="wave4"
data="EWAS"

python nutpie_sampler_parallel.py $data $num1 $num2

