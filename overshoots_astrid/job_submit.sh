#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=earth_cascades
#SBATCH --account=dominoes

#SBATCH --workdir=/p/projects/dominoes/nicowun/conceptual_tipping/uniform_distribution/overshoot_study/out_err_files
#SBATCH --output=outfile-%j.txt
#SBATCH --error=error-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-23:50:00

module load anaconda/5.0.0_py3
source activate earth_cascades

python /p/projects/dominoes/nicowun/conceptual_tipping/uniform_distribution/overshoot_study/MAIN_cluster_earth_system_complete_no_enso.py $SLURM_NTASKS 0.8451485300727355 6.954908706086238 2.7541616548054972 5.12923512097775 4.031849117513919 0.12044247597466146 0.8297385254986199 0.6306413024291091 0.16115075074426244 0.125233991633061 0.38621843447988247 0.13889173072011363 0.2816929849380546 0.13590374491339607 0.13348208470042439 0.1403179515650196 0.12507693814240828 8239.509524179897 232.24472957409026 2703.6664084220806 177.60179581871208 159.71462390019462 0099
