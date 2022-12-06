#!/bin/bash
#SBATCH -A 'IFML'
#SBATCH -o 'VF'
#SBATCH -p 'v100'
#SBATCH -J 'VF'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH --mail-user 'sriram.ravula@utexas.edu' 
#SBATCH --mail-type 'ALL'

echo "SLURM Job ID        : ${SLURM_JOB_ID}"

cd /scratch/04703/sravula/CurveFitting-DIP
module load conda
conda activate devito-base

python3 run_vf.py