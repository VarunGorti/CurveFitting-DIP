#!/bin/bash
#SBATCH -A 'IFML'
#SBATCH -o 'DIP1'
#SBATCH -p 'v100'
#SBATCH -J 'DIP1'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 48:00:00
#SBATCH --mail-user 'sriram.ravula@utexas.edu' 
#SBATCH --mail-type 'ALL'
#SBATCH --cpus-per-task 10

echo "SLURM Job ID        : ${SLURM_JOB_ID}"

cd /scratch/04703/sravula/CurveFitting-DIP
module load conda
conda activate score2

srun python3 main.py