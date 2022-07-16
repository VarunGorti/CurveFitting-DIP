#!/bin/bash
#SBATCH -A 'IFML'
#SBATCH -o 'DIP'
#SBATCH -p 'v100'
#SBATCH -J 'DIP'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH -t 48:00:00
#SBATCH --mail-user 'sriram.ravula@utexas.edu' 
#SBATCH --mail-type 'ALL'
#SBATCH --cpus-per-task 10

cd /scratch/04703/sravula/CurveFitting-DIP
module load conda
conda activate score2

srun python3 main.py