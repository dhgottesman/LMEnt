#! /bin/sh
#SBATCH --job-name=save_batch_indices
#SBATCH --output=save_batch_indices_%a.out
#SBATCH --error=save_batch_indices_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --gpus=0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=gpu-research
#SBATCH --partition=cpu-killable
#SBATCH --array=1-6

python /home/morg/students/gottesman3/LMEnt/save_batch_indices.py --epoch ${SLURM_ARRAY_TASK_ID}
