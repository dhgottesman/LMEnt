#! /bin/sh
#SBATCH --job-name=wikipedia_parser
#SBATCH --output=slurm_output/wikipedia_parser.out
#SBATCH --error=tmp_job/wikipedia_parser.err
#SBATCH --partition=gpu-h100-killable
#SBATCH --account=gpu-research
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --nodelist=n-100
python /home/morg/students/ohavbarbi/knowledge_analysis_suite/data/experimentation/wikipedia_parser.py

