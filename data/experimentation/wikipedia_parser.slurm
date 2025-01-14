#! /bin/sh
#SBATCH --job-name=wikipedia_parser
#SBATCH --output=slurm_output/wikipedia_parser.out
#SBATCH --error=slurm_output/wikipedia_parser.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --time=1440
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --ntasks=1

filelist=$1

OUT_DIR="/home/morg/dataset/parsed_wiki_rerun/"
mkdir OUT_DIR
# Read the file list, unzip then run each file.
while IFS= read -r filename; do
	echo $filename
#	bzip2 -dk $filename
	python /home/morg/students/ohavbarbi/knowledge_analysis_suite/data/experimentation/wikipedia_parser.py --dump_file "$filename" --output $OUT_DIR$(basename "$filename" )
done < $filelist
