#! /bin/sh
#SBATCH --job-name=wp_19_19_20_20_20_21
#SBATCH --output=wp_19_19_20_20_20_21.out
#SBATCH --error=wp_19_19_20_20_20_21.err
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1               # Request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1             # Only 1 task needed
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4  # Allocate 4 CPUs per GPU task
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-morgeva

binary=/home/morg/students/gottesman3/knowledge-analysis-suite/data/experimentation/wikipedia_parser.py

OUT_DIR="/home/morg/dataset/parsed_enwiki/"

file1="/home/morg/dataset/enwiki/enwiki-latest-pages-articles19.xml-p27121851p28621850"
file2="/home/morg/dataset/enwiki/enwiki-latest-pages-articles19.xml-p30121851p31308442"
file3="/home/morg/dataset/enwiki/enwiki-latest-pages-articles20.xml-p31308443p32808442"
file4="/home/morg/dataset/enwiki/enwiki-latest-pages-articles20.xml-p32808443p34308442"
file5="/home/morg/dataset/enwiki/enwiki-latest-pages-articles20.xml-p34308443p35522432"
file6="/home/morg/dataset/enwiki/enwiki-latest-pages-articles21.xml-p35522433p37022432"

echo $file1
echo $file2
echo $file3
echo $file4
echo $file5
echo $file6

nohup python ${binary} --dump_file "$file1" --output $OUT_DIR$(basename "$file1") > output_19_p27121851p28621850.log 2>&1 &
nohup python ${binary} --dump_file "$file2" --output $OUT_DIR$(basename "$file2") > output_19_p30121851p31308442.log 2>&1 &
nohup python ${binary} --dump_file "$file3" --output $OUT_DIR$(basename "$file3") > output_20_p31308443p32808442.log 2>&1 &
nohup python ${binary} --dump_file "$file4" --output $OUT_DIR$(basename "$file4") > output_20_p32808443p34308442.log 2>&1 &
nohup python ${binary} --dump_file "$file5" --output $OUT_DIR$(basename "$file5") > output_20_p34308443p35522432.log 2>&1 &
nohup python ${binary} --dump_file "$file6" --output $OUT_DIR$(basename "$file6") > output_21_p35522433p37022432.log 2>&1 &

echo "Job steps submitted..."
sleep 1
squeue -u `id -un` -s

# Wait for all jobs to complete...
wait

echo "All Steps completed."
