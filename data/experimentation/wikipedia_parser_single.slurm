#! /bin/sh
#SBATCH --job-name=wikipedia_parser_articles1.xml-p1p41242
#SBATCH --output=slurm_output/wikipedia_parser_articles1.xml-p1p41242_3.out
#SBATCH --error=slurm_output/wikipedia_parser_articles1.xml-p1p41242_3.err
#SBATCH --partition=killable
#SBATCH --constraint="quadro_rtx_8000|a6000"
#SBATCH --account=gpu-research
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --ntasks=1

OUT_DIR="/home/morg/dataset/parsed_wiki_rerun_3/"
# mkdir $OUT_DIR
file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles1.xml-p1p41242"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles2.xml-p41243p151573"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles3.xml-p151574p311329"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles4.xml-p311330p558391"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles5.xml-p558392p958045"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles6.xml-p958046p1483661"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles7.xml-p1483662p2134111"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles8.xml-p2134112p2936260"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles9.xml-p2936261p4045402"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles10.xml-p4045403p5399366"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles11.xml-p6899367p7054859"
# file="/home/morg/dataset/enwiki/enwiki-latest-pages-articles11.xml-p5399367p6899366"

echo $file
python /home/morg/students/gottesman3/knowledge-analysis-suite/data/experimentation/wikipedia_parser.py --dump_file "$file" --output $OUT_DIR$(basename "$file" )

# # 1) Test if the gzip file is valid
# if ! gunzip -t "$file" 2>/dev/null; then
#   echo "File '$file' is corrupt or has unexpected EOF. Repairing..."
#   zcat "$file" > tmp.jsonl
#   gzip tmp.jsonl
#   mv -f tmp.jsonl.gz "$file"
#   echo "File '$file' successfully repaired."
# else
#   echo "File '$file' is OK (no unexpected EOF). No action needed."
# fi

# temp_file="${file}.cleaned"

# # Check if the last line is invalid JSON
# if ! zcat "$file" | tail -n 1 | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
#     echo "Last JSON line is invalid. Writing cleaned data to $temp_file..."

#     # Decompress, remove the last line, and recompress to the temp file
#     zcat "$file" | head -n -1 | gzip > "$temp_file"

#     echo "Invalid JSON line removed. Cleaned file created at: $temp_file"
#     mv "$temp_file" "$file"
    
#     echo "Cleaned file moved to: $file"
# else
#     echo "Last JSON line is valid. No action needed."
# fi

