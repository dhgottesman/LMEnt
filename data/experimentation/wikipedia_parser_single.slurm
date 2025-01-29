#! /bin/sh
#SBATCH --job-name=wikipedia_parser_articles1.xml-p1p41242
#SBATCH --output=wikipedia_parser_articles1.xml-p1p41242.out
#SBATCH --error=wikipedia_parser_articles1.xml-p1p41242.err
#SBATCH -t 1-00:00:00
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p mi2104x

OUT_DIR="/work1/mgeva/dhgottesman/parsed_enwiki/"
mkdir $OUT_DIR

file="/work1/mgeva/dhgottesman/enwiki/enwiki-latest-pages-articles1.xml-p1p41242"

echo $file
python /work1/mgeva/dhgottesman/knowledge-analysis-suite/data/experimentation/wikipedia_parser.py --dump_file "$file" --output $OUT_DIR$(basename "$file" )
